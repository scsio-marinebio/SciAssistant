# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""
Planner Agent for Multi-Agent Task Coordination

This agent serves as a coordinator for complex tasks that require multiple agents
working together. It implements the ReAct pattern for reasoning and action.
"""
import time
import logging
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
try:
    import litellm
except ImportError:
    litellm = None
from concurrent.futures import ThreadPoolExecutor, as_completed

# Base imports
from .base_agent import BaseAgent, AgentConfig, AgentResponse, WriterAgentTaskInput
from config.config import (
    build_llm_request_body,
    build_model_request_headers,
    extract_reasoning_from_response,
    extract_tool_calls_from_response,
    get_tool_call_format_instruction,
    get_tool_schemas_prompt,
)
# Import agent creators for built-in task assignment
from .writer_agent import create_writer_agent
from .. import (
    clear_thread_session,
    get_thread_human_in_loop_phase2,
    get_thread_search_sources,
    get_thread_session_id,
    get_thread_workspace_path,
    inherit_thread_session,
)


class PlannerAgent(BaseAgent):
    """
    PlannerAgent coordinates multiple agents to handle complex user queries.
    
    The agent uses the ReAct pattern (Reasoning + Acting) to analyze user requests,
    break them down into manageable tasks, and coordinate the appropriate agents
    to complete the work.
    """

    MAX_UNSAFE_GENERATION_CORRECTIONS = 2
    MAX_TOOL_CALLS_PER_GENERATION = 24
    MAX_TOOL_PROTOCOL_CORRECTIONS = 5
    UNSAFE_GENERATION_CORRECTION_PROMPT = (
        "上一轮 Planner 响应因工具调用数量异常而被系统整体丢弃，其中没有任何工具调用被执行。"
        "不要假设上一轮的任何步骤已经完成。请只生成当前下一步真正需要的工具调用，"
        "优先合并同类操作，不要重复查询工作区，并将本轮工具调用控制在 8 个以内。 /no_think"
    )

    def __init__(self, config: AgentConfig = None, shared_mcp_client=None, task_id: Optional[str] = None):
        # Set default agent name if not specified
        if config and not config.agent_name:
            config.agent_name = "PlannerAgent"
        elif not config:
            config = AgentConfig(agent_name="PlannerAgent")

        super().__init__(config, shared_mcp_client)

        # Planner-specific state
        self.execution_plan = []
        self.task_queue = []
		
		# Task management for cancellation support
        self.task_id = task_id
        self._cancellation_token = None
        
        # Progress callback for SSE streaming
        self.progress_callback = None
		
        # Add built-in task assignment methods to available tools
        self._add_builtin_assignment_tools()

        # Regenerate tool schemas with built-in assignment tools
        self.tool_schemas = self._build_tool_schemas()

        self.sub_agent_configs = {}

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _send_progress(self, stage: str, message: str, details: dict = None):
        """发送进度更新"""
        self.logger.info(f"[PROGRESS] _send_progress called: stage={stage}, message={message}, has_callback={self.progress_callback is not None}, task_id={self.task_id}")
        if self.progress_callback and self.task_id:
            # 统一设置task_type为writing，所有任务都显示详细进度
            details = details or {}
            if 'task_type' not in details:
                details['task_type'] = 'writing'
            
            progress_data = {
                'type': 'progress',
                'stage': stage,
                'message': message,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'details': details
            }
            self.logger.info(f"[PROGRESS] Progress data: {progress_data}")
            self.logger.info(f"[PROGRESS] Calling progress_callback with task_id={self.task_id}")
            self.progress_callback(self.task_id, progress_data)
        else:
            self.logger.warning(f"[PROGRESS] Progress not sent: has_callback={self.progress_callback is not None}, task_id={self.task_id}")
    
    def _add_builtin_assignment_tools(self):
        """Add built-in task assignment methods as available tools"""
        # Add assignment methods that share the MCP client connection
        self.available_tools.update({
            "assign_subjective_task_to_writer": self.assign_subjective_task_to_writer, # assign_subjective_task_to_writer
            "assign_multi_objective_tasks_to_info_seeker": self.assign_multi_objective_tasks_to_info_seeker,
            "assign_multi_subjective_tasks_to_info_seeker": self.assign_multi_subjective_tasks_to_info_seeker
        })
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the planner agent"""
        tool_schemas_str = get_tool_schemas_prompt(self.tool_schemas)
        tool_call_format = get_tool_call_format_instruction()

        # Use pre-detected language flag to make EXPLICIT language instruction
        _is_cn = getattr(self, '_is_chinese_query', False)
        if _is_cn:
            lang_instruction = """## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**你正在处理中文查询，必须使用中文回复。**
这条规则适用于所有输出：任务规划、最终答案、摘要等任何交付给用户的内容。
**禁止使用英文、韩文、日文或其他语言进行解释性文字。**
技术术语可以保留英文，但所有解释性文字必须使用中文。"""
        else:
            lang_instruction = """## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**You are handling an English query and MUST respond in English ONLY.**
This rule applies to ALL outputs: task planning, final answers, summaries, and any content delivered to the user.
**DO NOT use Chinese, Korean, Japanese, or any other language.**
Under NO circumstances should you produce Chinese text in your response.
All reasoning, planning, and outputs MUST be in English exclusively."""

        auto_system_prompt_template = """# PlannerAgent: Multi-Agent Task Coordinator

$lang_instruction

**Role:** Analyze complex queries, first distinguish query type (long-form writing type/objective question type), then create structured plans, and coordinate specialized agents to deliver comprehensive solutions—call corresponding tools based on query type, and only invoke writer for long-form writing type queries.

#### Available Sub-Agents:  
- **`information_seeker`**: Research, data gathering, web search (supports single/parallel multi-task; long-form writing type uses assign_multi_subjective_tasks_to_info_seeker, other types use assign_multi_objective_tasks_to_info_seeker)
- **`writer`**: Only invoke this sub-agent when long-form writing is required. 

---

## Optimized Workflow
### 1. Query Type Judgment & Analysis & Planning Phase
**Goal:** Use the `think` tool to analyze the problem and determine whether it is a simple task (refers to tasks that do not require calling the information search agent or tool) or a complex task (requires calling info seeker). If it is a complex task, it is necessary to further analyze whether it is a objective question（do not require calling the writer agent）or a long-form writing question (requires long-form expression and need to call the writer agent later).
- **Simple Tasks:** For simple tasks that do not require info seeker invocation, you can directly call the `planner_objective_task_done` tool and write the answer in `final_answer` field without creating a todo.md file.
- **Complex Tasks:**  
  - For objective tasks, must use `assign_multi_objective_tasks_to_info_seeker`
  - For long-form writing tasks, must use `assign_multi_subjective_tasks_to_info_seeker`, and call the writer agent to integrate the collected information to generate a very long text
  - **Task Decomposition Rules:** 
    - Construct a task tree with a tree-like structure, where the root node represents the user's input query. Each subtask is marked with its depth in the task tree, and the entire task tree is executed from shallow to deep. Tasks at the same depth in the task tree must be independent and can be executed in parallel (via `assign_multi_xxx_tasks_to_info_seeker`) without mutual dependencies.
    - At the first level of the task tree, it is essential to thoroughly design subtasks that can be executed in parallel to explore various potential background information, thereby providing more specific clues for the next step of planning.
    - Competitive Redundancy Mechanism:
      - For key subtasks that have a significant impact on subsequent reasoning and planning, a redundancy mechanism should be established. This involves duplicating the task at the same depth level in the task tree, enabling the parallel execution of nearly identical tasks to enhance the completion rate and robustness of the task execution.
  - **Task Parallel Sending Requirements:**
    - When using `assign_multi_xxx_tasks_to_info_seeker`, all parallel-sent subtasks must be independent of each other; the description of each subtask must not contain any mutual references or dependency requirements for other subtasks.
    - There is no sequential execution relationship among all parallel-sent subtasks.

  - **Mandatory Documentation:** Create and write `todo.md` (e.g., `todo_v1.md`) with fields:  
    ```markdown
    # Task Planning Document
    ## task_name: [Clear identifier]
    ## task_desc: [Detailed requirements - focus on WHAT not HOW]
    ## deliverable_contents: [Exact output format specs]
    ## success_criteria: [Measurable 100% completion metrics]
    ## context: [Background, constraints, prior results]
    ## task_steps_for_reference: [Tree-structured preliminary execution plan, tag tasks with the depth in task tree `[DEPTH:xx]`]
    ```  

### 2. Execution & Iteration Phase
#### A. Unified Iteration Triggers (Shared by Both Types)
- Based on upper-layer task results, refine the next layer of planning and document it in a new version of `todo.md` (e.g., `todo_v2.md`).  
- If upper-layer tasks fail/encounter challenges: Invoke the `reflect` tool for introspection (no new information acquired, only saves thoughts), adjust the plan, and re-invoke the corresponding `information_seeker` method (objective: `assign_multi_objective_tasks_to_info_seeker`; long-form writing: `assign_multi_subjective_tasks_to_info_seeker`).  
- If current tasks require prior round information: Clearly specify the context of each task and referenced files (e.g., `./data/agent_output_v1.json`) when calling `information_seeker`.  
- Decompose and refine clues from upper-layer results, then execute verification in parallel.  

#### B. Query-Type-Specific Operations
- **Objective tasks**: No additional operations (strictly no writer invocation). Continue iterating until information meets `success_criteria`.  
- **Long-form writing tasks**: Add **information sufficiency check before writer invocation**:  
  1. Evaluate collected information from two dimensions: quantity (e.g., "Enough case studies for 3 chapters") and comprehensiveness (e.g., "Covers both positive and negative impacts of AI on education").  
  2. If information is insufficient: Adjust subtask directions (e.g., "Supplement AI education failure cases") and re-invoke `assign_multi_subjective_tasks_to_info_seeker` for targeted collection.  
  3. If information is sufficient: Invoke the writer via `assign_subjective_task_to_writer` (provide all collected materials and `todo.md` as context).  
  4. If the writer returns an incomplete result: Do not assist in completing it; only feed back the current completion status to the user.  

### 3. Completion & Synthesis Phase
#### A. Unified Validation & Integration (Shared by Both Types)
- **Validation**: Cross-check multi-source `information_seeker` outputs for consistency (e.g., "NBS and World Bank GDP data differ by ≤1%").  
- **Integration**: Combine parallel outputs into a unified deliverable (e.g., "Merge two GDP data sources into a single table" or "Integrate writer’s report with supplementary case studies").  
- **Delivery**: Output language must match the user’s query language (e.g., Chinese query → Chinese deliverable).  

#### B. Query-Type-Specific Task Completion (Critical)
- **Objective tasks**: Call the `planner_objective_task_done` tool **only when** all planned tasks are completed and the final deliverable (e.g., verified data, clear answers) is ready for user delivery.  
- **Long-form writing tasks**: Call the `planner_subjective_task_done` tool **only when** the writer has finished executing and the final long-form content meets the `success_criteria` in `todo.md`.  

---

## Critical Protocols
1. **Dependency Management:**  
    - Prohibit parallel dispatch for sequential dependent tasks unless using competitive redundancy mechanism
    - Convert sequential chains to parallel where possible (e.g., Hypothesis_A vs Hypothesis_B testing)  
2. **File Traceability:**  
    - All output references use relative paths (`./data/agent_output_1.json`)  
    - Version `todo.md` after each iteration (e.g., `todo_v2.md`)
3. **Local File Reading Recommendations:**
    - For files crawled natively, it is not recommended to directly use the `file_read` tool to read the entire content (maybe too long). Instead, the `document_qa` tool should be used to extract and verify the required information.
    - For task deliverables and summary documents from sub-agents, the `file_read` tool can be used to read them.
4. The final deliverable presented to the user should be consistent with the language used in the user's question.
5. **Writer invocation**: Strictly prohibit calling the writer for objective tasks; for long-form writing tasks, **never directly answer based on collected information**—must invoke the writer to generate the final long-form content.

Below, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
$tool_schemas
</tools>
For each function call, return a JSON object with function name and arguments:
$tool_call_format"""
# For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
# [unused11][{\"name\": <function name>, \"arguments\": <args json object>}][unused12]"""

        writing_system_prompt_template = """### PlannerAgent: Multi-Agent Task Coordinator

$lang_instruction

**Role:** Analyze complex queries, create structured plans, and coordinate specialized agents to deliver comprehensive solutions.  

#### Available Sub-Agents:  
- **`information_seeker`**: Research, data gathering, web search (supports single/parallel multi-task)  
- **`writer`**: Creates content (e.g., reports, analysis, etc.), and synthesizes from existing materials

---

### Optimized Workflow  
#### 1. Analysis & Planning Phase  
**Goal:** Writing mode is already selected: deliver a long-form report through the writer agent. Assess the query's complexity only to determine research scope and depth, not whether to write a report. Distinguish subject-driven and objective-driven research needs to plan appropriate subtasks.
- **All Tasks:** Create a task plan, gather sufficient supporting material, and invoke `assign_subjective_task_to_writer`. Even a simple or specific question requires a report in writing mode; do not finish with a direct answer or an invented report path.
- **Complex Tasks:**  
  - For Objective-driven tasks, Adopt *diverge-converge* strategy:  
    1. Use `assign_multi_subjective_tasks_to_info_seeker` call for divergent background research  
    2. Converge findings to define specific sub-problems  
  - For Subject-driven tasks, Adopt *multi-perspective* strategy:
    1. Use assign_multi_subjective_tasks_to_info_seeker call for divergent multi-source exploration (each task targets independent dimensions)
    2. Converge findings to define focused sub-problems addressing distinct knowledge gaps
    3. When the information seeker collects information, start to call the writer agent to integrate the collected information to generate a very long text
  - **Task Decomposition Rules:**  
    - Construct a task tree with a tree-like structure, where the root node represents the user's input query. Each subtask is marked with its depth in the task tree, and the entire task tree is executed from shallow to deep. Tasks at the same depth in the task tree must be independent and can be executed in parallel (via `assign_multi_subjective_tasks_to_info_seeker`) without mutual dependencies.
    - At the first level of the task tree, it is essential to thoroughly design subtasks that can be executed in parallel to explore various potential background information, thereby providing more specific clues for the next step of planning.
    - Competitive Redundancy Mechanism:
      - For key subtasks that have a significant impact on subsequent reasoning and planning, a redundancy mechanism should be established. This involves duplicating the task at the same depth level in the task tree, enabling the parallel execution of nearly identical tasks to enhance the completion rate and robustness of the task execution.
  - **Task Parallel Sending Requirements:**
    - When using `assign_multi_subjective_tasks_to_info_seeker`, all parallel-sent subtasks must be independent of each other; the description of each subtask must not contain any mutual references or dependency requirements for other subtasks.
    - There is no sequential execution relationship among all parallel-sent subtasks.

  - **Mandatory Documentation:** Create and write `todo.md` (e.g., `todo_v1.md`) with fields:  
    ```markdown
    # Task Planning Document
    ## task_name: [Clear identifier]
    ## task_desc: [Detailed requirements - focus on WHAT not HOW]
    ## deliverable_contents: [Exact output format specs]
    ## success_criteria: [Measurable 100% completion metrics]
    ## context: [Background, constraints, prior results]
    ## task_steps_for_reference: [Tree-structured preliminary execution plan, tag tasks with the depth in task tree `[DEPTH:xx]`]
    ```  

#### 2. Execution & Iteration Phase
- **Iteration Triggers:**
  - Based on the execution results of the upper layer of the task tree, specify and refine the next layer and subsequent task planning, and document them in a new `todo.md` file (e.g., `todo_v2.md`).
  - If there are tasks in the previous layer that have failed or encountered challenges, it is necessary to invoke `reflect` for introspection, consider more possibilities, and make new task planning and invoke `assign_multi_subjective_tasks_to_info_seeker` again. 
  - If the tasks sent in the current round require reference to task information from previous rounds, it is essential to clearly specify the context of each task and the files that may need to be used or referenced when calling `assign_multi_subjective_tasks_to_info_seeker`.
  - For the multiple clues of the execution results from the previous layer, they should be decomposed and refined, and executed in parallel for verification.
- **Information check required before calling writer:**  
  - Before invoking writer, analyze collected information for sufficiency: evaluate both quantity and comprehensiveness to ensure adequate material for long article generation
  - If information is insufficient, adjust subtask direction and initiate additional targeted information collection
- **When information is sufficient, invoke writer agent** via `assign_subjective_task_to_writer`
  - **CRITICAL:** You MUST collect ALL key_files returned by ALL completed information_seeker subtasks and pass them to the writer agent in the key_files parameter
  - Each key_file should include the file_path field containing the exact relative path returned by information_seeker
  - Do NOT filter or select files - pass ALL key_files from ALL subtasks to ensure the writer has complete information

#### 3. Completion & Synthesis Phase  
- **Validation:** Cross-check multi-source outputs for consistency, and Check whether the information source is sufficient
- **Integration:** Combine parallel outputs into unified deliverable  
- **Delivery:** Output language must match user's query language  
- When the writer agent is finished executing, planner_subjective_task_done tool needs to be called to end the current task

---

### Critical Protocols  
1. **Dependency Management:**  
   - Prohibit parallel dispatch for sequential dependent tasks unless using competitive redundancy mechanism
   - Convert sequential chains to parallel where possible (e.g., Hypothesis_A vs Hypothesis_B testing)
2. **File Traceability:**  
   - All output references use relative paths (`./data/agent_output_1.json`)  
   - Version `todo.md` after each iteration (e.g., `todo_v2.md`)  
3. **Iteration Discipline:**  
   - Minimum 2 parallel agents for critical hypothesis-validation tasks  
   - Terminate only when ALL success criteria are met at 100%  
5. **Usage of Think Tool:**
   - `think` is a systematic tool. After receiving the response from the complex tool or before invoking any other tools, you must **first invoke the `think` tool**: to deeply reflect on the results of previous tool invocations (if any), and to thoroughly consider and plan the user's task. The `think` tool does not acquire new information; it only saves your thoughts into memory.
6. **Usage of Reflect Tool:**
    `reflect` is a systematic tool. When encountering a failure in tool execution, it is necessary to invoke the reflect tool to conduct a review and revise the task plan. It does not acquire new information; it only saves your thoughts into memory.
7. Always prioritize complete solutions over partial delivery. Use parallel redundancy for critical path tasks, and convert agent disagreements into new parallel investigation branches.
8. **CRITICAL:** When you determine that the information_seeker has gathered sufficient information, you must invoke the writer agent to draft the final article in response to the user's query. You are not allowed to reply directly based on the collected information!
9.Also note that when the writing agent returns a result that shows it is not completed, you do not need to help it complete it further. You only need to feedback the current completion status to the user.

Below, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
$tool_schemas
</tools>
For each function call, return a JSON object with function name and arguments:
$tool_call_format"""
# For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
# [unused11][{\"name\": <function name>, \"arguments\": <args json object>}][unused12]"""

        qa_system_prompt_template = """### PlannerAgent: Multi-Agent Task Coordinator

$lang_instruction

**Role:** Analyze complex queries, create structured plans, and coordinate specialized agents to deliver comprehensive solutions.  

#### Available Sub-Agents:  
- **`information_seeker`**: Research, data gathering, web search (supports single/parallel multi-task)  

---

### Optimized Workflow  
#### 1. Analysis & Planning Phase  
**Goal:** Decompose problems into executable units with clear dependencies  
- **Simple Tasks:** For simple tasks that do not require sub-agent invocation, you can directly answer and call `planner_objective_task_done` without creating a todo.md file
- **Complex Tasks:**
  - **Task Decomposition Rules:**  
    - Construct a task tree with a tree-like structure, where the root node represents the user\'s input query. Each subtask is marked with its depth in the task tree, and the entire task tree is executed from shallow to deep. Tasks at the same depth in the task tree must be independent and can be executed in parallel (via `assign_multi_objective_tasks_to_info_seeker`) without mutual dependencies.
    - At the first level of the task tree, it is essential to thoroughly design subtasks that can be executed in parallel to explore various potential background information, thereby providing more specific clues for the next step of planning.
    - Competitive Redundancy Mechanism:
      - For key subtasks that have a significant impact on subsequent reasoning and planning, a redundancy mechanism should be established. This involves duplicating the task at the same depth level in the task tree, enabling the parallel execution of nearly identical tasks to enhance the completion rate and robustness of the task execution.
  - **Task Parallel Sending Requirements:**
    - When using `assign_multi_objective_tasks_to_info_seeker`, all parallel-sent subtasks must be independent of each other; the description of each subtask must not contain any mutual references or dependency requirements for other subtasks.
    - There is no sequential execution relationship among all parallel-sent subtasks.

  - **Mandatory Documentation:** Create and write `todo.md` (e.g., `todo_v1.md`) with fields:  
    ```markdown
    # Task Planning Document
    ## task_name: [Clear identifier]
    ## task_desc: [Detailed requirements - focus on WHAT not HOW]
    ## deliverable_contents: [Exact output format specs]
    ## success_criteria: [Measurable 100% completion metrics]
    ## context: [Background, constraints, prior results]
    ## task_steps_for_reference: [Tree-structured preliminary execution plan, tag tasks with the depth in task tree `[DEPTH:xx]`]
    ```  

#### 2. Execution & Iteration Phase
- **Iteration Triggers:**
  - Based on the execution results of the upper layer of the task tree, specify and refine the next layer and subsequent task planning, and document them in a new `todo.md` file (e.g., `todo_v2.md`).
  - If there are tasks in the previous layer that have failed or encountered challenges, it is necessary to invoke `reflect` for introspection, consider more possibilities, and make new task planning and invoke `assign_multi_objective_tasks_to_info_seeker` again. 
  - If the tasks sent in the current round require reference to task information from previous rounds, it is essential to clearly specify the context of each task and the files that may need to be used or referenced when calling `assign_multi_objective_tasks_to_info_seeker`.
  - For the multiple clues of the execution results from the previous layer, they should be decomposed and refined, and executed in parallel for verification.

#### 3. Completion & Synthesis Phase  
- **Validation:** Cross-check multi-source outputs for consistency
- **Integration:** Combine parallel outputs into unified deliverable  
- **Delivery:** Output language must match user\'s query language  
- **Task Completed:** The `planner_objective_task_done` can only be called when all planned tasks have been completed and the final results are ready to be delivered to the user.

---

### Critical Protocols  
1. **Dependency Management:**  
   - Prohibit parallel dispatch for sequential dependent tasks unless using competitive redundancy mechanism
   - Convert sequential chains to parallel where possible (e.g., Hypothesis_A vs Hypothesis_B testing)  
2. **File Traceability:**  
   - All output references use relative paths (`./data/agent_output_1.json`)  
   - Version `todo.md` after each iteration (e.g., `todo_v2.md`)
3. **Local File Reading Recommendations:**
    - For files crawled natively, it is not recommended to directly use the `file_read` tool to read the entire content (maybe too long). Instead, the `document_qa` tool should be used to extract and verify the required information.
    - For task deliverables and summary documents from sub-agents, the `file_read` tool can be used to read them.
4. The final deliverable presented to the user should be consistent with the language used in the user\'s question.

Below, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
$tool_schemas
</tools>
For each function call, return a JSON object with function name and arguments:
$tool_call_format"""
# For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
# [unused11][{\"name\": <function name>, \"arguments\": <args json object>}][unused12]"""

        planner_mode_system_prompt_map = {
            "auto": auto_system_prompt_template,
            "writing": writing_system_prompt_template,
            "qa": qa_system_prompt_template
        }

        # system_prompt = planner_mode_system_prompt_map[self.config.planner_mode].replace("$tool_schemas", tool_schemas_str)
        system_prompt = (
            planner_mode_system_prompt_map[self.config.planner_mode]
            .replace("$tool_schemas", tool_schemas_str)
            .replace("$tool_call_format", tool_call_format)
			.replace("$lang_instruction", lang_instruction)
        )

        return system_prompt

    def assign_multi_objective_tasks_to_info_seeker(
            self,
            tasks: List[Dict[str, str]],
            max_workers: int = 5
        ) -> Dict[str, Any]:
        """
        Creates multiple TaskInput objects and routes them to info_seeker agents for concurrent execution.
        This tool enables the PlannerAgent to assign multiple research tasks through the MCP tool interface.
        
        Args:
            tasks: List of task dictionaries with the following keys:
                - task_content (required): The specific task content
                - task_steps_for_reference: Optional reference steps for execution
                - deliverable_contents: Format of expected deliverable
                - acceptance_checking_criteria: Criteria for task completion and quality
                - workspace_id: Workspace ID for stored files and memory
                - current_task_status: Description of current task status
                
            max_workers: Maximum concurrent threads (default=4)
            
        Returns:
            MCPToolResult with execution results for all tasks
        """
        try:
            # 发送进度：开始信息搜集（包含子任务列表）
            subtask_names = [task.get('task_content', '')[:50] for task in tasks]  # 截取前50字符
            _info_msg = '正在搜集信息' if getattr(self, '_is_chinese_query', True) else 'Gathering information'
            self._send_progress('info_seeking', _info_msg, {
                'subtasks_count': len(tasks),
                'task_type': 'objective',
                'subtask_names': subtask_names
            })
            
            # Validate task count (1-4 tasks)
            if not (1 <= len(tasks) <= 5):
                return {
                    "success": False,
                    "error": f"Invalid task count ({len(tasks)}). Must assign 1~5 tasks. Please re-plan the task execution schedule or re-decompose the task."
                }
            
            # Import here to avoid circular imports
            try:
                from agents import TaskInput, create_objective_information_seeker
            except ImportError:
                from ..agents import TaskInput, create_objective_information_seeker
            
            results = []
            import threading
            lock = threading.Lock()
            parent_session_id = get_thread_session_id()
            parent_workspace_path = get_thread_workspace_path()
            parent_phase2 = get_thread_human_in_loop_phase2()
            parent_search_sources = get_thread_search_sources()
            
            def process_task(task: Dict[str, str]):
                """Process a single task with thread-safe result collection"""
                try:
                    inherit_thread_session(
                        parent_session_id,
                        parent_workspace_path,
                        parent_phase2,
                        parent_search_sources,
                    )
                    
                    
                    # Create TaskInput object
                    task_input = TaskInput(
                        task_content=task["task_content"],
                        task_steps_for_reference=task.get("task_steps_for_reference"),
                        deliverable_contents=task.get("deliverable_contents"),
                        current_task_status=task.get("current_task_status"),
                        workspace_id=None,  # Session/workspace is managed by the server; no need to set explicitly
                        acceptance_checking_criteria=task.get("acceptance_checking_criteria")
                    )
                    
                    # Create and execute with info seeker agent - use shared MCP client for session consistency
                    info_seeker_config = getattr(self, 'sub_agent_configs', {}).get('information_seeker', {})
                    info_seeker = create_objective_information_seeker(
                        model=info_seeker_config.get('model', self.config.model),
                        max_iterations=info_seeker_config.get('max_iterations', 30),
                        shared_mcp_client=self.mcp_tools.client if hasattr(self.mcp_tools, 'client') else self.mcp_tools
                    )
                    # 【关键修复】传递取消令牌给子 Agent
                    if self._cancellation_token:
                        info_seeker.set_cancellation_token(self._cancellation_token)
                    # 传递语言标志给子 Agent
                    info_seeker._is_chinese_query = getattr(self, '_is_chinese_query', True)

                    self.logger.info(f"Assigning task to InformationSeekerAgent: {task['task_content'][:8000]}...")


                    # Execute the task
                    response = info_seeker.execute_task(task_input)

                    if response.success:
                        response_data = {
                            "task_content": task.get("task_content", "Unknown task"),
                            "success": True,
                            "data": response.result,
                            "agent_name": response.agent_name,
                            "iterations": response.iterations,
                            "execution_time": response.execution_time,
                            # "reasoning_trace": response.reasoning_trace
                        }
                    else:
                        response_data = {
                            "task_content": task.get("task_content", "Unknown task"),
                            "success": False,
                            "error": response.error,
                            "agent_name": response.agent_name
                        }

                    # Thread-safe result collection
                    with lock:
                        results.append(response_data)

                    return response_data

                except Exception as e:
                    error_msg = f"Task processing failed: {str(e)}"
                    self.logger.error(error_msg)
                    with lock:
                        results.append({
                            "task_content": task.get("task_content", "Unknown task"),
                            "success": False,
                            "error": error_msg
                        })
                    return None
                finally:
                    clear_thread_session()
            
            # Execute tasks in parallel with thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_task, task) for task in tasks]
                # Wait for all tasks to complete
                for future in futures:
                    future.result()  # Raise exceptions if any
            
            # Check overall success
            all_success = all(task_result.get("success", False) for task_result in results)
            
            return {
                "success": all_success,
                "data": {"tasks": results},
                "error": None if all_success else "Some tasks failed",
                "metadata": {
                    "tool_name": "assign_multi_objective_tasks_to_info_seeker",
                    "task_count": len(tasks),
                    "success_count": sum(1 for r in results if r.get("success")),
                    "failure_count": sum(1 for r in results if not r.get("success"))
                }
            }
                
        except Exception as e:
            self.logger.error(f"Multi-task assignment failed: {e}")
            return {
                "success": False,
                "error": f"Multi-task assignment failed: {str(e)}"
            }
    

    def assign_multi_subjective_tasks_to_info_seeker(
            self,
            tasks: List[Dict[str, str]],
            max_workers: int = 5
        ) -> Dict[str, Any]:
        """
        Creates multiple TaskInput objects and routes them to info_seeker agents for concurrent execution.
        This tool enables the PlannerAgent to assign multiple research tasks through the MCP tool interface.
        
        Args:
            tasks: List of task dictionaries with the following keys:
                - task_content (required): The specific task content
                - task_steps_for_reference: Optional reference steps for execution
                - deliverable_contents: Format of expected deliverable
                - acceptance_checking_criteria: Criteria for task completion and quality
                - workspace_id: Workspace ID for stored files and memory
                - current_task_status: Description of current task status
                
            max_workers: Maximum concurrent threads (default=4)
            
        Returns:
            MCPToolResult with execution results for all tasks
        """
        try:
            # 发送进度：开始信息搜集（主观型，包含子任务列表）
            subtask_names = [task.get('task_content', '')[:100] + ('...' if len(task.get('task_content', '')) > 100 else '') for task in tasks]  # 截取前100字符
            _info_msg = '正在搜集信息' if getattr(self, '_is_chinese_query', True) else 'Gathering information'
            self._send_progress('info_seeking', _info_msg, {
                'subtasks_count': len(tasks),
                'task_type': 'subjective',
                'subtask_names': subtask_names
            })
            
            # Validate task count (1-4 tasks)
            if not (1 <= len(tasks) <= 6):
                return {
                    "success": False,
                    "error": f"Invalid task count ({len(tasks)}). Must assign 1-6 tasks."
                }

            # Import here to avoid circular imports
            try:
                from agents import TaskInput, create_subjective_information_seeker
            except ImportError:
                from ..agents import TaskInput, create_subjective_information_seeker

            results = []
            import threading
            lock = threading.Lock()
            parent_session_id = get_thread_session_id()
            parent_workspace_path = get_thread_workspace_path()
            parent_phase2 = get_thread_human_in_loop_phase2()
            parent_search_sources = get_thread_search_sources()

            def process_task(task: Dict[str, str]):
                """Process a single task with thread-safe result collection"""
                try:
                    inherit_thread_session(
                        parent_session_id,
                        parent_workspace_path,
                        parent_phase2,
                        parent_search_sources,
                    )
                    # Create TaskInput object
                    task_input = TaskInput(
                        task_content=task["task_content"],
                        task_steps_for_reference=task.get("task_steps_for_reference"),
                        deliverable_contents=task.get("deliverable_contents"),
                        current_task_status=task.get("current_task_status"),
                        workspace_id=self.get_session_info()["session_id"],  # Session/workspace is managed by the server; no need to set explicitly
                        acceptance_checking_criteria=task.get("acceptance_checking_criteria")
                    )

                    # Create and execute with info seeker agent - use shared MCP client for session consistency
                    info_seeker_config = getattr(self, 'sub_agent_configs', {}).get('information_seeker', {})
                    info_seeker = create_subjective_information_seeker(
                        model=info_seeker_config.get('model', self.config.model),
                        max_iterations=info_seeker_config.get('max_iterations', 30),
                        shared_mcp_client=self.mcp_tools.client if hasattr(self.mcp_tools, 'client') else self.mcp_tools
                    )
                    # 【关键修复】传递取消令牌给子 Agent
                    if self._cancellation_token:
                        info_seeker.set_cancellation_token(self._cancellation_token)
                    # 传递语言标志给子 Agent
                    info_seeker._is_chinese_query = getattr(self, '_is_chinese_query', True)

                    self.logger.info(f"Assigning task to InformationSeekerAgent: {task['task_content'][:8000]}...")

                    # Execute the task
                    response = info_seeker.execute_task(task_input)

                    if response.success:
                        response_data = {
                            "task_content": task.get("task_content", "Unknown task"),
                            "success": True,
                            "data": response.result,
                            "agent_name": response.agent_name,
                            "iterations": response.iterations,
                            "execution_time": response.execution_time,
                            # "reasoning_trace": response.reasoning_trace
                        }
                    else:
                        response_data = {
                            "task_content": task.get("task_content", "Unknown task"),
                            "success": False,
                            "error": response.error,
                            "agent_name": response.agent_name
                        }

                        # Thread-safe result collection
                    with lock:
                        results.append(response_data)

                    return response_data

                except Exception as e:
                    error_msg = f"Task processing failed: {str(e)}"
                    self.logger.error(error_msg)
                    with lock:
                        results.append({
                            "task_content": task.get("task_content", "Unknown task"),
                            "success": False,
                            "error": error_msg
                        })
                    return None
                finally:
                    clear_thread_session()

            # Execute tasks in parallel with thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_task, task) for task in tasks]
                # Wait for all tasks to complete
                for future in futures:
                    future.result()  # Raise exceptions if any

            # Check overall success
            all_success = all(task_result.get("success", False) for task_result in results)
            
            # 信息搜集完成后，发送简单的完成消息
            # 文件列表将在WriterAgent开始写作时推送
            if all_success:
                _completed_msg = '信息搜集完成' if getattr(self, '_is_chinese_query', True) else 'Information gathering completed'
                self._send_progress('info_seeking_completed', _completed_msg)
                self.logger.info(f"[PROGRESS] 信息搜集完成，文件列表将在写作开始时推送")

            return {
                "success": all_success,
                "data": {"tasks": results},
                "error": None if all_success else "Some tasks failed",
                "metadata": {
                    "tool_name": "assign_multi_subjective_tasks_to_info_seeker",
                    "task_count": len(tasks),
                    "success_count": sum(1 for r in results if r.get("success")),
                    "failure_count": sum(1 for r in results if not r.get("success"))
                }
            }

        except Exception as e:
            self.logger.error(f"Multi-task assignment failed: {e}")
            return {
                "success": False,
                "error": f"Multi-task assignment failed: {str(e)}"
            }

    def assign_subjective_task_to_writer(
            self,
            task_content: str,
            user_query: str,
            key_files: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Assign a writing or content creation task to the WriterAgent

        Args:
            task_content: Detailed description of the writing task to be performed
            user_query: List storing previous information seeker subtask summaries intact to preserve information from each completed research task
            key_files: Curated list of relevant files with file_path and desc for each file

        Returns:
            Dictionary with task assignment results
        """
        try:

            self.logger.info("Assigning task to WriterAgent")

            # Create task input
            task_input = WriterAgentTaskInput(
                task_content=task_content,
                user_query=user_query,
                key_files=key_files,
                workspace_id=self.get_session_info()["session_id"],
            )

            # Create writer agent with shared MCP client and sub-agent configuration
            writer_config = getattr(self, 'sub_agent_configs', {}).get('writer', {})
            writer = create_writer_agent(
                shared_mcp_client=self.mcp_tools.client,
                model=writer_config.get('model', self.config.model),
                max_iterations=writer_config.get('max_iterations', 20),
                temperature=writer_config.get('temperature', 0.3),
                # max_tokens=writer_config.get('max_tokens', 16384),
                max_tokens=writer_config.get('max_tokens', 8192),
                task_id=self.task_id
            )
            # 【关键修复】传递取消令牌给子 Agent
            if self._cancellation_token:
                writer.set_cancellation_token(self._cancellation_token)
            # 传递进度回调给子 Agent
            if self.progress_callback:
                writer.set_progress_callback(self.progress_callback)
            # 传递语言标志给子 Agent
            writer._is_chinese_query = getattr(self, '_is_chinese_query', True)

            self.logger.info(f"Assigning task to WriterAgent: {task_content[:800]}...")

            # 文件列表将由WriterAgent在过滤完文件后推送，确保显示的是实际使用的文件数量

            # Execute the task with shared connection
            response = writer.execute_task(task_input)

            if response.success:
                return {
                    "success": True,
                    "data": response.result,
                    "agent_name": response.agent_name,
                    "iterations": response.iterations,
                    "execution_time": response.execution_time,
                    "metadata": response.metadata,
                    # "reasoning_trace": response.reasoning_trace
                }
            else:
                return {
                    "success": False,
                    "error": response.error,
                    "agent_name": response.agent_name,
                    "metadata": response.metadata,
                }

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to assign task to WriterAgent: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Task assignment failed: {str(e)}"
            }

    @staticmethod
    def _is_terminal_ultra_writer_failure(tool_result: Dict[str, Any]) -> bool:
        """Classify only explicit, structured Ultra Writer terminal failures."""
        if not isinstance(tool_result, dict) or tool_result.get("success"):
            return False
        metadata = tool_result.get("metadata") or {}
        return (
            metadata.get("retryable") is False
            and metadata.get("error_type") in {
                "pangu_ultra_classifier_format_failure",
                "pangu_ultra_writer_fallback_exhausted",
                "pangu_ultra_writer_invocation_budget_exhausted",
            }
        )

    def _build_agent_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Build tool schemas for PlannerAgent using proper MCP architecture.
        Schemas come from MCP server via client, not direct imports.
        """

        # Get MCP tool schemas from server via client (proper MCP architecture)
        schemas = super()._build_agent_specific_tool_schemas()

        # Add schemas for built-in task assignment tools
        planner_mode_builtin_tools_map = {
            "auto": ["think", "reflect", "assign_multi_subjective_tasks_to_info_seeker", "assign_multi_objective_tasks_to_info_seeker", "assign_subjective_task_to_writer", "writer_subjective_task_done", "planner_subjective_task_done", "planner_objective_task_done"],
            "writing": ["think", "reflect", "assign_multi_subjective_tasks_to_info_seeker", "assign_subjective_task_to_writer", "writer_subjective_task_done", "planner_subjective_task_done"],
            "qa": ["think", "reflect", "assign_multi_objective_tasks_to_info_seeker", "planner_objective_task_done"],
        }
        builtin_assignment_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "think",
                    "description": "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Your thoughts."
                            }
                        },
                        "required": ["thought"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reflect",
                    "description": "When multiple attempts yield no progress, use this tool to reflect on previous reasoning and planning, considering possible overlooked clues and exploring more possibilities. It will not obtain new information or make any changes to the repository.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reflect": {
                                "type": "string",
                                "description": "The specific content of your reflection"
                            }
                        },
                        "required": ["reflect"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assign_multi_subjective_tasks_to_info_seeker",
                    "description": "Assign 1~6 research or information gathering tasks to different InformationSeekerAgents for parallel execution, each task descriptions must be semantically complete and clearly provide contextual information and potentially important reference documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "description": "List of tasks to be assigned to multiple InformationSeekerAgents",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "task_content": {
                                            "type": "string",
                                            "description": "Detailed description of the task to be performed"
                                        },
                                        "task_steps_for_reference": {
                                            "type": "string",
                                            "description": "Optional reference steps for task execution"
                                        },
                                        "deliverable_contents": {
                                            "type": "string",
                                            "description": "Expected format and content of deliverables"
                                        },
                                        "current_task_status": {
                                            "type": "string",
                                            "description": "Current status and context of the task, important documents that may be used and referenced"
                                        },
                                        "acceptance_checking_criteria": {
                                            "type": "string",
                                            "description": "Criteria for determining task completion and quality"
                                        },
                                    },
                                    "required": ["task_content"]
                                }
                            }
                        },
                        "required": ["tasks"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assign_multi_objective_tasks_to_info_seeker",
                    "description": "Assign 1~5 research or information gathering tasks to different InformationSeekerAgents for parallel execution, each task descriptions must be semantically complete and clearly provide contextual information and potentially important reference documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "description": "List of tasks to be assigned to multiple InformationSeekerAgents",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "task_content": {
                                            "type": "string",
                                            "description": "Detailed description of the task to be performed, the task description must be semantically complete"
                                        },
                                        "task_steps_for_reference": {
                                            "type": "string",
                                            "description": "Optional reference steps for task execution"
                                        },
                                        "deliverable_contents": {
                                            "type": "string",
                                            "description": "Expected format and content of deliverables"
                                        },
                                        "current_task_status": {
                                            "type": "string",
                                            "description": "Current status and context of the task, important documents that may be used and referenced"
                                        },
                                        "acceptance_checking_criteria": {
                                            "type": "string",
                                            "description": "Criteria for determining task completion and quality, and the requirements in the event of task completion failure"
                                        },
                                    },
                                    "required": ["task_content"]
                                }
                            }
                        },
                        "required": ["tasks"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assign_subjective_task_to_writer",
                    "description": "Assign a writing or content creation task to the WriterAgent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_query": {
                                "type": "string",
                                "description": "Pass in the original user question."
                            },
                            "task_content": {
                                "type": "string",
                                "description": "Integrate and synthesize provided materials to generate comprehensive long-form content exceeding 10,000 words, especially careful not to give specific details, such as an outline plan, you are only providing the writer with a general description of the task."
                            },
                            "key_files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {
                                            "type": "string",
                                            "description": "Relative path to the file containing research content"
                                        }
                                    },
                                    "required": ["file_path"]
                                },
                                "description": "Collect all key_files returned by the information seeker for long-form content creation."
                            }
                        },
                        "required": ["user_query", "task_content", "key_files"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "writer_subjective_task_done",
                    "description": "Writer Agent task completion reporting for complete long-form content. Called after all chapters/sections are written to provide a summary of the complete long article, final completion status and analysis, and the storage path of the final consolidated article.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "final_article_path": {
                                "type": "string",
                                "description": "The file path where the final article is saved."
                            },
                            "article_summary": {
                                "type": "string",
                                "description": "Comprehensive summary of the complete long-form article, including main themes, key points covered, and overall narrative structure.",
                                "format": "markdown"
                            },
                            "completion_status": {
                                "type": "string",
                                "enum": ["completed", "partial", "failed"],
                                "description": "Final status of the complete long-form writing task"
                            },
                            "completion_analysis": {
                                "type": "string",
                                "description": "Analysis of the overall writing project completion including: assessment of article coherence and quality, evaluation of content organization and flow, identification of any challenges in the writing process, and overall evaluation of the long-form content creation success."
                            }
                        },
                        "required": ["final_article_path", "article_summary", "completion_status", "completion_analysis"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "planner_subjective_task_done",
                    "description": "When the writer agent is executed, the task done tool is called to end the planner's task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "final_article_path": {
                                "type": "string",
                                "description": "The file path where the final article is saved."
                            },
                            "task_summary": {
                                "type": "string",
                                "description": "This field is mainly used to describe the main content of the article, briefly summarize it, and finally indicate the path where the final article is saved. CRITICAL: Must use the SAME LANGUAGE as the user's query (Chinese query → Chinese summary, English query → English summary).",
                                "format": "markdown"
                            },
                            "task_name": {
                                "type": "string",
                                "description": "The name of the task currently assigned to the agent, usually with underscores (e.g., 'web_research_ai_trends')"
                            },
                            "completion_status": {
                                "type": "string",
                                "enum": ["completed", "partial", "failed"],
                                "description": "Final task status"
                            }
                        },
                        "required": ["final_article_path", "task_summary", "task_name", "completion_status"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "planner_objective_task_done",
                    "description": "Structured reporting of task completion details including summary, decisions, and final answer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_summary": {
                                "type": "string",
                                "description": "Comprehensive markdown covering what the agent was asked to do, steps taken, tools used, key findings, files created, challenges. CRITICAL: Must use the SAME LANGUAGE as the user's query (Chinese query → Chinese summary, English query → English summary).",
                                "format": "markdown"
                            },
                            "task_name": {
                                "type": "string",
                                "description": "The name of the task currently assigned to the agent, usually with underscores (e.g., 'web_research_ai_trends')"
                            },
                            "key_files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {
                                            "type": "string",
                                            "description": "Relative path to created/modified file"
                                        },
                                        "desc": {
                                            "type": "string",
                                            "description": "File contents and creation purpose"
                                        },
                                        "is_final_output_file": {
                                            "type": "boolean",
                                            "description": "Whether file is primary deliverable"
                                        }
                                    },
                                    "required": ["file_path", "desc", "is_final_output_file"]
                                },
                                "description": "List of key files generated or modified during the task, with their details."
                            },
                            "completion_status": {
                                "type": "string",
                                "enum": ["completed", "partial", "failed"],
                                "description": "Final task status"
                            },
                            "final_answer": {
                                "type": "string",
                                "description": "The final response displayed to the user. CRITICAL: Must use the SAME LANGUAGE as the user's query (Chinese query → Chinese answer, English query → English answer).",
                            }
                        },
                        "required": ["task_summary", "task_name", "key_files", "completion_status", "final_answer"]
                    }
                }
            },
        ]

        used_builtin_schemas = [schema for schema in builtin_assignment_schemas if schema["function"]["name"] in planner_mode_builtin_tools_map[self.config.planner_mode]]
        schemas.extend(used_builtin_schemas)

        return schemas

    def set_cancellation_token(self, cancellation_token):
        """
        Set the cancellation token for this agent
        设置此代理的取消令牌

        Args:
            cancellation_token: threading.Event object that will be set when task should be cancelled
        """
        self._cancellation_token = cancellation_token

    def _check_cancellation(self) -> bool:
        """
        Check if task has been cancelled
        检查任务是否已被取消

        Returns:
            True if task should be cancelled, False otherwise
        """
        if self._cancellation_token and self._cancellation_token.is_set():
            self.logger.info(f"Task {self.task_id} cancellation detected")
            return True
        return False

    def _execute_react_loop(self, initial_message: str, max_iterations: int = 20) -> Dict[str, Any]:
        """
        Execute the ReAct loop for planning tasks

        Args:
            initial_message: Initial message to start the planning process
            max_iterations: Maximum number of iterations to perform

        Returns:
            Dictionary with execution results and trace
        """
        start_time = time.time()
        try:
            # Reset trace for new task
            self.reset_trace()
            # Initialize conversation history
            conversation_history = []

            has_assigned_info_tasks = False
            forced_assignment_prompted = False
            writer_terminal_failure = None
            writer_invocation_count = 0

            # Build system prompt for planning
            system_prompt = self._build_system_prompt()
            localtime = time.localtime()
            # Add to conversation
            conversation_history.append({"role": "system", "content":f"当前时间为：{localtime}"+ system_prompt})
            conversation_history.append({"role": "user", "content": initial_message + " /no_think"})

            iteration = 0
            task_completed = False
            unsafe_generation_correction_count = 0
            tool_protocol_correction_count = 0
            writer_invoked = False

            # Get model endpoint configuration from env-backed config
            from config.config import get_config
            config = get_config()
            model_config = config.get_custom_llm_config()
            from config.config import get_model_provider, is_pangu_ultra_moe_compat_enabled
            writer_model_config = getattr(self, "sub_agent_configs", {}).get("writer", {})
            planner_model_name = writer_model_config.get("model") or model_config.get("model") or self.config.model
            ultra_compat_enabled = is_pangu_ultra_moe_compat_enabled(
                model_name=planner_model_name,
                provider=model_config.get("provider") or get_model_provider(planner_model_name),
                enabled=model_config.get("pangu_ultra_moe_compat_enabled", False),
            )
            ultra_writer_max_invocations = max(
                1,
                int(model_config.get("pangu_ultra_writer_max_invocations", 2)),
            )
            
            # Centralized LLM client usage
            from src.utils.llm_client import LLMEmptyContentError, llm_chat
            # ReAct Loop: Reasoning -> Acting -> Reasoning -> Acting...
            while iteration < self.config.max_iterations and not task_completed:
                # 【关键修复】检查任务是否被取消
                if self._check_cancellation():
                    self.logger.warning(f"Task {self.task_id} was cancelled at iteration {iteration}")
                    return {
                        "success": False,
                        "error": "Task was cancelled by user",
                        "reasoning_trace": self.reasoning_trace,
                        "iterations": iteration,
                        "execution_time": time.time() - start_time
                    }
                
                iteration += 1
                self.logger.info(f"Planning iteration {iteration}")

                try:
                    # Only constrain the final four iterations after successful
                    # information gathering. Normal planning/research behavior
                    # remains unchanged while the iteration budget is healthy.
                    from src.utils.planner_budget import get_planner_convergence_policy
                    convergence_instruction, allowed_convergence_tools = (
                        get_planner_convergence_policy(
                            planner_mode=self.config.planner_mode,
                            total_iterations=self.config.max_iterations,
                            current_iteration=iteration,
                            has_assigned_info_tasks=has_assigned_info_tasks,
                            writer_invoked=writer_invoked,
                        )
                    )
                    if convergence_instruction:
                        conversation_history.append({
                            "role": "user",
                            "content": convergence_instruction,
                        })

                    # 【取消检查】在LLM调用前检查取消状态
                    if self._check_cancellation():
                        self.logger.info(f"Task {self.task_id} cancelled before LLM call at iteration {iteration}")
                        return {
                            "success": False,
                            "error": "Task was cancelled by user",
                            "reasoning_trace": self.reasoning_trace,
                            "iterations": iteration,
                            "execution_time": time.time() - start_time
                        }
                    
                    # Get LLM response (reasoning + potential tool calls)
                    # 【取消检查】在每次重试前检查
                    if self._check_cancellation():
                        self.logger.info(f"Task {self.task_id} cancelled during LLM retry preparation")
                        return {
                            "success": False,
                            "error": "Task was cancelled by user",
                            "reasoning_trace": self.reasoning_trace,
                            "iterations": iteration,
                            "execution_time": time.time() - start_time
                        }

                    llm_turn = llm_chat(
                        conversation_history,
                        model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=model_config.get("timeout", 180),
                        max_retries=10,
                        tool_schemas=self.tool_schemas,
                        tool_call_mode=model_config.get("tool_call_mode"),
                        return_tool_turn=True,
                    )
                    llm_turn = self._coerce_llm_tool_turn(llm_turn)
                    assistant_text = llm_turn.content
                    assistant_message = {"content": assistant_text}

                    # Log the reasoning
                    try:
                        reasoning_source = llm_turn.reasoning_content or assistant_message["content"]
                        if reasoning_source:
                            # reasoning_content = assistant_message["content"].split("[unused16]")[-1].split("[unused17]")[0]
                            reasoning_content = extract_reasoning_from_response(reasoning_source)
                            if len(reasoning_content) > 0:
                                self.log_reasoning(iteration, reasoning_content)
                    except Exception as e:
                        self.logger.warning(f"Tool call parsing error: {e}")
                        # Parse error, rerun
                        followup_prompt = f"There is a problem with the format of model generation: {e}. Please try again."
                        conversation_history.append({"role": "user", "content": followup_prompt + " /no_think"})
                        continue

                    # def extract_tool_calls(content):
                    #     import re
                    #     if not content:
                    #         return []
                    #     tool_call_str = re.findall(r"\[unused11\]([\s\S]*?)\[unused12\]", content)
                    #     if len(tool_call_str) > 0:
                    #         try:
                    #             tool_calls = json.loads(tool_call_str[0].strip())
                    #         except:
                    #             return []
                    #     else:
                    #         return []
                    #     return tool_calls

                    # tool_calls = extract_tool_calls(assistant_message["content"])
                    if llm_turn.tool_call_mode == "native":
                        tool_calls = llm_turn.tool_calls
                    else:
                        tool_calls = extract_tool_calls_from_response(
                            assistant_message["content"],
                            tool_schemas=self.tool_schemas,
                            api_profile=model_config.get("api_profile"),
                        )

                    if len(tool_calls) > self.MAX_TOOL_CALLS_PER_GENERATION:
                        if (
                            unsafe_generation_correction_count
                            < self.MAX_UNSAFE_GENERATION_CORRECTIONS
                            and iteration < self.config.max_iterations
                        ):
                            unsafe_generation_correction_count += 1
                            self.logger.warning(
                                "Planner discarded generation with %s tool calls and requested "
                                "correction (%s/%s; limit=%s); no tools were executed",
                                len(tool_calls),
                                unsafe_generation_correction_count,
                                self.MAX_UNSAFE_GENERATION_CORRECTIONS,
                                self.MAX_TOOL_CALLS_PER_GENERATION,
                            )
                            conversation_history.append({
                                "role": "user",
                                "content": self.UNSAFE_GENERATION_CORRECTION_PROMPT,
                            })
                            continue

                        if (
                            unsafe_generation_correction_count
                            >= self.MAX_UNSAFE_GENERATION_CORRECTIONS
                        ):
                            error_msg = (
                                "Planner tool-call batch remained unsafe after "
                                f"{self.MAX_UNSAFE_GENERATION_CORRECTIONS} corrective generations "
                                f"(received {len(tool_calls)}, limit "
                                f"{self.MAX_TOOL_CALLS_PER_GENERATION})"
                            )
                        else:
                            error_msg = (
                                "Planner tool-call batch exceeded the per-generation limit and "
                                "no iteration remained for a corrective generation "
                                f"(received {len(tool_calls)}, limit "
                                f"{self.MAX_TOOL_CALLS_PER_GENERATION})"
                            )
                        self.log_error(iteration, error_msg)
                        return {
                            "success": False,
                            "error": error_msg,
                            "reasoning_trace": self.reasoning_trace,
                            "iterations": iteration,
                            "execution_time": time.time() - start_time,
                        }

                    # Keep a rollback point. If every call in this generation
                    # fails schema validation, replace the malformed response
                    # and verbose tool errors with one compact correction.
                    generation_history_start = len(conversation_history)
                    self._append_assistant_tool_turn(
                        conversation_history,
                        llm_turn,
                        assistant_message["content"],
                    )
                    generation_tool_results = []

                    if tool_calls:
                        parsed_names = [call.get("name") for call in tool_calls if isinstance(call, dict)]
                        self.logger.info(f"Parsed tool calls: {parsed_names}")
                    else:
                        preview = (assistant_message["content"] or "").strip()
                        preview = preview[:2000] + ("..." if len(preview) > 2000 else "")
                        self.logger.warning(f"No tool calls parsed. Assistant content preview: {preview}")

                    # Execute tool calls if any (Acting phase)

                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict) or "name" not in tool_call or "arguments" not in tool_call:
                            continue
                        # 【取消检查】在每个工具调用前检查取消状态
                        if self._check_cancellation():
                            self.logger.info(f"Task {self.task_id} cancelled during tool execution loop")
                            return {
                                "success": False,
                                "error": "Task was cancelled by user",
                                "reasoning_trace": self.reasoning_trace,
                                "iterations": iteration,
                                "execution_time": time.time() - start_time
                            }
                        
                        arguments = tool_call["arguments"]
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except Exception:
                                arguments = {"raw_arguments": arguments}
                        tool_call["arguments"] = arguments
                        self.logger.debug(f"Arguments is string: {isinstance(arguments, str)}")

                        # Validate every tool uniformly. Completion and
                        # think/reflect used to bypass BaseAgent validation,
                        # allowing malformed terminal calls or polluted text
                        # arguments to be treated as successful actions.
                        validation_error = self._validate_tool_call_arguments(
                            tool_call["name"],
                            arguments,
                        )

                        # Check if planning is complete only after validation.
                        if validation_error:
                            tool_result = validation_error
                        elif (
                            allowed_convergence_tools is not None
                            and tool_call["name"] not in allowed_convergence_tools
                        ):
                            tool_result = {
                                "success": False,
                                "error_code": "PLANNER_CONVERGENCE_TOOL_BLOCKED",
                                "error": (
                                    f"Tool '{tool_call['name']}' is not allowed during Planner "
                                    "convergence. Use one of: "
                                    f"{sorted(allowed_convergence_tools)}"
                                ),
                                "tool_name": tool_call["name"],
                                "retryable": True,
                            }
                        elif tool_call["name"] in ["planner_subjective_task_done", "planner_objective_task_done", "writer_subjective_task_done"]:
                            if (
                                self.config.planner_mode == "writing"
                                and not writer_invoked
                            ):
                                tool_result = {
                                    "success": False,
                                    "error_code": "PLANNER_WRITER_NOT_DELIVERED",
                                    "error": (
                                        "Writing mode cannot complete before Writer successfully "
                                        "delivers its result. Invoke assign_subjective_task_to_writer."
                                    ),
                                    "retryable": True,
                                }
                            else:
                                task_completed = True
                                self.log_action(iteration, tool_call["name"], arguments, arguments)
                                generation_tool_results.append((tool_call["name"], arguments))
                                break
                        elif tool_call["name"] in ["think", "reflect"]:
                            tool_result = {"tool_results": "You can proceed to invoke other tools if needed. "}
                        else:
                            # 【取消检查】在执行工具前最后检查一次
                            if self._check_cancellation():
                                self.logger.info(f"Task {self.task_id} cancelled before executing tool {tool_call['name']}")
                                return {
                                    "success": False,
                                    "error": "Task was cancelled by user",
                                    "reasoning_trace": self.reasoning_trace,
                                    "iterations": iteration,
                                    "execution_time": time.time() - start_time
                                }
                            
                            if (
                                ultra_compat_enabled
                                and tool_call["name"] == "assign_subjective_task_to_writer"
                                and writer_terminal_failure is not None
                            ):
                                tool_result = {
                                    "success": False,
                                    "error": (
                                        "Writer terminal failure already recorded for this task; "
                                        "duplicate Writer invocation was blocked."
                                    ),
                                    "metadata": dict(writer_terminal_failure),
                                }
                                self.logger.warning(
                                    "Blocked duplicate Ultra Writer invocation after terminal failure: %s",
                                    writer_terminal_failure.get("error_type"),
                                )
                            elif (
                                ultra_compat_enabled
                                and tool_call["name"] == "assign_subjective_task_to_writer"
                                and writer_invocation_count >= ultra_writer_max_invocations
                            ):
                                writer_terminal_failure = {
                                    "error_type": "pangu_ultra_writer_invocation_budget_exhausted",
                                    "retryable": False,
                                    "stage": "planner",
                                    "writer_invocations": writer_invocation_count,
                                }
                                tool_result = {
                                    "success": False,
                                    "error": "Pangu Ultra Writer invocation budget exhausted.",
                                    "metadata": dict(writer_terminal_failure),
                                }
                                self.logger.warning(
                                    "Blocked Ultra Writer invocation after budget %s was exhausted.",
                                    ultra_writer_max_invocations,
                                )
                            else:
                                if tool_call["name"] == "assign_subjective_task_to_writer":
                                    writer_invocation_count += 1
                                tool_result = self.execute_tool_call(tool_call)
                                if (
                                    tool_call["name"] == "assign_subjective_task_to_writer"
                                    and isinstance(tool_result, dict)
                                    and tool_result.get("success") is True
                                ):
                                    writer_invoked = True

                            if ultra_compat_enabled and tool_call["name"] == "assign_subjective_task_to_writer":
                                writer_metadata = tool_result.get("metadata") or {}
                                if self._is_terminal_ultra_writer_failure(tool_result):
                                    writer_terminal_failure = dict(writer_metadata)

                            if (
                                tool_call["name"] in [
                                    "assign_multi_subjective_tasks_to_info_seeker",
                                    "assign_multi_objective_tasks_to_info_seeker",
                                ]
                                and isinstance(tool_result, dict)
                                and tool_result.get("success") is True
                            ):
                                has_assigned_info_tasks = True
                            
                            # 【取消检查】工具执行后立即检查
                            if self._check_cancellation():
                                self.logger.info(f"Task {self.task_id} cancelled after executing tool {tool_call['name']}")
                                return {
                                    "success": False,
                                    "error": "Task was cancelled by user",
                                    "reasoning_trace": self.reasoning_trace,
                                    "iterations": iteration,
                                    "execution_time": time.time() - start_time
                                }

                        # Log the action using base class method
                        self.log_action(iteration, tool_call["name"], arguments, tool_result)
                        generation_tool_results.append((tool_call["name"], tool_result))

                        # Add tool result to conversation
                        self._append_tool_result_turn(
                            conversation_history,
                            tool_call,
                            tool_result,
                            llm_turn.tool_call_mode,
                        )

                    validation_only_generation = bool(generation_tool_results) and all(
                        isinstance(result, dict)
                        and result.get("error_code") in {
                            "TOOL_ARGUMENT_VALIDATION_FAILED",
                            "TOOL_CALL_PROTOCOL_FAILED",
                        }
                        for _, result in generation_tool_results
                    )
                    if validation_only_generation:
                        tool_protocol_correction_count += 1
                        # The malformed assistant text may contain thousands of
                        # tokens and broken tool-result markers. Do not feed it
                        # back to the model; preserve it only in raw logs/trace.
                        del conversation_history[generation_history_start:]

                        first_tool, first_error = generation_tool_results[0]
                        target_tool = first_error.get("suggested_tool_name") or first_tool
                        input_schema = self._get_tool_input_schema(target_tool)
                        properties = input_schema.get("properties", {})
                        if not isinstance(properties, dict):
                            properties = {}
                        contract = {
                            name: (
                                schema.get("type", "any")
                                if isinstance(schema, dict)
                                else "any"
                            )
                            for name, schema in properties.items()
                        }
                        correction_protocol = (
                            "请只重新通过 API 原生 function calling 调用该工具；"
                            "不要在 message.content 中输出 <tool_call>、Markdown JSON 或标签。"
                            if llm_turn.tool_call_mode == "native"
                            else (
                                "请只重新生成一个该工具调用，使用完整格式 "
                                f"<tool_call>{target_tool}({{\"参数名\": \"参数值\"}})</tool_call>。"
                                "数组和对象必须输出为真正的 JSON，不要放在字符串中；"
                                "不要省略 arg_key/arg_value 标签，不要拼接工具返回结果。"
                            )
                        )
                        correction_prompt = (
                            "上一轮工具调用未执行，因为工具名或参数格式不符合 Schema。"
                            f"原始工具：{first_tool}；应使用工具：{target_tool}；参数类型："
                            f"{json.dumps(contract, ensure_ascii=False)}；"
                            f"缺失参数：{first_error.get('missing_required_fields', [])}；"
                            f"非法参数：{first_error.get('invalid_fields', [])}。"
                            f"{correction_protocol} /no_think"
                        )

                        if (
                            tool_protocol_correction_count
                            >= self.MAX_TOOL_PROTOCOL_CORRECTIONS
                            or iteration >= self.config.max_iterations
                        ):
                            error_msg = (
                                "Planner stopped after repeated malformed tool arguments "
                                f"({tool_protocol_correction_count} consecutive corrective "
                                f"generations; last tool={first_tool})"
                            )
                            self.log_error(iteration, error_msg)
                            return {
                                "success": False,
                                "error": error_msg,
                                "reasoning_trace": self.reasoning_trace,
                                "iterations": iteration,
                                "execution_time": time.time() - start_time,
                            }

                        self.logger.warning(
                            "Planner discarded validation-only generation and requested compact "
                            "tool protocol correction (%s/%s; tool=%s)",
                            tool_protocol_correction_count,
                            self.MAX_TOOL_PROTOCOL_CORRECTIONS,
                            first_tool,
                        )
                        conversation_history.append({
                            "role": "user",
                            "content": correction_prompt,
                        })
                        continue
                    elif generation_tool_results:
                        tool_protocol_correction_count = 0

                    # If no tool calls, encourage continued planning
                    if len(tool_calls) == 0:
                        # Add follow-up prompt to encourage action or completion
                        followup_prompt = (
                            "Continue your planning process. Use available tools to assign tasks to agents, "
                            "search for information, or coordinate work. When you have a complete answer, "
                            "call planner_subjective_task_done or planner_objective_task_done. /no_think"
                        )
                        if self.config.planner_mode == "writing":
                            followup_prompt = (
                                "Writing mode requires a report delivered by the Writer; a direct answer "
                                "does not complete this task. Create a plan and gather sufficient supporting "
                                "material, then invoke assign_subjective_task_to_writer. If Writer has already "
                                "successfully delivered its result, call planner_subjective_task_done. /no_think"
                            )
                        conversation_history.append({"role": "user", "content": followup_prompt})
                    elif (
                        not has_assigned_info_tasks
                        and not forced_assignment_prompted
                        and self.config.planner_mode in ["writing", "auto"]
                        and any(call.get("name") in ["file_write", "str_replace_based_edit_tool"] for call in tool_calls if isinstance(call, dict))
                    ):
                        forced_assignment_prompted = True
                        conversation_history.append({
                            "role": "user",
                            "content": (
                                "The planning document is ready. You must now call "
                                "assign_multi_subjective_tasks_to_info_seeker to dispatch 1-6 parallel research tasks. "
                                "Do not call file_write or str_replace_based_edit_tool again. /no_think"
                            ),
                        })

                except LLMEmptyContentError as e:
                    error_msg = (
                        "Planner stopped because the upstream LLM repeatedly returned empty "
                        f"content: {e}"
                    )
                    self.log_error(iteration, error_msg)
                    self.logger.error(
                        "Planner is terminating the current execution after empty-response "
                        "retries were exhausted; the outer task runner may retry the whole task"
                    )
                    return {
                        "success": False,
                        "error": error_msg,
                        "reasoning_trace": self.reasoning_trace,
                        "iterations": iteration,
                        "execution_time": time.time() - start_time,
                    }
                except Exception as e:
                    error_msg = f"Error in planning iteration {iteration}: {e}"
                    self.log_error(iteration, error_msg)
                    # 【关键修复】不要在异常时立即break，而是继续下一次迭代
                    # 这样可以给任务更多机会完成，避免因为临时网络问题导致任务失败
                    self.logger.warning(f"Iteration {iteration} failed, will retry in next iteration: {e}")
                    # 添加短暂延迟后继续
                    time.sleep(2)
                    continue

            execution_time = time.time() - start_time

            # Extract final result
            if task_completed:
                # Find the completion result in the trace
                completion_result = None
                for step in reversed(self.reasoning_trace):
                    if step.get("type") == "action" and step.get("tool") in ["planner_subjective_task_done",
                                                                             "planner_objective_task_done"]:
                        completion_result = step.get("result")
                        break

                return {
                    "success": True,
                    "data": completion_result,
                    "reasoning_trace": self.reasoning_trace,
                    "iterations": iteration,
                    "execution_time": execution_time
                }
            else:
                return {
                    "success": False,
                    "error": f"Planning task not completed within {max_iterations} iterations",
                    "reasoning_trace": self.reasoning_trace,
                    "iterations": iteration,
                    "execution_time": execution_time
                }
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            self.logger.error(f"Error in execute_react_loop: {e}")
            return {
                "success": False,
                "error": str(e),
                "reasoning_trace": self.reasoning_trace,
                "iterations": iteration if 'iteration' in locals() else 0,
                "execution_time": execution_time
            }


    def execute_task(self, user_query: str) -> AgentResponse:
        """
        Execute a planning task for the given user query

        Args:
            user_query: The user's query or request

        Returns:
            AgentResponse with planning results and process trace
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting planner task: {user_query}")
            
            # 检测用户查询语言
            import re as _re
            _zh_count = len(_re.findall(r'[\u4e00-\u9fff]', user_query))
            _total_chars = len(user_query.strip())
            # 只有当中文字符占比超过30%时才判定为中文查询
            self._is_chinese_query = (_zh_count / max(_total_chars, 1)) > 0.3
            
            # 发送初始进度（根据语言切换）
            _init_msg = '开始分析任务' if self._is_chinese_query else 'Analyzing task'
            self._send_progress('init', _init_msg, {'query': user_query[:100]})

            # Human in the loop 阶段2：跳过搜索，直接使用已有结果调用 WriterAgent
            human_in_loop_phase2 = get_thread_human_in_loop_phase2()
            if human_in_loop_phase2:
                self.logger.info("Human in the loop Phase 2: 跳过搜索阶段，直接调用 WriterAgent")
                
                # 读取 workspace 中已有的 key_files
                workspace_path = get_thread_workspace_path()
                key_files = []
                # 优先使用调用方显式注入的大纲（避免并发任务下全局环境变量串扰）
                user_outline = (getattr(self, "_hitl_user_outline", "") or "").strip()
                if workspace_path:
                    from pathlib import Path
                    workspace_path_obj = Path(workspace_path)
                    
                    # 回退：未注入时再从 workspace 读取用户确认大纲
                    if not user_outline:
                        user_outline_file = workspace_path_obj / '.user_outline'
                        if user_outline_file.exists():
                            with open(user_outline_file, 'r', encoding='utf-8') as f:
                                user_outline = f.read().strip()
                
                    self.logger.info(f"Human in the loop Phase 2: 读取用户确认大纲，长度 {len(user_outline)} 字符")
                    
                    # 读取已有的 key_files
                    file_analysis_path = workspace_path_obj / "doc_analysis" / "file_analysis.jsonl"
                    if file_analysis_path.exists():
                        with open(file_analysis_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    file_info = json.loads(line.strip())
                                    if file_info.get('file_path'):
                                        key_files.append({
                                            'file_path': file_info.get('file_path'),
                                            'desc': file_info.get('core_content', '')[:200]
                                        })
                                except:
                                    continue
                        self.logger.info(f"Human in the loop Phase 2: 加载了 {len(key_files)} 个已有文件")
                
                # 构建包含用户大纲的任务内容，明确指示 WriterAgent 跳过大纲生成步骤
                task_content = f"""【重要】这是 Human in the loop 阶段2，用户已确认大纲，请严格按照以下要求执行：

1. 【跳过大纲生成】不要自己生成大纲，直接使用下面用户确认的大纲
2. 【直接进行文件分类】调用 search_result_classifier 工具时，必须使用下面的用户确认大纲作为 outline 参数
3. 【按大纲写作】严格按照用户确认的大纲章节结构进行写作

用户确认的大纲：
{user_outline}

用户原始查询：{user_query}"""
                
                # 直接调用 WriterAgent
                result = self.assign_subjective_task_to_writer(
                    task_content=task_content,
                    user_query=user_query,
                    key_files=key_files
                )
                
                execution_time = time.time() - start_time
                
                return AgentResponse(
                    success=result.get("success", False),
                    result=result.get("data"),
                    error=result.get("error"),
                    reasoning_trace=[],
                    iterations=1,
                    execution_time=execution_time,
                    agent_name=self.config.agent_name
                )

            # Execute the planning task using ReAct pattern
            result = self._execute_react_loop(
                initial_message=user_query,
                max_iterations=self.config.max_iterations  # Reasonable limit for planning tasks
            )

            execution_time = time.time() - start_time

            return AgentResponse(
                success=result.get("success", False),
                result=result.get("data"),
                error=result.get("error"),
                reasoning_trace=result.get("reasoning_trace", []),
                iterations=result.get("iterations", 0),
                execution_time=execution_time,
                agent_name=self.config.agent_name
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Planner execution failed: {e}")

            return AgentResponse(
                success=False,
                error=f"Planner execution failed: {str(e)}",
                reasoning_trace=[],
                iterations=0,
                execution_time=execution_time,
                agent_name=self.config.agent_name
            )


def create_planner_agent(
        model: Any = None,
        sub_agent_configs: Dict[str, Dict[str, Any]] = None,
        shared_mcp_client=None,
        **kwargs
) -> PlannerAgent:
    """
    Create a PlannerAgent instance with server-managed sessions.
    
    Args:
        model: The LLM model to use
        sub_agent_configs: Configuration for sub-agents (information_seeker, writer)
        shared_mcp_client: Optional shared MCP client to prevent duplicate connections
        **kwargs: Additional configuration options
        
    Returns:
        Configured PlannerAgent instance
    """
    # Import the enhanced config function
    from .base_agent import create_agent_config

    # Handle agent_name if provided in kwargs
    agent_name = kwargs.pop("agent_name", "PlannerAgent")
    
    # Handle task_id if provided in kwargs
    task_id = kwargs.pop("task_id", None)

    # Create agent configuration (session managed by MCP server)
    config = create_agent_config(
        agent_name=agent_name,
        model=model,
        **kwargs
    )

    # Create planner agent with optional shared MCP client
    planner = PlannerAgent(config=config, shared_mcp_client=shared_mcp_client, task_id=task_id)

    # Store sub-agent configurations for use when creating sub-agents
    planner.sub_agent_configs = sub_agent_configs or {
        "information_seeker": {},
        "writer": {}
    }

    return planner
