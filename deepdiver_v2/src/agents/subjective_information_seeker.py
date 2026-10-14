# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
import json
from typing import Dict, Any, List, Optional
import time
import os
from .base_agent import BaseAgent, AgentConfig, AgentResponse, TaskInput
from .. import get_thread_search_source
from config.logging_config import get_logger
from config.config import (
    extract_reasoning_from_response,
    extract_tool_calls_from_response,
    get_tool_call_format_instruction,
    get_tool_schemas_prompt,
)
from src.utils.llm_client import LLMOutputTruncatedError, llm_chat
logger = get_logger()



class InformationSeekerAgent(BaseAgent):
    """
    Information Seeker Agent that follows ReAct pattern (Reasoning + Acting)
    
    This agent takes decomposed sub-questions or tasks from parent agents,
    thinks interleaved (reasoning -> action -> reasoning -> action),
    uses MCP tools to gather information, and returns structured results.
    """

    MAX_UNSAFE_GENERATION_CORRECTIONS = 2
    MAX_TOOL_CALLS_PER_GENERATION = 24
    UNSAFE_GENERATION_CORRECTION_PROMPT = (
        "上一轮响应因输出截断或工具调用数量异常而被系统整体丢弃，其中没有任何工具调用被执行。"
        "不要假设上一轮的任何步骤已经完成。请只生成当前下一步真正需要的工具调用，"
        "优先使用批量工具参数合并同类请求，不要展开大量重复调用，并将本轮工具调用控制在 8 个以内。 /no_think"
    )

    def __init__(self, config: AgentConfig = None, shared_mcp_client=None):
        # Set default agent name if not specified
        if config is None:
            config = AgentConfig(agent_name="InformationSeekerAgent")
        elif config.agent_name == "base_agent":
            config.agent_name = "InformationSeekerAgent"

        super().__init__(config, shared_mcp_client)

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
            self.logger.info("InformationSeekerAgent task cancellation detected")
            return True
        return False

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the ReAct agent"""
        tool_schemas_str = get_tool_schemas_prompt(self.tool_schemas)
        tool_call_format = get_tool_call_format_instruction()

        # Search-source preferences are request-scoped for concurrent tasks.
        use_websearch = get_thread_search_source('websearch')
        use_pubmed = get_thread_search_source('pubmed')
        use_arxiv = get_thread_search_source('arxiv')
        use_google_scholar = get_thread_search_source('google_scholar')
        use_scihub = get_thread_search_source('scihub')
        use_rag = get_thread_search_source('rag')
        # use_springer = os.environ.get('SEARCH_SOURCE_SPRINGER', 'True').lower() == 'true'  # DISABLED

        # Get all available tools from MCP
        # Tool schemas have structure: {'type': 'function', 'function': {'name': '...', ...}}
        available_tools = []
        for tool in self.tool_schemas:
            if isinstance(tool, dict):
                if 'function' in tool and isinstance(tool['function'], dict) and 'name' in tool['function']:
                    available_tools.append(tool['function']['name'])
                elif 'name' in tool:
                    available_tools.append(tool['name'])

        # Define tool category patterns (only need to maintain this mapping when adding new sources)
        tool_category_patterns = {
            'websearch': ['batch_web_search', 'web_search'],
            'pubmed': ['pubmed', 'medrxiv'],
            'arxiv': ['arxiv'],
            'google_scholar': ['google_scholar', 'scholar'],
            'scihub': ['scihub'],
			'rag': ['search_rag_knowledge', 'rag_knowledge'],
            # 'springer': ['springer']  # DISABLED
        }

        # Dynamically filter tools based on environment variables
        enabled_tools = []
        disabled_tools = []

        # Log environment variable values for debugging
        logger.info(f"[SEARCH_SOURCE_DEBUG] WebSearch={use_websearch}, PubMed={use_pubmed}, arXiv={use_arxiv}, GoogleScholar={use_google_scholar}, SciHub={use_scihub}, RAG={use_rag}")
        logger.info(f"[SEARCH_SOURCE_DEBUG] Available tools from MCP: {available_tools}")

        for tool_name in available_tools:
            tool_lower = tool_name.lower()
            is_enabled = False

            # Check if tool belongs to any enabled category
            if use_websearch and any(pattern in tool_lower for pattern in tool_category_patterns['websearch']):
                is_enabled = True
            elif use_pubmed and any(pattern in tool_lower for pattern in tool_category_patterns['pubmed']):
                is_enabled = True
            elif use_arxiv and any(pattern in tool_lower for pattern in tool_category_patterns['arxiv']):
                is_enabled = True
            elif use_google_scholar and any(pattern in tool_lower for pattern in tool_category_patterns['google_scholar']):
                is_enabled = True
            elif use_scihub and any(pattern in tool_lower for pattern in tool_category_patterns['scihub']):
                is_enabled = True
            elif use_rag and any(pattern in tool_lower for pattern in tool_category_patterns['rag']):
                is_enabled = True
                logger.info(f"[SEARCH_SOURCE_DEBUG] Tool '{tool_name}' matched RAG pattern and is_enabled={is_enabled}")
            # elif use_springer and any(pattern in tool_lower for pattern in tool_category_patterns['springer']):
            #     is_enabled = True
            #     logger.info(f"[SEARCH_SOURCE_DEBUG] Tool '{tool_name}' matched Springer pattern and is_enabled={is_enabled}")

            # Categorize tool
            if any(pattern in tool_lower for pattern in tool_category_patterns['websearch'] + tool_category_patterns['pubmed'] + tool_category_patterns['arxiv'] + tool_category_patterns['google_scholar'] + tool_category_patterns['scihub'] + tool_category_patterns['rag']):
                if is_enabled:
                    enabled_tools.append(tool_name)
                else:
                    disabled_tools.append(tool_name)

        logger.info(f"[SEARCH_SOURCE_DEBUG] Enabled tools: {enabled_tools}")
        logger.info(f"[SEARCH_SOURCE_DEBUG] Disabled tools: {disabled_tools}")

        # Build search source guidance message with priority strategy
        search_source_guidance = ""
        if enabled_tools:
            # Check if RAG is enabled
            has_rag = any('rag' in tool.lower() or 'search_rag_knowledge' in tool.lower() for tool in enabled_tools)
            # Check if other search sources are enabled (only check actual search tools, not document processing tools)
            search_tool_patterns = tool_category_patterns['websearch'] + tool_category_patterns['pubmed'] + tool_category_patterns['arxiv'] + tool_category_patterns['google_scholar']
            has_other_sources = any(
                tool for tool in enabled_tools
                if any(pattern in tool.lower() for pattern in search_tool_patterns)
            )
            # Categorize tools by type
            api_tools = [t for t in enabled_tools if any(p in t.lower() for p in ['arxiv', 'pubmed', 'medrxiv', 'google_scholar', 'scholar', 'scihub'])]
            web_tools = [t for t in enabled_tools if any(p in t.lower() for p in ['web_search', 'batch_web'])]
            other_tools = [t for t in enabled_tools if t not in api_tools and t not in web_tools]

            # Build concise search source guidance
            available_tools = []
            if api_tools:
                available_tools.extend(api_tools)
            if web_tools:
                available_tools.extend(web_tools)

            if available_tools:
                search_source_guidance = f"\n**Available Search Tools**: {', '.join(available_tools)}\n"
                search_source_guidance += "**Quick Guide**: Use academic APIs (arxiv_search, pubmed_search, etc.) for papers; use batch_web_search for general content.\n"
            else:
                search_source_guidance = ""
                search_source_guidance = f"\n\n**📚 AVAILABLE SEARCH TOOLS:**\n"
                search_source_guidance += f"You have access to the following search tools: **{', '.join(enabled_tools)}**\n"
                search_source_guidance += f"These tools are fully functional and ready to use. Focus on using these tools effectively to gather comprehensive information.\n"
                # Dynamic search strategy based on available tools
            if has_rag and has_other_sources:
                # Multiple sources available - require comprehensive strategy
                search_source_guidance += f"\n**CRITICAL - COMPREHENSIVE SEARCH STRATEGY (Multiple Sources Available):**\n"
                search_source_guidance += f"- **MANDATORY:** You MUST use MULTIPLE search sources to gather comprehensive information\n"
                search_source_guidance += f"- **RAG Knowledge Base:** ALWAYS call RAG multiple times (3-5 times minimum) with detailed, precise query terms from different angles\n"
                search_source_guidance += f"- **Other Search Tools:** Use WebSearch, PubMed, arXiv, Springer as appropriate\n"
                search_source_guidance += f"- **INTEGRATION REQUIREMENT:** Combine results from RAG AND other search sources, cross-reference findings\n"
            elif has_rag and not has_other_sources:
            # Only RAG available - focus on RAG optimization
                search_source_guidance += f"\n**CRITICAL - RAG-FOCUSED SEARCH STRATEGY (RAG Only Mode):**\n"
                search_source_guidance += f"- **MANDATORY:** Call RAG multiple times (3-5 times minimum) with different query formulations\n"
                search_source_guidance += f"- Use detailed, precise query terms (e.g., 'sodium-ion battery cathode P2-type layered oxide rate performance K+ doping')\n"
                search_source_guidance += f"- Search from different angles: material types, modification strategies, performance metrics, synthesis methods\n"
                search_source_guidance += f"- RAG contains high-quality academic papers - maximize its value through comprehensive multi-angle searches\n"
            elif not has_rag and has_other_sources:
                # No RAG, only other sources
                search_source_guidance += f"\n**SEARCH STRATEGY (Non-RAG Sources):**\n"
                search_source_guidance += f"- Use available search tools (WebSearch, PubMed, arXiv, Springer) comprehensively\n"
                search_source_guidance += f"- Generate multiple search queries from different angles\n"

            if disabled_tools:
                search_source_guidance += f"**Disabled**: {', '.join(disabled_tools)}\n"
        else:
            search_source_guidance = "**WARNING: ALL SEARCH TOOLS DISABLED**\n"
            search_source_guidance += f"No external search tools are available in this session. You can only work with existing files in the workspace (user_uploads/, library_refs/, etc.).\n"
            search_source_guidance += f"Focus on analyzing existing documents and files using document_extract, document_qa, and file operations.\n"

        # Add current date for time awareness
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Use pre-detected language flag to make EXPLICIT language instruction
        _is_cn = getattr(self, '_is_chinese_query', True)
        if _is_cn:
            lang_instruction = """## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**你正在处理的是中文查询的任务，必须使用中文回复。**
这条规则适用于所有输出：任务总结、研究发现、task_done报告等所有交付内容。
**禁止使用英文、韩文、日文或其他语言生成解释性文字。**
技术术语可以保留英文，但所有解释性文字必须使用中文。"""
        else:
            lang_instruction = """## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**You are handling an English-query task and MUST respond in English ONLY.**
This rule applies to ALL outputs: task summaries, findings, and task_done reports.
**DO NOT use Chinese, Korean, Japanese, or any other language.**
Under NO circumstances should you produce Chinese text in your response.
All reasoning, findings, and outputs MUST be in English exclusively."""

        system_prompt_template = f"""You are an Information Seeker Agent that follows the ReAct pattern (Reasoning + Acting).

{lang_instruction}

**IMPORTANT - Current Date: {current_date}**
When searching for recent information or papers, be aware that the current date is {current_date}. Papers and content from 2024, 2025, and 2026 are recent and relevant.
        
        Your role is to:
        1. Take decomposed sub-questions or tasks from parent agents
        2. Think step-by-step through reasoning 
        3. Use available tools to gather information when needed
        4. Continue reasoning based on tool results
        5. Repeat this process until you have sufficient information
        6. Call info_seeker_subjective_task_done to provide a structured summary and key files
        
        TOOL USAGE STRATEGY:
        Follow this optimized workflow for information gathering:
        
        0. **FIRST STEP - Check Workspace Files (Smart Detection):**
           - Use `list_workspace` to quickly check if `./user_uploads/` or `./library_refs/` contain any files
           - If files exist: Use `document_extract` to analyze ALL files (include ALL .pdf, .doc, .docx, .txt files)
           - If no files: Skip this step and proceed directly to web search
        
        1. INITIAL RESEARCH:{search_source_guidance}
           - Generate focused search queries (≤10) to avoid increased failure rates from excessive decomposition
           - Use `batch_web_search` to find relevant URLs for your queries. When calling the search statement, consider the language of the user's question. For example, for a Chinese question, generate a part of the search statement in Chinese.
           - Analyze the search results (titles, snippets, URLs, paper metadata) to identify promising sources

        2. CONTENT EXTRACTION:
           - **CRITICAL: For RAG knowledge base search results from `search_rag_knowledge`, you MUST use `rag_document_saver` to save documents (similar to how `url_crawler` works for web pages):**
                a) Save documents using RAG response content
                b) Save documents to workspace with proper metadata (title, journal, DOI, etc.)
                c) Enable proper citation generation in the final report
                d) **MANDATORY:** Always call `rag_document_saver` immediately after `search_rag_knowledge`
                e) **Usage (similar to url_crawler):**
                   ```
                   # Step 1: Search RAG - returns documents_for_download list
                   search_rag_knowledge(content="catalyst performance")
                   # Result contains: {{"doc_list": [...], "documents_for_download": [{{"file_id": ..., "title": ...}}, ...]}}

                   # Step 2: Pass the documents_for_download list to rag_document_saver
                   rag_document_saver(documents=[
                       {{"file_id": "rag_xxx", "title": "Document 1"}},
                       {{"file_id": "rag_yyy", "title": "Document 2"}}
                   ])
                   # Or simply call without arguments to download all:
                   rag_document_saver()
                   ```
                f) Documents will be saved under `rag_downloads/research/` with metadata preserved
                g) **MANDATORY:** After `rag_document_saver`, explicitly call `document_extract` with the saved RAG file paths so they enter `file_analysis.jsonl` and appear in references
                   ```
                   # Step 3: Analyze saved RAG files
                   # Use rag_document_saver result paths (data[].file_path)
                   document_extract(tasks=[
                       {{"file_path": "rag_downloads/research/xxx.md", "task": "Analyze this RAG paper"}}
                   ])
                   ```
           - Use `url_crawler` to extract content from URLs and save to `./url_crawler_save_files/`
           - For known paper IDs, you can use dedicated tools (arxiv_read_paper, get_pubmed_article, medrxiv_read_paper, etc.)
           - Store results with meaningful file paths (e.g., `url_crawler_save_files/research/ai_trends_2024.txt`)
        
        3. CONTENT ANALYSIS:
           - Use `document_qa` to ask focused questions about saved files
           - Use `document_extract` for multi-dimensional analysis of saved files:
                a) Provides structured analysis across five key dimensions: doc time source authority, core content and task relevance
        
        4. FILE MANAGEMENT:
           - Use `file_write` to save important findings or summaries
           - For reviewing saved content:
                a) Prefer `document_extract` to get comprehensive multi-dimensional analysis of saved files
                b) Use `file_read` ONLY for small files (<1000 tokens) when you need the entire content
                c) Avoid reading large files directly as it may exceed context limits
        
        5. TASK COMPLETION:
           - Call `info_seeker_subjective_task_done` with markdown summary and list of key files
        
        ### Systematic Tools:
        - `think` is a systematic tool. After receiving the response from the complex tool or before invoking any other tools, you must **first invoke the `think` tool**: to deeply reflect on the results of previous tool invocations (if any), and to thoroughly consider and plan the user's task. The `think` tool does not acquire new information; it only saves your thoughts into memory.
        
        Always provide clear reasoning for your actions and synthesize information effectively.

Below, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
$tool_schemas
</tools>
For each function call, return a JSON object with function name and arguments:
$tool_call_format
"""
        return (
            system_prompt_template
            .replace("$tool_schemas", tool_schemas_str)
            .replace("$tool_call_format", tool_call_format)
        )
# For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
# [unused11][{{"name": <function name>, "arguments": <args json object>}}][unused12]
# """
#         return system_prompt_template.replace("$tool_schemas", tool_schemas_str)

    @staticmethod
    def _build_initial_message_from_task_input(task_input: TaskInput) -> str:
        """Build the initial user message from TaskInput"""
        message = task_input.format_for_prompt()

        message += "\nPlease analyze this task and start your ReAct process:\n"
        message += "1. Reason about what information you need to gather\n"
        message += "2. Use appropriate tools to get that information\n"
        message += "3. Continue reasoning and acting until you have sufficient information\n"
        message += "4. Call info_seeker_subjective_task_done when ready to provide your complete findings\n\n"
        message += "Begin with your initial reasoning about the task."

        return message

    def execute_task(self, task_input: TaskInput) -> AgentResponse:
        """
        Execute a task using ReAct pattern (Reasoning + Acting)

        Args:
            task_input: TaskInput object with standardized task information

        Returns:
            AgentResponse with results and process trace
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting information seeker task: {task_input.task_content}")

            # Reset trace for new task
            self.reset_trace()

            # Initialize conversation history
            conversation_history = []

            # Build initial system prompt for ReAct
            system_prompt = self._build_system_prompt()

            # Build initial user message from TaskInput
            user_message = self._build_initial_message_from_task_input(task_input)

            # Add to conversation
            conversation_history.append({"role": "system", "content": system_prompt})
            conversation_history.append({"role": "user", "content": user_message + " /no_think"})

            iteration = 0
            task_completed = False
            unsafe_generation_correction_count = 0
            terminal_error = None
            # Get model configuration from config
            from config.config import get_config
            config = get_config()
            model_config = config.get_custom_llm_config()

            # ReAct Loop: Reasoning -> Acting -> Reasoning -> Acting...
            while iteration < self.config.max_iterations and not task_completed:
                iteration += 1
                self.logger.info(f"Planning iteration {iteration}")

                try:
                    # Get LLM response (reasoning + potential tool calls)
                    llm_turn = llm_chat(
                        conversation_history,
                        model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=model_config.get("timeout", 180),
                        max_retries=10,
                        reject_truncated=True,
                        retry_truncated_same_request=False,
                        tool_schemas=self.tool_schemas,
                        tool_call_mode=model_config.get("tool_call_mode"),
                        return_tool_turn=True,
                    )
                    llm_turn = self._coerce_llm_tool_turn(llm_turn)
                    assistant_text = llm_turn.content
                    assistant_message = {"content": assistant_text}
                    # Log the reasoning
                    try:
                        reasoning_source = llm_turn.reasoning_content or assistant_message.get("content")
                        if reasoning_source:
                            reasoning_content = extract_reasoning_from_response(reasoning_source)
                        # if assistant_message["content"]:
                        #     reasoning_content = assistant_message["content"].split("[unused16]")[-1].split("[unused17]")[0]
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
                    #         except Exception as ee:
                    #             return ["fail_tools_load", ee]
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
                                "InformationSeeker discarded generation with %s tool calls "
                                "and requested correction (%s/%s; limit=%s)",
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
                            terminal_error = (
                                "InformationSeeker tool-call batch remained unsafe after "
                                f"{self.MAX_UNSAFE_GENERATION_CORRECTIONS} corrective generations "
                                f"(received {len(tool_calls)}, limit "
                                f"{self.MAX_TOOL_CALLS_PER_GENERATION})"
                            )
                        else:
                            terminal_error = (
                                "InformationSeeker tool-call batch exceeded the per-generation "
                                "limit and no iteration remained for a corrective generation "
                                f"(received {len(tool_calls)}, limit "
                                f"{self.MAX_TOOL_CALLS_PER_GENERATION})"
                            )
                        self.log_error(iteration, terminal_error)
                        break

                    # Add only a validated, bounded assistant message to conversation.
                    self._append_assistant_tool_turn(
                        conversation_history,
                        llm_turn,
                        assistant_message["content"],
                    )

                    # if len(tool_calls) > 0 and tool_calls[0] == "fail_tools_load":
                    #     # Parse error, rerun
                    #     followup_prompt = f"There was a parsing error in the format of the tool call" \
                    #                       f" you generated:{tool_calls[1]} Please regenerate it."
                    #     conversation_history.append({"role": "user", "content": followup_prompt + " /no_think"})
                    #     continue


                    # Execute tool calls if any (Acting phase)

                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict) or "name" not in tool_call or "arguments" not in tool_call:
                            continue
                        arguments = tool_call["arguments"]
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except Exception:
                                arguments = {"raw_arguments": arguments}
                        tool_call["arguments"] = arguments

                        # Check if planning is complete
                        if tool_call["name"] in ["info_seeker_subjective_task_done"]:
                            task_completed = True
                            self.log_action(iteration, tool_call["name"], arguments, arguments)
                            break
                        if tool_call["name"] in ["think", "reflect"]:
                            tool_result = {"tool_results": "You can proceed to invoke other tools if needed."}
                        else:
                            tool_result = self.execute_tool_call(tool_call)

                        # Log the action using base class method
                        self.log_action(iteration, tool_call["name"], arguments, tool_result)

                        # Add tool result to conversation
                        self._append_tool_result_turn(
                            conversation_history,
                            tool_call,
                            tool_result,
                            llm_turn.tool_call_mode,
                        )

                    # If no tool calls, encourage continued planning
                    if len(tool_calls) == 0:
                        # Add follow-up prompt to encourage action or completion
                        followup_prompt = (
                            "Continue your analysis. If you need more information, use available tools. "
                            "If you have enough information to answer the question, call info_seeker_subjective_task_done with your complete context."
                        )
                        conversation_history.append({"role": "user", "content": followup_prompt + " /no_think"})
                    if iteration == self.config.max_iterations-3:
                        followup_prompt = "Due to length and number of rounds restrictions, you must now call the `info_seeker_subjective_task_done` tool to report the completion of your task."
                        conversation_history.append({"role": "user", "content": followup_prompt + " /no_think"})

                except LLMOutputTruncatedError as e:
                    if (
                        unsafe_generation_correction_count
                        < self.MAX_UNSAFE_GENERATION_CORRECTIONS
                        and iteration < self.config.max_iterations
                    ):
                        unsafe_generation_correction_count += 1
                        self.logger.warning(
                            "InformationSeeker discarded truncated generation and requested "
                            "correction (%s/%s): %s",
                            unsafe_generation_correction_count,
                            self.MAX_UNSAFE_GENERATION_CORRECTIONS,
                            e,
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
                        terminal_error = (
                            "InformationSeeker output remained unsafe after "
                            f"{self.MAX_UNSAFE_GENERATION_CORRECTIONS} corrective generations; "
                            "last generation was truncated"
                        )
                    else:
                        terminal_error = (
                            "InformationSeeker output was truncated and no iteration remained "
                            "for a corrective generation"
                        )
                    self.log_error(iteration, f"{terminal_error}: {e}")
                    break
                except Exception as e:
                    error_msg = f"Error in planning iteration {iteration}: {e}"
                    terminal_error = error_msg
                    self.log_error(iteration, error_msg)
                    break

            execution_time = time.time() - start_time
            # Extract final result
            if task_completed:
                # Find the task_done result in the trace
                task_done_result = None
                for step in reversed(self.reasoning_trace):
                    if step.get("type") == "action" and step.get("tool") == "info_seeker_subjective_task_done":
                        task_done_result = step.get("result")
                        break

                return self.create_response(
                    success=True,
                    result=task_done_result,
                    iterations=iteration,
                    execution_time=execution_time
                )
            else:
                return self.create_response(
                    success=False,
                    error=terminal_error or f"Task not completed within {self.config.max_iterations} iterations",
                    iterations=iteration,
                    execution_time=execution_time
                )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error in execute_task: {e}")
            return self.create_response(
                success=False,
                error=str(e),
                iterations=iteration if 'iteration' in locals() else 0,
                execution_time=execution_time
            )

    def _build_agent_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Build tool schemas for InformationSeekerAgent using proper MCP architecture.
        Schemas come from MCP server via client, not direct imports.
        """
        # Get MCP tool schemas from server via client (proper MCP architecture)
        schemas = super()._build_agent_specific_tool_schemas()

        # Add schemas for built-in task assignment tools
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
                    "name": "info_seeker_subjective_task_done",
                    "description": "Information Seeker Agent task completion reporting with information collection summary and related files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_summary": {
                                "type": "string",
                                "description": "Simple summary of what information has been collected for the current task and what new discoveries have been made.",
                                "format": "markdown"
                            },
                            "key_files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {
                                            "type": "string",
                                            "description": "Relative path to the file with collected content"
                                        },
                                    },
                                    "required": ["file_path"]
                                },
                                "description": "Collect files highly relevant to this task. "
                            },
                            "completion_status": {
                                "type": "string",
                                "enum": ["completed", "partial", "failed"],
                                "description": "Final status of the information gathering task"
                            },
                            "completion_analysis": {
                                "type": "string",
                                "description": "Brief analysis of task completion quality, information thoroughness, and any limitations or gaps."
                            }
                        },
                        "required": ["task_summary", "key_files", "completion_status", "completion_analysis"]
                    }
                }
            },

        ]

        schemas.extend(builtin_assignment_schemas)

        return schemas


# Factory function for creating the agent
def create_subjective_information_seeker(
    model: Optional[str] = None,
    max_iterations: int = 10,
    shared_mcp_client=None,
    **kwargs
) -> InformationSeekerAgent:
    """
    Create an InformationSeekerAgent instance with server-managed sessions.
    
    Args:
        model: The LLM model to use
        max_iterations: Maximum number of iterations
        shared_mcp_client: Optional shared MCP client from parent agent (prevents extra sessions)
        **kwargs: Additional configuration options
        
    Returns:
        Configured InformationSeekerAgent instance with appropriate tools
    """
    # Import the enhanced config function
    from .base_agent import create_agent_config

    # Create agent configuration (session managed by MCP server)
    config = create_agent_config(
        agent_name="InformationSeekerAgent",
        model=model,
        max_iterations=max_iterations,
        **kwargs
    )

    # Create agent instance with shared MCP client (filtered tools for information seeking)
    agent = InformationSeekerAgent(config=config, shared_mcp_client=shared_mcp_client)

    return agent
