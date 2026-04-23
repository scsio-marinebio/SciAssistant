# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
import json
from typing import Dict, Any, List
import time
import requests
import os
from .base_agent import BaseAgent, AgentConfig, AgentResponse, WriterAgentTaskInput



class WriterAgent(BaseAgent):
    """
    Writer Agent that follows ReAct pattern for content synthesis and generation
    
    This agent takes writing tasks from parent agents, searches through existing
    files and knowledge base, and creates long-form content through iterative
    reasoning and refinement. It does NOT access internet resources, only
    local files and memories.
    """

    def __init__(self, config: AgentConfig = None, shared_mcp_client=None, task_id: str = None):
        # Set default agent name if not specified
        if config is None:
            config = AgentConfig(agent_name="WriterAgent")
        elif config.agent_name == "base_agent":
            config.agent_name = "WriterAgent"

        super().__init__(config, shared_mcp_client)

        # Rebuild tool schemas with writer-specific tools only
        self.tool_schemas = self._build_tool_schemas()
        # Cancellation support
        self._cancellation_token = None
        # Progress callback support
        self.task_id = task_id
        self.progress_callback = None
        # Chapter progress tracking
        self._crash_test_part_count = 0
        # Chapter numbering validation
        self._expected_next_chapter = 1
        self._total_chapters = 0
        self._written_chapters = set()
        self._has_merged_final_report = False

    def set_cancellation_token(self, cancellation_token):
        """
        Set the cancellation token for this agent
        设置此代理的取消令牌

        Args:
            cancellation_token: threading.Event object that will be set when task should be cancelled
        """
        self._cancellation_token = cancellation_token

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _send_progress(self, stage: str, message: str, details: dict = None):
        """发送进度更新"""
        if self.progress_callback and self.task_id:
            import time
            progress_data = {
                'type': 'progress',
                'stage': stage,
                'message': message,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'details': details or {}
            }
            self.progress_callback(self.task_id, progress_data)

    def _check_cancellation(self) -> bool:
        """
        Check if task has been cancelled
        检查任务是否已被取消

        Returns:
            True if task should be cancelled, False otherwise
        """
        if self._cancellation_token and self._cancellation_token.is_set():
            self.logger.info("WriterAgent task cancellation detected")
            return True
        return False

    def _build_agent_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Build tool schemas for WriterAgent using proper MCP architecture.
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
                        "required": ["final_article_path", "article_summary", "completion_status",
                                     "completion_analysis"]
                    }
                }
            },
        ]

        schemas.extend(builtin_assignment_schemas)

        return schemas

    def _build_system_prompt(self, is_phase2: bool = False, user_outline: str = "") -> str:
        """Build the system prompt for the writer agent
        
        Args:
            is_phase2: Whether this is Human-in-the-Loop Phase 2 (user has confirmed outline)
            user_outline: The user-confirmed outline (only used in Phase 2)
        """
        tool_schemas_str = json.dumps(self.tool_schemas, ensure_ascii=False)
        
        # 根据_is_chinese_query标志明确指定输出语言
        _is_cn = getattr(self, '_is_chinese_query', True)
        if _is_cn:
            language_instruction = """## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**You MUST write the ENTIRE article in Chinese (中文).**
This rule applies to ALL outputs including: outline generation, chapter content, summaries, and the final article.
**所有内容必须使用中文撰写，包括：大纲生成、章节内容、摘要和最终文章。**"""
        else:
            language_instruction = """## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**You MUST write the ENTIRE article in English.**
This rule applies to ALL outputs including: outline generation, chapter content, summaries, and the final article.
**DO NOT use Chinese characters in the article content.**"""
        
        if is_phase2:
            # Phase 2: 用户已确认大纲，完全跳过大纲生成步骤
            system_prompt_template = f"""You are a professional writing master. The user has already confirmed an outline. Your task is to classify files into sections based on the PROVIDED outline, and iteratively call section_writer tool to create comprehensive content.

{language_instruction}

**CRITICAL: This is Human-in-the-Loop Phase 2. The user has already confirmed the outline below. You MUST NOT generate your own outline.**

=== USER CONFIRMED OUTLINE (DO NOT MODIFY) ===
{user_outline}
=== END OF USER CONFIRMED OUTLINE ===

MANDATORY WORKFLOW (Phase 2 - Outline Already Confirmed):

1. FILE CLASSIFICATION (FIRST STEP - NO OUTLINE GENERATION)
   - Use the search_result_classifier tool to classify key files according to the USER CONFIRMED OUTLINE above.
   - Pass the EXACT user-confirmed outline as the 'outline' parameter - DO NOT modify it in any way.
   - Ensure optimal distribution of reference materials across chapters based on content relevance.

2. ITERATIVE SECTION WRITING
   - Call section_writer tool sequentially for each chapter
   - CRITICAL: Must wait for previous chapter completion before starting the next chapter
   - Pass only the specific chapter outline, target file path and corresponding classified files to each section writer
   - Generate save path for each chapter using \"./report/part_X.md\" format (e.g., \"./report/part_1.md\" for first chapter)
   - Check section writer results after completion; retry up to 2 times per chapter if quality is insufficient based on returned fields (do not read saved files)
   - When you call the section_writer tool, pay special attention to the fact that the parameter value of written_chapters_summary is a summary of the content returned by all previously completed chapters. Be careful not to make any changes to the summary content, including compressing the content.

3. TASK COMPLETION
   - After all chapters are written, you must first call the concat_section_files tool to merge the saved chapter files into one file, then call writer_subjective_task_done to finalize and return.

CRITICAL REQUIREMENTS:
- **DO NOT generate your own outline** - the user has already confirmed the outline above
- **Use the user-confirmed outline EXACTLY as provided** when calling search_result_classifier
- No parallel writing - strictly sequential chapter execution
- Wait for each section writer completion before proceeding to next chapter
- Classify files appropriately to support each chapter's content needs
- Note again that to merge all the written chapter files, you must use the concat_section_files tool!!! You are not allowed to call any other tools for merging!!!

FORBIDDEN ACTIONS:
- DO NOT call think tool to plan or generate a new outline
- DO NOT modify the user-confirmed outline in any way
- NEVER generate meta-structural chapters that describe how the article is organized
- AVOID introductory sections that outline \"Chapter 1 will cover..., Chapter 2 will discuss...\"
- DO NOT create chapters that explain the report structure or methodology
- Each chapter must contain SUBSTANTIVE CONTENT, not descriptions of what other chapters contain

Usage of TOOLS:
- search_result_classifier: Classify key files into outline sections (use user-confirmed outline)
- section_writer: Write individual chapters sequentially  
- writer_subjective_task_done: Complete the writing task
- concat_section_files: Concatenate the content of the saved section files into a single file

Below, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
{tool_schemas_str}
</tools>
For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
[unused11][{{"name": <function name>, "arguments": <args json object>}}][unused12]

Execute workflow systematically to produce high-quality, coherent long-form content with substantive chapters."""
        else:
            # Phase 1 or normal mode: 需要生成大纲
            system_prompt_template = f"""You are a professional writing master. You will receive key files and user problems. Your task is to generate an outline highly consistent with the user problem, classify files into sections, and iteratively call section_writer tool to create comprehensive content.

{language_instruction}

Then you strictly follow the steps given below:
        
        MANDATORY WORKFLOW:
        
        1. OUTLINE GENERATION
        Based on the core content of the provided key files collection(file_core_content), generate a high-quality outline suitable for long-form writing. Strictly adhere to the following requirements during generation:  
        - Before generating the outline, carefully review the provided **file_core_content**, prioritizing sections with:  
            1.**Higher authority** (credible sources)
            2.**Greater information richness** (substantive, detailed content)
            3.**Stronger relevance** (direct alignment with user query)
            4.**Timeliness** (if user's query is time-sensitive, prioritize recent/updated content)
        Select these segments as the basis for outline generation. Note that we only focus on relevance to the question, so when generating the outline, do not add unrelated sections just for the sake of length. Additionally, the sections should flow logically and not be too disjointed, as this would harm the readability of the final output.  
        - The overall structure must be **logically clear**, with **no repetition or redundancy** between chapters.  
        - **Note1:** The generated outline must not only have chapter-level headings (Level 1) highly relevant to the user's question, but the subheadings (Level 2) must also be highly relevant to the user's question. It is not permitted to generate chapter titles with weak relevance, whether Level 1 or Level 2.
        - **Note2:** STRICT NUMBERING FORMAT REQUIRED (CRITICAL FOR PDF TOC): 
            - Level 1 headings (Chapters) MUST use Markdown '##' (H2) and Arabic numerals followed by a period:
              * For English: "## 1. Introduction", "## 2. Core Concepts"
              * For Chinese: "## 1. 引言", "## 2. 核心概念"
              * Do NOT use Chinese numerals like "一、" or "Chapter 1".
            - Level 2 headings (Subsections) MUST be PLAIN TEXT WITHOUT any markdown symbols (no ###, no **, no *):
              * CRITICAL RULE: DO NOT add "###" before Level 2 headings! They must be plain text only!
              * Sub-heading numbers MUST match parent chapter number: Chapter 1 → 1.1, 1.2; Chapter 2 → 2.1, 2.2; Chapter 3 → 3.1, 3.2, etc.
              * For English: "1.1 Background", "1.2 Main Findings" (for Chapter 1), "2.1 Methods" (for Chapter 2)
              * For Chinese: "1.1 背景", "1.2 主要发现" (第1章), "2.1 方法" (第2章)
              * WRONG FORMAT EXAMPLES (NEVER use these):
                - "### 2.1 Title" (has ### symbol)
                - "### 2.4 大规模语言模型与强化学习的融合" (has ### symbol)
                - "**2.1 Title**" (has ** symbols)
                - "2.1 xxx" under "## 1. Title" (Should be "1.1 xxx" to match chapter 1)
                - "2.2 xxx" under "## 3. Title" (Should be "3.2 xxx" to match chapter 3)
              * CORRECT FORMAT EXAMPLES (ALWAYS use these):
                - "1.1 Title" (plain text, under "## 1. xxx")
                - "2.4 大规模语言模型与强化学习的融合" (plain text, under "## 2. xxx")
                - "3.2 Methods" (plain text, under "## 3. xxx")
            - This structure is CRITICAL for the final PDF table of contents.
            - REMINDER: Every Level 2 heading (1.1, 1.2, 2.1, 2.2, 2.3, 2.4, etc.) MUST be plain text without any markdown symbols!
        - **Note3:** The number of chapters must not exceed 7, dynamic evaluation can be performed based on the collected content. For example, if there is a lot of content, more chapters can be generated, and vice versa. But each chapter should only include Level 1 and Level 2 headings. Also, please generate more Level 2 headings (suggest 3-6) to ensure the content is rich and detailed. However, if the first chapter is an abstract or introduction, do not generate subheadings (level-2 headings)—only include the main heading (level-1). Additionally, tailor the outline style based on the type of document. For example, in a research report, the first chapter should preferably be titled \"Abstract\" or \"Introduction.\"  
        
        2. FILE CLASSIFICATION  
        - Use the search_result_classifier tool to reasonably split the outline generated above and accurately assign key files to each chapter of the outline.
        - Ensure optimal distribution of reference materials across chapters based on content relevance.
        
        3. ITERATIVE SECTION WRITING
        - Call section_writer tool sequentially for each chapter
        - CRITICAL: Must wait for previous chapter completion before starting the next chapter
        - Pass only the specific chapter outline , target file path and corresponding classified files to each section writer
        - Generate save path for each chapter using \"./report/part_X.md\" format (e.g., \"./report/part_1.md\" for first chapter)
        - Check section writer results after completion; retry up to 2 times per chapter if quality is insufficient based on returned fields (do not read saved files)
        - When you call the section_writer tool, pay special attention to the fact that the parameter value of written_chapters_summary is a summary of the content returned by all previously completed chapters. Be careful not to make any changes to the summary content, including compressing the content.
        
        4. TASK COMPLETION
        - After all chapters are written, you must first call the concat_section_files tool to merge the saved chapter files into one file, then call writer_subjective_task_done to finalize and return.
        
        CRITICAL REQUIREMENTS:
        - The creation of the outline is crucial! Therefore, you must strictly adhere to the above requirements for generating the outline.
        - No parallel writing - strictly sequential chapter execution
        - Wait for each section writer completion before proceeding to next chapter
        - Classify files appropriately to support each chapter's content needs
        - Note again that to merge all the written chapter files, you must use the concat_section_files tool!!! You are not allowed to call any other tools for merging!!!
        
        FORBIDDEN CONTENT PATTERNS:
        - NEVER generate meta-structural chapters that describe how the article is organized
        - AVOID introductory sections that outline \"Chapter 1 will cover..., Chapter 2 will discuss...\"
        - DO NOT create chapters that explain the report structure or methodology
        - Each chapter must contain SUBSTANTIVE CONTENT, not descriptions of what other chapters contain
        - When generating an outline, if it is not a professional term, the language should remain consistent with the user's question.\"
        
        Usage of TOOLS:
        - search_result_classifier: Classify key files into outline sections. **IMPORTANT**: You MUST include the `reasoning_text` parameter with your reasoning process explaining how you analyzed the research materials and why you generated this specific outline structure. This text will be displayed to the user before the outline. Example: "基于提供的研究资料，我分析了关于[主题]的多个维度，包括[维度1]、[维度2]等。结合用户的查询需求，我将为您撰写一篇综合研究报告，大纲结构如下："
        - section_writer: Write individual chapters sequentially  
        - writer_subjective_task_done: Complete the writing task
        - concat_section_files: Concatenate the content of the saved section files into a single file
        - think tool: \"Think\" is a systematic tool requiring its use during key steps. Before executing actions like generating an outline, you must first call this tool to deeply consider the given content and key requirements, ensuring the output meets specifications. Similarly, during iterative chapter generation, after receiving feedback and before writing the next chapter, call \"think\" to reflect on the current chapter. This provides guidance to avoid content repetition and ensure smooth transitions between chapters.
        
        SPECIAL HANDLING - Human in the Loop Mode:
        - If search_result_classifier returns error "WAITING_FOR_OUTLINE_CONFIRMATION", this means the system is in Human-in-the-Loop mode and waiting for user to confirm the outline.
        - In this case, you MUST immediately call writer_subjective_task_done with completion_status="partial" to end the current task and allow the user to review the outline.
        - Do NOT retry search_result_classifier or attempt other operations when you see this error.
        
        Execute workflow systematically to produce high-quality, coherent long-form content with substantive chapters.

Below, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
{tool_schemas_str}
</tools>
For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
[unused11][{{"name": <function name>, "arguments": <args json object>}}][unused12]
"""
        return system_prompt_template

    def _build_initial_message_from_task_input(self, task_input: WriterAgentTaskInput) -> str:
        """Build the initial user message from TaskInput"""
        message = ""

        # Add key files information with reliability dimensions
        def load_json_from_server(file_path):
            """Load JSONL file from MCP server using unlimited internal tool"""
            res = []
            try:
                # Use json read tool directly through raw MCP client
                raw_result = self.mcp_tools.client.call_tool("load_json", {"file_path": file_path})
                
                if not raw_result.success:
                    self.logger.error(f"Failed to read file from server: {raw_result.error}")
                    return res
                
                res = json.loads(raw_result.data["content"][0]["text"])["data"]
                                            
            except Exception as e:
                self.logger.error(f"Error loading file {file_path} from MCP server: {e}")
                import traceback
                self.logger.debug(f"Full traceback: {traceback.format_exc()}")
                
            return res

        key_files_dict = {}
        # 【关键修复】使用连续编号作为文件序号，确保和 merge_reports 一致
        file_path_to_continuous_num = {}

        server_analysis_path = f"doc_analysis/file_analysis.jsonl"
        self.logger.debug(f"Loading analysis from MCP server: {server_analysis_path}")
        file_analysis_list = load_json_from_server(server_analysis_path)

        # 【智能过滤】基于information_richness字段判断，而不是关键词匹配
        continuous_num = 0  # 使用连续编号计数器
        for line_num, file_info in enumerate(file_analysis_list, 1):
            if file_info.get('file_path'):
                file_path = file_info.get('file_path')
                doc_time = file_info.get('doc_time', '')
                info_richness = file_info.get('information_richness', '')
                
                # 跳过处理失败的文件
                if doc_time == "Processing failed":
                    self.logger.warning(f"跳过处理失败的文件 [原始行号{line_num}]: {file_path}")
                    continue
                
                # 【智能过滤】基于information_richness判断
                # 检查明确的负面表述：considered scarce, indicating scarcity, lacks substantive content
                info_richness_lower = info_richness.lower()
                negative_indicators = [
                    'considered scarce', 'indicating scarcity', 'is scarce',
                    'lacks substantive content', 'no substantive content',
                    'very limited information', 'does not provide any substantive'
                ]
                if info_richness and any(indicator in info_richness_lower for indicator in negative_indicators):
                    self.logger.warning(f"跳过信息稀缺的文件 [原始行号{line_num}]: {file_path} (richness: {info_richness[:80]})")
                    continue
                
                # 有效文件使用连续编号
                continuous_num += 1
                key_files_dict[file_path] = file_info
                file_path_to_continuous_num[file_path] = continuous_num
                self.logger.debug(f"映射连续编号 {continuous_num} (原始行号{line_num}) 到文件: {file_path}")

        file_core_content = ""
        valid_file_paths = []  # 收集有效文件路径用于推送
        if hasattr(task_input, 'key_files') and task_input.key_files:
            message += "Key Files:\n"
            valid_file_count = 0
            for file_ in task_input.key_files:
                file_path = file_.get('file_path')
                if file_path in key_files_dict:
                    valid_file_count += 1
                    valid_file_paths.append(file_path)  # 记录有效文件路径
                    # 【关键修复】使用连续编号作为引用序号，与 merge_reports 保持一致
                    continuous_num = file_path_to_continuous_num.get(file_path, valid_file_count)
                    file_info = key_files_dict[file_path]
                    doc_time = file_info.get('doc_time', 'Not specified')
                    source_authority = file_info.get('source_authority', 'Not assessed')
                    task_relevance = file_info.get('task_relevance', 'Not assessed')
                    information_richness = file_info.get('information_richness', 'Not assessed')
                    message += f"{continuous_num}. File: {file_path}\n"

                    file_core_content += f"[{str(continuous_num)}]doc_time:{doc_time}|||source_authority:{source_authority}|||task_relevance:{task_relevance}|||information_richness:{information_richness}|||summary_content:{file_info.get('core_content', '')}\n"
            
            # 【Fallback】只在匹配文件数极少（<3个）且明显异常时才回退
            # 原因：可能是路径不匹配问题，而非PlannerAgent的正常筛选
            # 注意：如果PlannerAgent有意只选择少量文件，此fallback可能违背其意图
            if valid_file_count < 3 and len(key_files_dict) > 10:
                self.logger.warning(
                    f"Planner传入的key_files仅匹配到 {valid_file_count} 个文件（阈值: 3），"
                    f"可能存在路径不匹配问题，回退使用file_analysis.jsonl中全部 {len(key_files_dict)} 个有效文件"
                )
                # 重置，使用全部有效文件
                message = "Key Files:\n"
                file_core_content = ""
                valid_file_paths = []
                valid_file_count = 0
                for file_path, file_info in key_files_dict.items():
                    valid_file_count += 1
                    valid_file_paths.append(file_path)
                    continuous_num = file_path_to_continuous_num.get(file_path, valid_file_count)
                    doc_time = file_info.get('doc_time', 'Not specified')
                    source_authority = file_info.get('source_authority', 'Not assessed')
                    task_relevance = file_info.get('task_relevance', 'Not assessed')
                    information_richness = file_info.get('information_richness', 'Not assessed')
                    message += f"{continuous_num}. File: {file_path}\n"
                    file_core_content += f"[{str(continuous_num)}]doc_time:{doc_time}|||source_authority:{source_authority}|||task_relevance:{task_relevance}|||information_richness:{information_richness}|||summary_content:{file_info.get('core_content', '')}\n"

            message += "\n"
            message += f"file_core_content: {file_core_content}\n"
            self.logger.info(f"Writer 使用 {valid_file_count} 个有效文件（已过滤处理失败和内容无效的文件）")
            
            # 推送文件列表进度（WriterAgent实际使用的文件）
            if valid_file_count > 0 and self.progress_callback:
                try:
                    # 提取文件名（去除路径，限制长度）
                    file_names = []
                    for file_path in valid_file_paths[:10]:  # 最多显示10个
                        # 提取文件名
                        file_name = file_path.split('/')[-1].split('\\')[-1]
                        # 限制长度为50个字符
                        if len(file_name) > 50:
                            file_name = file_name[:47] + '...'
                        file_names.append(file_name)
                    
                    # 统计file_analysis.jsonl中的总文件数（检索到的相关文献总数）
                    total_retrieved_count = len(key_files_dict)  # 过滤无效后的总数
                    
                    # 发送进度更新
                    self.progress_callback(self.task_id, {
                        'type': 'progress',
                        'stage': 'writing_started',
                        'message': '开始撰写报告' if getattr(self, '_is_chinese_query', True) else 'Starting report writing',
                        'details': {
                            'key_files_count': valid_file_count,
                            'total_retrieved_count': total_retrieved_count,
                            'file_names': file_names
                        }
                    })
                    self.logger.info(f"[PROGRESS] 推送文件列表: {valid_file_count}个核心文件（检索到{total_retrieved_count}个相关文献）")
                except Exception as e:
                    self.logger.warning(f"[PROGRESS] 推送文件列表失败: {e}，继续执行任务")
        else:
            message += "Key Files: None provided\n"

        message += "\n"
        # Add user query
        if hasattr(task_input, 'user_query') and task_input.user_query:
            message += f"User Query: {task_input.user_query}\n"
        else:
            message += "User Query: Not provided\n"

        return message

    def execute_task(self, task_input: WriterAgentTaskInput) -> AgentResponse:
        """
        Execute a writing task using ReAct pattern

        Args:
            task_input: TaskInput object with standardized task information

        Returns:
            AgentResponse with writing results and process trace
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting writing task: {task_input.task_content}")

            # Reset trace for new task
            self.reset_trace()

            # Initialize conversation history
            conversation_history = []

            # 检测 Human in the loop Phase 2
            is_phase2 = False
            user_outline = ""
            if hasattr(task_input, 'task_content') and task_input.task_content:
                if "Human in the loop 阶段2" in task_input.task_content or "Human in the loop Phase 2" in task_input.task_content:
                    is_phase2 = True
                    # 提取用户确认的大纲
                    import re as _re_outline
                    outline_match = _re_outline.search(r'用户确认的大纲[：:]?\s*\n(.*?)(?:\n用户原始查询|$)', task_input.task_content, _re_outline.DOTALL)
                    if outline_match:
                        user_outline = outline_match.group(1).strip()
                    self.logger.info(f"[HITL] Phase 2 detected, user_outline length={len(user_outline)}")
            
            # Also check environment variable
            if not is_phase2:
                is_phase2 = os.environ.get('HUMAN_IN_LOOP_PHASE2', 'false').lower() == 'true'
                if is_phase2:
                    # 从 workspace 文件读取大纲
                    workspace_path = os.environ.get('AGENT_WORKSPACE_PATH', '')
                    if workspace_path:
                        from pathlib import Path
                        outline_file = Path(workspace_path) / '.user_outline'
                        if outline_file.exists():
                            with open(outline_file, 'r', encoding='utf-8') as f:
                                user_outline = f.read().strip()
                    self.logger.info(f"[HITL] Phase 2 detected via env var, user_outline length={len(user_outline)}")
            
            # Build system prompt for writing
            system_prompt = self._build_system_prompt(is_phase2=is_phase2, user_outline=user_outline)

            # Build initial user message from TaskInput
            user_message = self._build_initial_message_from_task_input(task_input)

            # Add to conversation
            conversation_history.append({"role": "system", "content": system_prompt})
            conversation_history.append({"role": "user", "content": user_message + " /no_think"})

            iteration = 0
            task_completed = False

            self.logger.debug("Checking conversation history before model call")
            self.logger.debug(f"Conversation history: {conversation_history}")
            # ReAct Loop for Writing: Research → Plan → Write → Refine → Complete
            # Get model configuration from config
            from config.config import get_config
            config = get_config()
            model_config = config.get_custom_llm_config()
            
            pangu_url = model_config.get('url') or os.getenv('MODEL_REQUEST_URL', '')
            model_token = model_config.get('token') or os.getenv('MODEL_REQUEST_TOKEN', '')
            headers = {'Content-Type': 'application/json', 'csb-token': model_token}

            while iteration < self.config.max_iterations and not task_completed:
                # Check for cancellation at the start of each iteration
                if self._check_cancellation():
                    self.logger.info(f"WriterAgent task cancelled at iteration {iteration}")
                    execution_time = time.time() - start_time
                    return self.create_response(
                        success=False,
                        result="Task was cancelled by user",
                        iterations=iteration,
                        execution_time=execution_time
                    )

                iteration += 1
                self.logger.info(f"Writing iteration {iteration}")

                try:
                    # Get LLM response (reasoning + potential tool calls) with retry

                    max_retries = 10
                    response = None

                    for attempt in range(max_retries):
                        try:

                            response = requests.post(
                                url=pangu_url,
                                headers=headers,
                                json={
                                    "model": self.config.model,
                                    "chat_template":"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<s>[unused9]系统：[unused10]' }}{% endif %}{% if message['role'] == 'system' %}{{'<s>[unused9]系统：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'assistant' %}{{'[unused9]助手：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'tool' %}{{'[unused9]工具：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'function' %}{{'[unused9]方法：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'user' %}{{'[unused9]用户：' + message['content'] + '[unused10]'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '[unused9]助手：' }}{% endif %}",
                                    "messages": conversation_history,
                                    "temperature": self.config.temperature,
                                    "max_tokens": self.config.max_tokens,
                                    "spaces_between_special_tokens": False,
                                },
                                timeout=model_config.get("timeout", 180)
                            )
                            response = response.json()

                            self.logger.debug(f"API response received")
                            break  # Success, exit retry loop

                        except Exception as e:
                            self.logger.warning(f"LLM API call attempt {attempt + 1} failed: {e}")
                            if attempt == max_retries - 1:
                                raise e  # Last attempt, re-raise the exception
                            time.sleep(6)  # Simple 1 second delay between retries

                    if response is None:
                        raise Exception("Failed to get response after all retries")

                    assistant_message = response["choices"][0]["message"]

                    try:
                        if assistant_message["content"]:
                            reasoning_content = assistant_message["content"].split("[unused16]")[-1].split("[unused17]")[0]
                            if len(reasoning_content) > 0:
                                self.log_reasoning(iteration, reasoning_content)
                    except Exception as e:
                        self.logger.warning(f"Tool call parsing error: {e}")
                        # Parse error, rerun
                        followup_prompt = f"There is a problem with the format of model generation: {e}. Please try again."
                        conversation_history.append({"role": "user", "content": followup_prompt + " /no_think"})
                        continue

                    def extract_tool_calls(content):
                        import re
                        if not content:
                            return []
                        tool_call_str = re.findall(r"\[unused11\]([\s\S]*?)\[unused12\]", content)
                        if len(tool_call_str) > 0:
                            try:
                                tool_calls = json.loads(tool_call_str[0])
                            except:
                                return []
                        else:
                            return []
                        return tool_calls

                    # Add assistant message to conversation
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_message["content"]
                    })

                    tool_calls = extract_tool_calls(assistant_message["content"])
                    
                    # 增强日志：记录工具调用解析结果
                    if len(tool_calls) == 0:
                        self.logger.warning(f"[工具调用] 第{iteration}次迭代未解析到任何工具调用")
                        # 记录原始内容的前500字符用于调试
                        content_preview = assistant_message["content"][:500] if assistant_message.get("content") else "None"
                        self.logger.debug(f"[工具调用] 原始响应内容预览: {content_preview}")
                        
                        # 智能检测：如果Reasoning中提到section_writer但未成功调用，立即重试
                        content = assistant_message.get("content", "")
                        has_tool_marker = "[unused11]" in content and "[unused12]" in content
                        mentions_section_writer = "section_writer" in content
                        
                        if mentions_section_writer and has_tool_marker:
                            # 确定：AI生成了工具调用标记但解析失败（JSON格式错误或不完整）
                            self.logger.error(f"[工具调用] 检测到工具调用标记但解析失败，JSON可能格式错误或不完整")
                            retry_prompt = (
                                "工具调用格式有误，未能成功解析。请重新生成section_writer工具调用，"
                                "确保JSON格式正确且完整。格式示例：[unused11][{\"name\": \"section_writer\", \"arguments\": {...}}][unused12] /no_think"
                            )
                            conversation_history.append({"role": "user", "content": retry_prompt})
                            continue  # 立即进入下一次LLM调用，不增加迭代计数
                        elif mentions_section_writer and not has_tool_marker:
                            # 推断：AI在思考中提到section_writer但可能忘记生成工具调用
                            self.logger.warning(f"[工具调用] Reasoning中提到section_writer但未生成工具调用标记")
                            retry_prompt = (
                                "你在思考中提到要调用section_writer工具，但没有生成工具调用。"
                                "请使用正确的格式生成工具调用：[unused11][{\"name\": \"section_writer\", \"arguments\": {...}}][unused12] /no_think"
                            )
                            conversation_history.append({"role": "user", "content": retry_prompt})
                            continue
                    else:
                        self.logger.debug(f"[工具调用] 第{iteration}次迭代解析到{len(tool_calls)}个工具调用: {[tc.get('name') for tc in tool_calls]}")

                    # Execute tool calls if any (Acting phase)
                    for tool_call in tool_calls:
                        tool_name = tool_call["name"]
                        # Str
                        arguments = tool_call["arguments"]
                        tool_name = tool_call["name"]
                        self.logger.debug(f"Arguments is string: {isinstance(arguments, str)}")

                        # Check if planning is complete
                        if tool_name in ["writer_subjective_task_done"]:
                            if self._written_chapters and not self._has_merged_final_report:
                                tool_result = {
                                    "success": False,
                                    "error": "最终报告尚未成功合并。请先调用 concat_section_files 生成 final_report，再调用 writer_subjective_task_done。"
                                }
                            else:
                                task_completed = True
                                self.log_action(iteration, tool_name, arguments, arguments)
                                break
                        elif tool_name in ["think"]:
                            tool_result = {
                                "tool_results": "You can proceed to invoke other tools if needed. But the next step cannot call the reflect tool"}
                        elif tool_name == "section_writer":
                            # 章节编号验证
                            import re
                            write_file_path = ""
                            if isinstance(arguments, dict):
                                write_file_path = arguments.get('target_file_path', '') or arguments.get('write_file_path', '')
                            elif isinstance(arguments, str):
                                args_dict = json.loads(arguments)
                                write_file_path = args_dict.get('target_file_path', '') or args_dict.get('write_file_path', '')
                            
                            # 提取章节编号
                            chapter_match = re.search(r'part_(\d+)\.md', write_file_path)
                            if chapter_match:
                                chapter_num = int(chapter_match.group(1))
                                
                                # 验证章节编号连续性
                                if chapter_num != self._expected_next_chapter:
                                    error_msg = f"章节编号错误：期望写第{self._expected_next_chapter}章，但调用了第{chapter_num}章。请按顺序写作，不要跳过章节。"
                                    self.logger.error(f"[章节验证] {error_msg}")
                                    tool_result = {
                                        "success": False,
                                        "error": error_msg,
                                        "expected_chapter": self._expected_next_chapter,
                                        "actual_chapter": chapter_num
                                    }
                                else:
                                    # 在真正开始写章节前就推送进度，避免前端长时间停留在上一状态
                                    try:
                                        outline = ""
                                        if isinstance(arguments, dict):
                                            outline = arguments.get('current_chapter_outline', '')
                                        elif isinstance(arguments, str):
                                            args_dict = json.loads(arguments)
                                            outline = args_dict.get('current_chapter_outline', '')

                                        if outline:
                                            chapter_title = outline.split('\n')[0].strip()
                                            chapter_title = chapter_title.replace('#', '').strip()[:50]

                                            _writing_prefix = '正在撰写: ' if getattr(self, '_is_chinese_query', True) else 'Writing: '
                                            self._send_progress('writing_chapter', f'{_writing_prefix}{chapter_title}', {
                                                'chapter_title': chapter_title
                                            })
                                    except Exception as e:
                                        self.logger.debug(f"Failed to send pre-write chapter progress: {e}")

                                    # 章节编号正确，执行工具调用
                                    tool_result = self.execute_tool_call(tool_call)
                                    
                                    # 如果成功，更新计数器
                                    if tool_result.get("success"):
                                        self._written_chapters.add(chapter_num)
                                        self._expected_next_chapter = chapter_num + 1
                                        self.logger.info(f"[章节验证] 成功完成第{chapter_num}章，下一章应为第{self._expected_next_chapter}章")
                                        
                                        self._crash_test_part_count += 1
                                        try:
                                            _saved_msg = (
                                                f'已保存: part_{chapter_num}.md'
                                                if getattr(self, '_is_chinese_query', True)
                                                else f'Saved: part_{chapter_num}.md'
                                            )
                                            self._send_progress('chapter_saved', _saved_msg, {'chapter_num': chapter_num})
                                        except Exception:
                                            pass
                            else:
                                try:
                                    outline = ""
                                    if isinstance(arguments, dict):
                                        outline = arguments.get('current_chapter_outline', '')
                                    elif isinstance(arguments, str):
                                        args_dict = json.loads(arguments)
                                        outline = args_dict.get('current_chapter_outline', '')

                                    if outline:
                                        chapter_title = outline.split('\n')[0].strip()
                                        chapter_title = chapter_title.replace('#', '').strip()[:50]

                                        _writing_prefix = '正在撰写: ' if getattr(self, '_is_chinese_query', True) else 'Writing: '
                                        self._send_progress('writing_chapter', f'{_writing_prefix}{chapter_title}', {
                                            'chapter_title': chapter_title
                                        })
                                except Exception as e:
                                    self.logger.debug(f"Failed to send pre-write chapter progress: {e}")

                                # 无法提取章节编号，正常执行
                                tool_result = self.execute_tool_call(tool_call)
                                
                                # 更新计数（无章节编号的情况）
                                if tool_result.get("success"):
                                    self._crash_test_part_count += 1
                        elif tool_name == "concat_section_files":
                            # 在合并前检查章节完整性
                            if self._expected_next_chapter > 1:
                                expected_chapters = set(range(1, self._expected_next_chapter))
                                missing_chapters = expected_chapters - self._written_chapters
                                
                                if missing_chapters:
                                    error_msg = f"章节不完整：缺少第{sorted(missing_chapters)}章。请先完成所有章节再合并。"
                                    self.logger.error(f"[完整性检查] {error_msg}")
                                    tool_result = {
                                        "success": False,
                                        "error": error_msg,
                                        "missing_chapters": sorted(missing_chapters),
                                        "written_chapters": sorted(self._written_chapters)
                                    }
                                else:
                                    self.logger.info(f"[完整性检查] 所有{len(self._written_chapters)}个章节已完成，可以合并")
                                    tool_result = self.execute_tool_call(tool_call)
                                    if tool_result.get("success"):
                                        self._has_merged_final_report = True
                            else:
                                tool_result = self.execute_tool_call(tool_call)
                                if tool_result.get("success"):
                                    self._has_merged_final_report = True
                        else:
                            tool_result = self.execute_tool_call(tool_call)

                        # 当章节结构与大纲不一致时，显式提示模型重试当前章节
                        if tool_name == "section_writer" and not tool_result.get("success", False):
                            error_text = str(tool_result.get("error", ""))
                            if "chapter structure mismatch" in error_text:
                                retry_msg = (
                                    "section_writer 返回章节结构错误。请保持 current_chapter_outline 的标题逐行一致，"
                                    "不得新增/删除/改写标题，重新调用同一章节的 section_writer。"
                                )
                                conversation_history.append({"role": "user", "content": retry_msg + " /no_think"})

                        # Log the action using base class method
                        self.log_action(iteration, tool_name, arguments, tool_result)

                        # Add tool result to conversation
                        conversation_history.append({
                            "role": "tool",
                            "content": json.dumps(tool_result, ensure_ascii=False, indent=2) + " /no_think"
                        })

                    # If no tool calls, encourage continued writing
                    if len(tool_calls) == 0:
                        # Add follow-up prompt to encourage action or completion
                        followup_prompt = (
                            "Continue your writing process. If you need to research more, use available tools. "
                            "If you need to write or edit content, use file operations. "
                            "If your writing is complete and meets requirements, call writer_subjective_task_done. /no_think"
                        )
                        conversation_history.append({"role": "user", "content": followup_prompt})

                except Exception as e:
                    error_msg = f"Error in writing iteration {iteration}: {e}"
                    self.log_error(iteration, error_msg)
                    break

            # 【降级兜底A】writer agent 异常退出或超时时，尝试自动合并已有的 part_*.md
            # 采用分级降级策略：根据章节数决定是否合并以及如何标注
            if not task_completed:
                try:
                    workspace_path = os.environ.get('AGENT_WORKSPACE_PATH', '')
                    if workspace_path:
                        from pathlib import Path
                        import re as _re
                        report_dir = Path(workspace_path) / "report"
                        final_report_path = report_dir / "final_report.md"
                        if not final_report_path.exists() and report_dir.exists():
                            part_files = sorted(
                                report_dir.glob("part_*.md"),
                                key=lambda p: int(_re.search(r'part_(\d+)', p.name).group(1))
                                if _re.search(r'part_(\d+)', p.name) else 0
                            )
                            part_count = len(part_files)
                            
                            if part_count == 0:
                                self.logger.warning("[降级兜底A] 无可用章节，跳过合并")
                            elif part_count < 3:
                                # 内容太少，标注为"草稿"并建议重试
                                self.logger.warning(
                                    f"[降级兜底A] 仅 {part_count} 个章节，标注为草稿（建议用户重试）"
                                )
                                merged = ""
                                for pf in part_files:
                                    try:
                                        merged += pf.read_text(encoding='utf-8') + "\n\n"
                                    except Exception:
                                        pass
                                if merged.strip():
                                    final_content = f"""# 研究草稿（未完成）

⚠️ **系统提示**: 报告生成过程中出现异常，仅完成 {part_count} 个章节。建议重新提问以获取完整报告。

---

{merged.strip()}

---

💡 **建议**: 
- 重新提交相同问题以获取完整报告
"""
                                    final_report_path.write_text(final_content, encoding='utf-8')
                                    self.logger.info(
                                        f"[降级兜底A] 已保存草稿 ({len(final_content)} 字符)"
                                    )
                            else:
                                # >=3个章节，基本可用，添加警告说明
                                self.logger.info(
                                    f"[降级兜底A] 成功合并 {part_count} 个章节（添加警告说明）"
                                )
                                merged = ""
                                for pf in part_files:
                                    try:
                                        merged += pf.read_text(encoding='utf-8') + "\n\n"
                                    except Exception:
                                        pass
                                if merged.strip():
                                    final_content = f"""{merged.strip()}

---

⚠️ **编辑说明**: 本报告因系统异常未能完成最终审校和参考文献整理，内容仅供参考。如需完整报告，建议重新提问。
"""
                                    final_report_path.write_text(final_content, encoding='utf-8')
                                    self.logger.info(
                                        f"[降级兜底A] 成功合并为 final_report.md ({len(final_content)} 字符)"
                                    )
                except Exception as fallback_err:
                    self.logger.warning(f"[降级兜底A] 自动合并 part_*.md 失败: {fallback_err}")

            execution_time = time.time() - start_time
            # Extract final result
            if task_completed:
                # Find the completion result in the trace
                completion_result = None
                for step in reversed(self.reasoning_trace):
                    if step.get("type") == "action" and step.get("tool") in ["writer_subjective_task_done"]:
                        completion_result = step.get("result")
                        break
                return self.create_response(
                    success=True,
                    result=completion_result,
                    iterations=iteration,
                    execution_time=execution_time
                )
            else:

                return self.create_response(
                    success=False,
                    error=f"Writing task not completed within {self.config.max_iterations} iterations",
                    iterations=iteration,
                    execution_time=execution_time
                )

        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            self.logger.error(f"Error in execute_react_loop: {repr(e)}")

            return self.create_response(
                success=False,
                error=str(e),
                iterations=iteration if 'iteration' in locals() else 0,
                execution_time=execution_time
            )


# Factory function for creating the writer agent
def create_writer_agent(
        model: Any = None,
        max_iterations: int = 15,  # More iterations for writing tasks
        temperature: Any = None,  # Resolved from env if not provided
        max_tokens: Any = None,
        shared_mcp_client=None,
        task_id: str = None
) -> WriterAgent:
    """
    Create a WriterAgent instance with server-managed sessions.
    
    Args:
        model: The LLM model to use
        max_iterations: Maximum number of iterations for writing tasks
        temperature: Temperature setting for creativity
        max_tokens: Maximum tokens for the AI response
        shared_mcp_client: Optional shared MCP client from parent agent (prevents extra sessions)
        task_id: Optional task ID for progress tracking

    Returns:
        Configured WriterAgent instance with writing-focused tools
    """
    # Import the enhanced config function
    from .base_agent import create_agent_config

    # Create agent configuration (session managed by MCP server)
    config = create_agent_config(
        agent_name="WriterAgent",
        model=model,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Create agent instance with shared MCP client (filtered tools for writing)
    agent = WriterAgent(config=config, shared_mcp_client=shared_mcp_client, task_id=task_id)

    return agent
