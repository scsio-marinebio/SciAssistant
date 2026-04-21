# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
import json
from typing import Dict, Any, List
import time
import requests
import os
from .base_agent import BaseAgent, AgentConfig, AgentResponse, TaskInput
from config.logging_config import get_logger
logger = get_logger()



class InformationSeekerAgent(BaseAgent):
    """
    Information Seeker Agent that follows ReAct pattern (Reasoning + Acting)
    
    This agent takes decomposed sub-questions or tasks from parent agents,
    thinks interleaved (reasoning -> action -> reasoning -> action),
    uses MCP tools to gather information, and returns structured results.
    """
    
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
        tool_schemas_str = json.dumps(self.tool_schemas, ensure_ascii=False)
        
        # Read search source preferences from environment variables
        use_websearch = os.environ.get('SEARCH_SOURCE_WEBSEARCH', 'True').lower() == 'true'
        use_pubmed = os.environ.get('SEARCH_SOURCE_PUBMED', 'True').lower() == 'true'
        use_arxiv = os.environ.get('SEARCH_SOURCE_ARXIV', 'True').lower() == 'true'
        use_google_scholar = os.environ.get('SEARCH_SOURCE_GOOGLE_SCHOLAR', 'True').lower() == 'true'
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
            # 'springer': ['springer']  # DISABLED
        }
        
        # Dynamically filter tools based on environment variables
        enabled_tools = []
        disabled_tools = []
        
        # Log environment variable values for debugging
        logger.info(f"[SEARCH_SOURCE_DEBUG] WebSearch={use_websearch}, PubMed={use_pubmed}, arXiv={use_arxiv}, GoogleScholar={use_google_scholar}")
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
            # elif use_springer and any(pattern in tool_lower for pattern in tool_category_patterns['springer']):
            #     is_enabled = True
            #     logger.info(f"[SEARCH_SOURCE_DEBUG] Tool '{tool_name}' matched Springer pattern and is_enabled={is_enabled}")
            
            # Categorize tool
            if any(pattern in tool_lower for pattern in tool_category_patterns['websearch'] + tool_category_patterns['pubmed'] + tool_category_patterns['arxiv'] + tool_category_patterns['google_scholar']):
                if is_enabled:
                    enabled_tools.append(tool_name)
                else:
                    disabled_tools.append(tool_name)
        
        logger.info(f"[SEARCH_SOURCE_DEBUG] Enabled tools: {enabled_tools}")
        logger.info(f"[SEARCH_SOURCE_DEBUG] Disabled tools: {disabled_tools}")
        
        # Build search source guidance message with priority strategy
        search_source_guidance = ""
        if enabled_tools:
            # Categorize tools by type
            api_tools = [t for t in enabled_tools if any(p in t.lower() for p in ['arxiv', 'pubmed', 'medrxiv', 'google_scholar', 'scholar'])]
            web_tools = [t for t in enabled_tools if any(p in t.lower() for p in ['web_search', 'batch_web'])]
            other_tools = [t for t in enabled_tools if t not in api_tools and t not in web_tools]
            
            search_source_guidance = f"\n\n**📚 SEARCH STRATEGY (按优先级使用):**\n\n"
            
            # Priority 1: Specialized Academic APIs
            if api_tools:
                search_source_guidance += f"**🥇 优先级 1 - 专有学术 API** (强烈推荐优先使用):\n"
                for tool in api_tools:
                    if 'arxiv' in tool.lower():
                        search_source_guidance += f"  • **{tool}**: arXiv 预印本库（计算机科学、物理、数学等）\n"
                        search_source_guidance += f"    ✅ 完整元数据 | ✅ 直接下载 PDF 全文 | ✅ 无访问限制 | ✅ 100% 学术内容\n"
                    elif 'pubmed' in tool.lower():
                        search_source_guidance += f"  • **{tool}**: PubMed 权威医学文献数据库\n"
                        search_source_guidance += f"    ✅ MeSH 主题词标注 | ✅ 部分 PMC 全文 | ✅ 结构化元数据 | ✅ 医学权威来源\n"
                    elif 'medrxiv' in tool.lower():
                        search_source_guidance += f"  • **{tool}**: medRxiv 最新医学预印本\n"
                        search_source_guidance += f"    ✅ 最新医学研究 | ✅ 直接下载 PDF | ✅ 按类别组织 | ✅ 快速发布\n"
                    elif 'scholar' in tool.lower():
                        search_source_guidance += f"  • **{tool}**: Google Scholar 全学科学术搜索\n"
                        search_source_guidance += f"    ✅ 覆盖所有学科 | ✅ 引用数据 | ✅ 跨数据库搜索 | ✅ 可用 google_scholar_get_paper 获取论文内容\n"
                    # elif 'springer' in tool.lower():
                    #     search_source_guidance += f"  • **{tool}**: Springer Nature 期刊文章\n"
                    #     search_source_guidance += f"    ✅ 高质量期刊 | ✅ 完整元数据 | ✅ DOI 标识\n"
                search_source_guidance += f"\n"
            
            # Priority 2: Web Search
            if web_tools:
                search_source_guidance += f"**🥈 优先级 2 - 网页搜索** (补充使用):\n"
                for tool in web_tools:
                    search_source_guidance += f"  • **{tool}**: 通用网页搜索（已配置学术网站定向）\n"
                    search_source_guidance += f"    ✅ 覆盖面广 | ✅ 多领域内容 | ⚠️ 部分内容可能受访问限制（成功率 ~89%）\n"
                    search_source_guidance += f"    💡 适用场景: 工业应用、技术博客、新闻报道、非学术内容\n"
                search_source_guidance += f"\n"
            
            # Recommended workflow
            search_source_guidance += f"**💡 推荐工作流程**:\n"
            if api_tools:
                search_source_guidance += f"1. **首选**: 使用专有学术 API 获取核心学术文献\n"
                search_source_guidance += f"   - 优势: 完整元数据、高成功率（99%+）、可直接下载 PDF 全文\n"
                search_source_guidance += f"   - 示例: arxiv_search → arxiv_read_paper 获取完整论文\n"
                search_source_guidance += f"   - 示例: google_scholar_search 搜索全学科论文，再用对应源工具获取全文\n"
            if web_tools:
                search_source_guidance += f"2. **补充**: 使用网页搜索获取其他来源内容\n"
                search_source_guidance += f"   - 用途: 工业应用案例、技术博客、新闻动态、非学术资源\n"
                search_source_guidance += f"   - 注意: 部分网站可能有反爬虫限制或订阅墙\n"
            search_source_guidance += f"3. **深入**: 对重要论文使用 read_paper 工具获取完整全文\n"
            search_source_guidance += f"   - arxiv_read_paper, get_pubmed_article, medrxiv_read_paper 等\n"
            search_source_guidance += f"   - Google Scholar: google_scholar_search → google_scholar_get_paper 获取论文内容\n\n"
            
            if disabled_tools:
                search_source_guidance += f"⚠️ **不可用工具**: {', '.join(disabled_tools)}\n"
                search_source_guidance += f"如果尝试使用这些工具会收到错误，请使用上述可用工具。\n\n"
        else:
            search_source_guidance = f"\n\n**⚠️ WARNING: ALL SEARCH TOOLS DISABLED**\n"
            search_source_guidance += f"No external search tools are available in this session. You can only work with existing files in the workspace (user_uploads/, library_refs/, etc.).\n"
            search_source_guidance += f"Focus on analyzing existing documents and files using document_extract, document_qa, and file operations.\n"
        
        # Add current date for time awareness
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        system_prompt_template = f"""You are an Information Seeker Agent that follows the ReAct pattern (Reasoning + Acting).

## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**Detect the language of the user's query/task and respond accordingly:**
- **English query → Respond in English**
- **Chinese query (中文) → Respond in Chinese (中文回复)**
- **Mixed Chinese-English query → Respond in Chinese (中文回复)**
This rule applies to ALL outputs including: task summaries, findings, and any content in task_done reports.
        
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
        
        0. **MANDATORY FIRST STEP - Check Workspace for Existing Files:**
           - Check `./user_uploads/` directory for user-uploaded files (HIGH PRIORITY)
           - Check `./library_refs/` directory for user-selected library files (NORMAL PRIORITY)
           - **CRITICAL REQUIREMENT:** When calling `document_extract`, you MUST include ALL document files from BOTH directories:
             * Include ALL .pdf, .doc, .docx files (source documents)
             * Include ALL .txt files that are NOT converted from other documents (e.g., research/*.txt)
             * The system will automatically skip .pdf.txt, .doc.txt, .docx.txt if the source file exists
           - **DO NOT FILTER FILES:** Do NOT make assumptions about file relevance based on filenames
           - **DO NOT SELECT SUBSET:** Do NOT choose only "relevant-looking" files - analyze ALL files
           - **MANDATORY:** If library_refs has 12 files, you MUST pass all 12 files to document_extract
           - **CRITICAL:** Do NOT skip library_refs files even if user_uploads has files
           - Only proceed to web search after analyzing existing files

        
        1. INITIAL RESEARCH:{search_source_guidance}
           - Generate focused search queries (≤10): Limit to no more than 10 initial search queries to avoid increased failure rates from excessive decomposition.
           - **RECOMMENDED TOOL SELECTION STRATEGY**:
                a) **Prefer specialized academic APIs** when query matches their coverage:
                   • Biology/Medical topics → "search_pubmed_key_words", "search_pubmed_advanced", "medrxiv_search"
                   • Computer Science/Math/Physics topics → "arxiv_search"
                   • Cross-discipline broad academic search → "google_scholar_search", "advanced_google_scholar_search"
                   • Advantages: Complete metadata, direct PDF access, 99%+ success rate
                b) **Use "batch_web_search"** for:
                   • Topics outside specialized API coverage (engineering, social sciences, multi-disciplinary, etc.)
                   • Industry applications, technical blogs, news, case studies
                   • Supplementary content to complement academic sources
                   • Non-English academic content
                   • Note: Web search has academic site targeting enabled (academic sites including arXiv, Nature, IEEE, etc.)
                c) **Use Google Scholar** for cross-discipline searches or when unsure which specialized API to use - results may link to arXiv, PubMed, etc.
                d) **Combine all approaches** when appropriate - use specialized APIs for core academic papers, Google Scholar for broad discovery, and web search for broader context
           - When calling web search, consider the language of the user's question (e.g., use Chinese for Chinese questions)
           - Analyze the search results (titles, snippets, URLs, paper metadata) to identify promising sources
        
        2. CONTENT EXTRACTION:  
           - **CRITICAL: Special handling for academic paper URLs**:
                • If a URL from "batch_web_search" is an arXiv paper (e.g., https://arxiv.org/abs/XXXX.XXXXX):
                  → Extract the paper_id (e.g., "1206.3218" from "https://arxiv.org/abs/1206.3218")
                  → Use "arxiv_read_paper" with the paper_id (NOT url_crawler or download_files)
                  → This ensures you get the full paper content, not just the HTML abstract page
                • Similarly for other academic sources: PubMed URLs → get_pubmed_article, etc.
           - For important URLs searched by "batch_web_search", use `url_crawler` to:  
                a) Extract full content from the webpage  
                b) Save the content to a file in the workspace **under the relative path `./url_crawler_save_files/`**
                c) **Exception**: Do NOT use url_crawler for arXiv/PubMed/medRxiv URLs - use their dedicated tools instead
                d) For Google Scholar results: check if the URL points to arXiv/PubMed/medRxiv, and if so use the corresponding dedicated tool instead of url_crawler
           - For important articles searched with pubmed, medrxiv, or arxiv, use the corresponding retrieval tools:
                a) PubMed: "get_pubmed_article" (requires PMID from search results)
                b) medRxiv: "medrxiv_read_paper" (requires paper_id from search results)
                c) arXiv: "arxiv_read_paper" (requires paper_id from search results)
           - Store results with meaningful file paths (e.g., `url_crawler_save_files/research/ai_trends_2024.txt`)
        
        3. CONTENT ANALYSIS:
           - Use `document_qa` to ask specific questions about the saved files:
                a) Formulate focused questions to extract key insights
                b) Use answers to deepen your understanding
           - You can ask multiple questions about the same file
           - Use `document_extract` for multi-dimensional analysis of saved files:
                a) Provides structured analysis across five key dimensions: doc time source authority, core content and task relevance
        
        4. FILE MANAGEMENT:
           - Use `file_write` to save important findings or summaries
           - For reviewing saved content:
                a) Prefer `document_extract` to get comprehensive multi-dimensional analysis of saved files
                b) Use `file_read` ONLY for small files (<1000 tokens) when you need the entire content
                c) Avoid reading large files directly as it may exceed context limits
        
        5. TASK COMPLETION:
           - When ready to report, call `info_seeker_subjective_task_done` with:
                a) Comprehensive markdown summary of your process and findings
                b) List of key files created with descriptions
        
        ### Usage of Systematic Tool:
            - `think` is a systematic tool. After receiving the response from the complex tool or before invoking any other tools, you must **first invoke the `think` tool**: to deeply reflect on the results of previous tool invocations (if any), and to thoroughly consider and plan the user's task. The `think` tool does not acquire new information; it only saves your thoughts into memory.
            - `reflect` is a systematic tool. When encountering a failure in tool execution, it is necessary to invoke the reflect tool to conduct a review and revise the task plan. It does not acquire new information; it only saves your thoughts into memory.
        
        Always provide clear reasoning for your actions and synthesize information effectively.

Below, within the <tools></tools> tags, are the descriptions of each tool and the required fields for invocation:
<tools>
$tool_schemas
</tools>
For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
[unused11][{{"name": <function name>, "arguments": <args json object>}}][unused12]
"""
        return system_prompt_template.replace("$tool_schemas", tool_schemas_str)

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
            # Get model configuration from config
            from config.config import get_config
            config = get_config()
            model_config = config.get_custom_llm_config()
            
            pangu_url = model_config.get('url') or os.getenv('MODEL_REQUEST_URL', '')
            model_token = model_config.get('token') or os.getenv('MODEL_REQUEST_TOKEN', '')
            headers = {'Content-Type': 'application/json', 'csb-token': model_token}

            # ReAct Loop: Reasoning -> Acting -> Reasoning -> Acting...
            while iteration < self.config.max_iterations and not task_completed:
                iteration += 1
                self.logger.info(f"Planning iteration {iteration}")
                
                try:
                    # Get LLM response (reasoning + potential tool calls)
                    retry_num = 1
                    max_retry_num = 10
                    while retry_num < max_retry_num:
                        try:
                            response = requests.post(
                                url=pangu_url,
                                headers=headers,
                                json={
                                    "model": model_config.get('model', 'pangu_auto'),
                                    "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<s>[unused9]系统：[unused10]' }}{% endif %}{% if message['role'] == 'system' %}{{'<s>[unused9]系统：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'assistant' %}{{'[unused9]助手：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'tool' %}{{'[unused9]工具：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'function' %}{{'[unused9]方法：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'user' %}{{'[unused9]用户：' + message['content'] + '[unused10]'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '[unused9]助手：' }}{% endif %}",
                                    "messages": conversation_history,
                                    "spaces_between_special_tokens": False,
                                    "temperature": self.config.temperature,
                                },
                                timeout=model_config.get("timeout", 180)
                            )
                            response = response.json()

                            self.logger.debug(f"API response received")
                            break
                        except Exception as e:
                            time.sleep(3)
                            retry_num += 1
                            if retry_num == max_retry_num:
                                raise ValueError(str(e))
                            continue

                    assistant_message = response["choices"][0]["message"]
                    # Log the reasoning
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
                                tool_calls = json.loads(tool_call_str[0].strip())
                            except Exception as ee:
                                return ["fail_tools_load", ee]
                        else:
                            return []
                        return tool_calls
                    
                    # Add assistant message to conversation
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_message["content"]
                    })
                    
                    tool_calls = extract_tool_calls(assistant_message["content"])

                    if len(tool_calls) > 0 and tool_calls[0] == "fail_tools_load":
                        # Parse error, rerun
                        followup_prompt = f"There was a parsing error in the format of the tool call" \
                                          f" you generated:{tool_calls[1]} Please regenerate it."
                        conversation_history.append({"role": "user", "content": followup_prompt + " /no_think"})
                        continue


                    # Execute tool calls if any (Acting phase)

                    for tool_call in tool_calls:
                        arguments = tool_call["arguments"]

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
                        conversation_history.append({
                            "role": "tool",
                            "content": json.dumps(tool_result, ensure_ascii=False, indent=2) + " /no_think"
                        })
                    
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
                    
                    
                except Exception as e:
                    error_msg = f"Error in planning iteration {iteration}: {e}"
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
                    error=f"Task not completed within {self.config.max_iterations} iterations",
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
    model: str = "pangu_auto",
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
