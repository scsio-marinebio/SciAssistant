# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
import json
import re
from typing import Dict, Any, List, Optional
import time
import os
from pathlib import Path
from .base_agent import BaseAgent, AgentConfig, AgentResponse, WriterAgentTaskInput
from .. import get_thread_human_in_loop_phase2, get_thread_workspace_path
from ..utils.report_quality import normalize_and_validate_report
from config.config import (
    extract_reasoning_from_response,
    extract_tool_calls_from_response,
    get_tool_call_format_instruction,
    get_tool_schemas_prompt,
)
from src.utils.llm_client import LLMOutputTruncatedError, llm_chat


class WriterAgent(BaseAgent):
    """
    Writer Agent that follows ReAct pattern for content synthesis and generation
    
    This agent takes writing tasks from parent agents, searches through existing
    files and knowledge base, and creates long-form content through iterative
    reasoning and refinement. It does NOT access internet resources, only
    local files and memories.
    """

    MAX_TRUNCATION_CORRECTIONS = 2
    TRUNCATION_CORRECTION_PROMPT = (
        "上一轮响应因达到输出长度上限而被系统整体丢弃，其中没有任何工具调用被执行。"
        "不要假设上一轮的任何步骤已经完成。"
        "本轮只输出当前下一步所需的一个工具调用，参数必须完整，"
        "不要同时生成后续章节、合并和完成调用。 /no_think"
    )

    def __init__(self, config: AgentConfig = None, shared_mcp_client=None, task_id: str = None):
        self._bound_session_id = getattr(shared_mcp_client, "_session_id", None)
        # Set default agent name if not specified
        if config is None:
            config = AgentConfig(agent_name="WriterAgent")
        elif config.agent_name == "base_agent":
            config.agent_name = "WriterAgent"

        super().__init__(config, shared_mcp_client)

        if not self._bound_session_id:
            client = getattr(self.mcp_tools, "client", self.mcp_tools)
            self._bound_session_id = getattr(client, "_session_id", None)

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
        self._trusted_section_writer_context = {}
        self._trusted_chapter_summaries = []
        self._tool_validation_failure_counts = {}
        self._last_tool_validation_failure_fingerprint = None
        self._consecutive_tool_validation_failures = 0
        self._trusted_classification_sections = []
        self._tool_validation_repair_attempts = {}
        self._tool_validation_last_generation = {}

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

    def _reset_tool_argument_guard(
        self,
        task_input: WriterAgentTaskInput,
        overall_outline: str = ""
    ) -> None:
        """Reset trusted tool context and consecutive validation state."""
        self._trusted_section_writer_context = {
            "user_query": getattr(task_input, "user_query", ""),
            "task_content": getattr(task_input, "task_content", ""),
            "overall_outline": overall_outline or "",
            "key_files": list(getattr(task_input, "key_files", []) or []),
        }
        self._trusted_chapter_summaries = []
        self._tool_validation_failure_counts = {}
        self._last_tool_validation_failure_fingerprint = None
        self._consecutive_tool_validation_failures = 0
        self._trusted_classification_sections = []
        self._tool_validation_repair_attempts = {}
        self._tool_validation_last_generation = {}

    def _workspace_root(self):
        """Resolve only the workspace pinned to this Writer's MCP client."""
        session_id = (self._bound_session_id or "").strip()
        if not session_id:
            self.logger.error("Writer has no bound MCP session; refusing workspace file operations.")
            return None

        workspace_getter = globals().get("get_thread_workspace_path")
        env_workspace = (
            workspace_getter().strip()
            if callable(workspace_getter)
            else os.environ.get("AGENT_WORKSPACE_PATH", "").strip()
        )
        if env_workspace:
            candidate = Path(env_workspace)
            if candidate.name == session_id:
                return candidate
            self.logger.warning(
                "Ignoring mismatched AGENT_WORKSPACE_PATH for bound session %s: %s",
                session_id,
                candidate,
            )

        return Path(__file__).resolve().parents[3] / "workspaces" / session_id

    def _workspace_report_dir(self):
        """Return this Writer instance's report directory, or None when unbound."""
        workspace_root = self._workspace_root()
        return workspace_root / "report" if workspace_root else None

    def _is_final_report_ready(self) -> bool:
        """Require both successful merge state and a real non-empty report file."""
        report_dir = self._workspace_report_dir()
        final_report_path = report_dir / "final_report.md" if report_dir else None
        return bool(
            self._has_merged_final_report
            and final_report_path
            and final_report_path.is_file()
            and final_report_path.stat().st_size > 0
        )

    def _hydrate_search_result_classifier_tool_call(
        self,
        tool_call: Dict[str, Any]
    ) -> List[str]:
        """Fill only classifier inputs already owned by WriterAgent."""
        if tool_call.get("name") != "search_result_classifier":
            return []
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return []

        trusted_values = {
            "key_files": self._trusted_section_writer_context.get("key_files"),
            "outline": self._trusted_section_writer_context.get("overall_outline"),
        }
        hydrated_fields = []
        for field_name, trusted_value in trusted_values.items():
            if field_name in arguments or trusted_value in (None, "", []):
                continue
            arguments[field_name] = trusted_value
            hydrated_fields.append(field_name)
        if hydrated_fields:
            self.logger.warning(
                "[search_result_classifier argument hydration] fields=%s",
                hydrated_fields,
            )
        return hydrated_fields

    def _classification_contract_error(
        self,
        arguments: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> str:
        """Independently reject classifier data that cannot safely drive writing."""
        if not tool_result.get("success"):
            return ""
        result = tool_result.get("data")
        outline = arguments.get("outline", "") if isinstance(arguments, dict) else ""
        key_files = arguments.get("key_files", []) if isinstance(arguments, dict) else []
        if not isinstance(result, str) or not result.strip():
            return "Classifier returned empty formal content."
        paragraph_markers = re.findall(r"(?im)^\s*paragraph\s+\d+\s*:", result)
        file_markers = re.findall(r"(?im)^\s*file_path_list\s*:", result)
        if not paragraph_markers or len(paragraph_markers) != len(file_markers):
            return "Classifier markers are missing or unbalanced."

        sections = self._parse_classification_result(result)
        expected_sections = self._build_sections_from_outline(outline)
        if len(sections) != len(expected_sections):
            return f"Classifier chapter count mismatch: expected {len(expected_sections)}, got {len(sections)}."

        known_paths = {
            item.get("file_path") for item in (key_files or [])
            if isinstance(item, dict) and isinstance(item.get("file_path"), str)
        }
        for index, (actual, expected) in enumerate(zip(sections, expected_sections), start=1):
            actual_lines = [line.strip() for line in actual.get("outline", "").splitlines() if line.strip()]
            expected_lines = [line.strip() for line in expected.get("outline", "").splitlines() if line.strip()]
            if actual_lines != expected_lines:
                return f"Classifier outline mismatch in paragraph {index}."
            paths = actual.get("file_paths", [])
            if not paths:
                return f"Classifier paragraph {index} has no assigned files."
            unknown = [path for path in paths if path not in known_paths]
            if unknown:
                return f"Classifier paragraph {index} contains unknown files: {unknown}."
        return ""

    def _enforce_classification_contract(
        self,
        arguments: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> None:
        error = self._classification_contract_error(arguments, tool_result)
        if not error:
            return
        self.logger.error("Writer rejected untrusted classifier result: %s", error)
        tool_result.update({
            "success": False,
            "data": None,
            "error": f"Untrusted classification result: {error}",
            "error_code": "CLASSIFICATION_VALIDATION_FAILED",
        })

    def _remember_classification_context(
        self,
        arguments: Dict[str, Any],
        tool_result: Dict[str, Any]
    ) -> None:
        """Remember successful classifier output as trusted chapter context."""
        if not tool_result.get("success"):
            return
        outline = arguments.get("outline", "") if isinstance(arguments, dict) else ""
        if isinstance(outline, str) and outline.strip():
            self._trusted_section_writer_context["overall_outline"] = outline.strip()
        classification_result = tool_result.get("data", "")
        sections = self._parse_classification_result(classification_result)
        self._trusted_classification_sections = sections

    def _hydrate_section_writer_tool_call(self, tool_call: Dict[str, Any]) -> List[str]:
        """Fill only deterministic section_writer fields already owned by WriterAgent."""
        if tool_call.get("name") != "section_writer":
            return []
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return []

        supplied_outline = arguments.get("overall_outline")
        if supplied_outline and not self._trusted_section_writer_context.get("overall_outline"):
            self._trusted_section_writer_context["overall_outline"] = supplied_outline

        if self._trusted_chapter_summaries:
            trusted_summary = "\n".join(self._trusted_chapter_summaries)
        elif not getattr(self, "_written_chapters", set()):
            trusted_summary = "No previous chapters written yet."
        else:
            trusted_summary = None

        trusted_values = {
            "user_query": self._trusted_section_writer_context.get("user_query"),
            "task_content": self._trusted_section_writer_context.get("task_content"),
            "written_chapters_summary": trusted_summary,
            "overall_outline": self._trusted_section_writer_context.get("overall_outline"),
        }
        section_index = max(int(getattr(self, "_expected_next_chapter", 1)) - 1, 0)
        sections = getattr(self, "_trusted_classification_sections", []) or []
        if section_index < len(sections):
            section = sections[section_index]
            trusted_values.update({
                "current_chapter_outline": section.get("outline"),
                "target_file_path": f"./report/part_{section_index + 1}.md",
                "key_files": self._select_key_files_by_paths(
                    self._trusted_section_writer_context.get("key_files", []),
                    section.get("file_paths", []),
                ),
            })

        hydrated_fields = []
        for field_name, trusted_value in trusted_values.items():
            if field_name in arguments or trusted_value in (None, ""):
                continue
            arguments[field_name] = trusted_value
            hydrated_fields.append(field_name)

        if hydrated_fields:
            self.logger.warning(
                "[section_writer argument hydration] fields=%s expected_chapter=%s",
                hydrated_fields,
                self._expected_next_chapter,
            )
        return hydrated_fields

    def _remember_section_writer_summary(self, tool_result: Dict[str, Any]) -> None:
        """Store successful summaries for deterministic subsequent-call hydration."""
        if not tool_result.get("success"):
            return
        data = tool_result.get("data") or {}
        summary = data.get("chapter_summary", "") if isinstance(data, dict) else ""
        if isinstance(summary, str) and summary.strip():
            self._trusted_chapter_summaries.append(summary.strip())

    def _record_tool_validation_failure(
        self,
        tool_name: str,
        tool_result: Dict[str, Any],
        generation_id: Optional[int] = None,
    ) -> bool:
        """Offer one forced correction window before stopping repeated bad calls.

        Three identical failures trigger a targeted correction. Three more
        identical failures after that correction exhaust the local recovery
        budget and stop the Writer, preventing unbounded token consumption.
        """
        if not isinstance(getattr(self, "_tool_validation_repair_attempts", None), dict):
            self._tool_validation_repair_attempts = {}
        if not isinstance(getattr(self, "_tool_validation_last_generation", None), dict):
            self._tool_validation_last_generation = {}
        if tool_result.get("error_code") != "TOOL_ARGUMENT_VALIDATION_FAILED":
            self._tool_validation_failure_counts.clear()
            self._tool_validation_repair_attempts.clear()
            self._tool_validation_last_generation.clear()
            self._last_tool_validation_failure_fingerprint = None
            self._consecutive_tool_validation_failures = 0
            return False
        fingerprint = (
            tool_name,
            tuple(sorted(tool_result.get("missing_required_fields", []))),
            tuple(sorted(tool_result.get("invalid_fields", []))),
        )
        if (
            generation_id is not None
            and self._tool_validation_last_generation.get(fingerprint) == generation_id
        ):
            tool_result["validation_duplicate_in_generation"] = True
            tool_result["validation_generation_id"] = generation_id
            return False
        if generation_id is not None:
            self._tool_validation_last_generation[fingerprint] = generation_id
        if fingerprint == self._last_tool_validation_failure_fingerprint:
            failure_count = self._consecutive_tool_validation_failures + 1
        else:
            self._tool_validation_failure_counts.clear()
            self._tool_validation_repair_attempts.clear()
            failure_count = 1
        self._last_tool_validation_failure_fingerprint = fingerprint
        self._consecutive_tool_validation_failures = failure_count
        self._tool_validation_failure_counts[fingerprint] = failure_count
        tool_result["validation_failure_count"] = failure_count
        if failure_count < 3:
            return False

        repair_attempt = self._tool_validation_repair_attempts.get(fingerprint, 0) + 1
        if repair_attempt <= 1:
            self._tool_validation_repair_attempts[fingerprint] = repair_attempt
            tool_result["retryable"] = True
            tool_result["validation_retry_escalated"] = True
            tool_result["validation_repair_attempt"] = repair_attempt

            requires_classification = (
                tool_name == "section_writer"
                and not (getattr(self, "_trusted_classification_sections", []) or [])
            )
            if requires_classification:
                instruction = (
                    "Do not call section_writer again yet. First call "
                    "search_result_classifier with one complete argument object; "
                    "after classification succeeds, retry section_writer."
                )
            else:
                instruction = (
                    f"Retry exactly one complete {tool_name} call containing all "
                    "required arguments. Do not combine multiple tool calls."
                )
            tool_result["error"] = f"{tool_result.get('error', '')} {instruction}".strip()
            self.logger.warning(
                "[tool argument forced correction] tool=%s fingerprint=%s count=%s repair=%s",
                tool_name,
                fingerprint,
                failure_count,
                repair_attempt,
            )
            self._tool_validation_failure_counts.clear()
            self._last_tool_validation_failure_fingerprint = fingerprint
            self._consecutive_tool_validation_failures = 0
            return False

        tool_result["retryable"] = False
        tool_result["validation_repair_exhausted"] = True
        tool_result["circuit_breaker_open"] = True
        tool_result["validation_repair_attempt"] = repair_attempt
        tool_result["error"] = (
            f"{tool_result.get('error', '')} Forced argument correction was exhausted "
            "after another three identical invalid calls; stopping this writer run."
        ).strip()
        self.logger.error(
            "[tool argument recovery exhausted] tool=%s fingerprint=%s count=%s repair=%s",
            tool_name,
            fingerprint,
            failure_count,
            repair_attempt,
        )
        return True

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
        tool_schemas_str = get_tool_schemas_prompt(self.tool_schemas)
        tool_call_format = get_tool_call_format_instruction()
        
        # 根据_is_chinese_query标志明确指定输出语言
        _is_cn = getattr(self, '_is_chinese_query', True)
        if _is_cn:
            language_instruction = """## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**You MUST write the ENTIRE article in Chinese (中文).**
This rule applies to ALL outputs including: outline generation, chapter content, summaries, and the final article.
**DO NOT use English, Korean (한국어), Japanese (日本語), or any other language for explanatory text.**
Technical terms may remain in English, but all explanatory text MUST be in Chinese.
**所有内容必须使用中文撰写，包括：大纲生成、章节内容、摘要和最终文章。**"""
        else:
            language_instruction = """## 🌐 CRITICAL: Response Language Rules (MUST FOLLOW)
**You MUST write the ENTIRE article in English.**
This rule applies to ALL outputs including: outline generation, chapter content, summaries, and the final article.
**DO NOT use Chinese (中文), Korean (한국어), Japanese (日本語), or any other language.**
All content including headings, body text, tables, and citations MUST be in English only."""

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
For each function call, return a JSON object with function name and arguments:
{tool_call_format}

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
For each function call, return a JSON object with function name and arguments:
{tool_call_format}
"""
# For each function call, return a JSON object placed within the [unused11][unused12] tags, which includes the function name and the corresponding function arguments:
# [unused11][{{"name": <function name>, "arguments": <args json object>}}][unused12]
# """
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
                
                parsed_json = json.loads(raw_result.data["content"][0]["text"])
                res = parsed_json.get("data") if isinstance(parsed_json, dict) else []
                if res is None:
                    res = []
                                            
            except Exception as e:
                self.logger.error(f"Error loading file {file_path} from MCP server: {e}")
                import traceback
                self.logger.debug(f"Full traceback: {traceback.format_exc()}")
                
            return res if isinstance(res, list) else []

        key_files_dict = {}
        # 与 section_writer / merge_reports 共用解析后列表中的来源编号。
        # 过滤只隐藏资料，不压缩后续来源编号；最终连续编号由合并阶段生成。
        file_path_to_source_num = {}

        server_analysis_path = f"doc_analysis/file_analysis.jsonl"
        self.logger.debug(f"Loading analysis from MCP server: {server_analysis_path}")
        file_analysis_list = load_json_from_server(server_analysis_path)

        # 【智能过滤】基于information_richness字段判断，而不是关键词匹配
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
                
                key_files_dict[file_path] = file_info
                file_path_to_source_num[file_path] = line_num
                self.logger.debug(f"映射来源编号 {line_num} 到文件: {file_path}")

        file_core_content = ""
        valid_file_paths = []  # 收集有效文件路径用于推送
        source_background = ""
        # Optional retrieval enhancement: errors must preserve the original handoff.
        # Keep source IDs from the full analysis index; never renumber here.
        original_key_files = getattr(task_input, 'key_files', None)
        trusted_context = getattr(self, '_trusted_section_writer_context', None)
        original_trusted_files = trusted_context.get('key_files') if trusted_context is not None else None
        try:
            from src.utils.writer_sources import select_original_sources
            original_paths, background_paths = select_original_sources(
                key_files_dict, getattr(task_input, 'key_files', []) or [],
                getattr(task_input, 'user_query', ''),
            )
            if background_paths:
                source_background = (
                    "\nBackground summaries (organization only, not independent evidence):\n"
                    + "\n".join(
                        f"{path}: {str(key_files_dict[path].get('core_content', ''))[:1200]}"
                        for path in background_paths[:6]
                    )
                    + ("\nUse only original Key Files for citations; verify each claim against their content.\n"
                       if original_paths else
                       "\nNo verified original sources are available. Key Files are background routing only. "
                       "Write without numeric citations and clearly disclose the evidence limitation.\n")
                )
                selected_files = [{'file_path': path} for path in (original_paths or background_paths)]
                task_input.key_files = selected_files
                if getattr(self, '_trusted_section_writer_context', None) is not None:
                    self._trusted_section_writer_context['key_files'] = list(selected_files)
                self.logger.info(
                    "[WriterSources] selected %s original sources; %s confirmed generated backgrounds",
                    len(original_paths), len(background_paths),
                )
        except Exception as exc:
            source_background = ""
            if original_key_files is not None:
                task_input.key_files = original_key_files
            if trusted_context is not None:
                if original_trusted_files is None:
                    trusted_context.pop('key_files', None)
                else:
                    trusted_context['key_files'] = original_trusted_files
            self.logger.warning("[WriterSources] enhancement unavailable; keeping original handoff: %s", exc)
        if hasattr(task_input, 'key_files') and task_input.key_files:
            message += "Key Files:\n"
            valid_file_count = 0
            for file_ in task_input.key_files:
                file_path = file_.get('file_path')
                if file_path in key_files_dict:
                    valid_file_count += 1
                    valid_file_paths.append(file_path)  # 记录有效文件路径
                    # 资料数量与来源编号独立；不能用筛选后的计数重建来源身份。
                    source_num = file_path_to_source_num[file_path]
                    file_info = key_files_dict[file_path]
                    doc_time = file_info.get('doc_time', 'Not specified')
                    source_authority = file_info.get('source_authority', 'Not assessed')
                    task_relevance = file_info.get('task_relevance', 'Not assessed')
                    information_richness = file_info.get('information_richness', 'Not assessed')
                    message += f"{source_num}. File: {file_path}\n"

                    file_core_content += f"[{source_num}]doc_time:{doc_time}|||source_authority:{source_authority}|||task_relevance:{task_relevance}|||information_richness:{information_richness}|||summary_content:{file_info.get('core_content', '')}\n"
            
            # 【Fallback】只在匹配文件数极少（<3个）且明显异常时才回退
            # 原因：可能是路径不匹配问题，而非PlannerAgent的正常筛选
            # 注意：如果PlannerAgent有意只选择少量文件，此fallback可能违背其意图
            if valid_file_count < 3 and len(key_files_dict) > 10 and not source_background:
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
                    source_num = file_path_to_source_num[file_path]
                    doc_time = file_info.get('doc_time', 'Not specified')
                    source_authority = file_info.get('source_authority', 'Not assessed')
                    task_relevance = file_info.get('task_relevance', 'Not assessed')
                    information_richness = file_info.get('information_richness', 'Not assessed')
                    message += f"{source_num}. File: {file_path}\n"
                    file_core_content += f"[{source_num}]doc_time:{doc_time}|||source_authority:{source_authority}|||task_relevance:{task_relevance}|||information_richness:{information_richness}|||summary_content:{file_info.get('core_content', '')}\n"

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

        return message + source_background

    def _should_use_program_driven_writing(self, model_config: Dict[str, Any]) -> bool:
        """Enable normal program-driven sequential writing only for DeepSeek."""
        from config.config import get_model_provider
        model_name = model_config.get("model") or self.config.model
        provider = get_model_provider(model_name)
        if provider != "deepseek":
            return False
        toggle = os.environ.get("DEEPSEEK_PROGRAM_DRIVEN", "").strip().lower()
        if toggle:
            return toggle in {"1", "true", "yes", "on"}
        return True

    def _should_use_ultra_classifier_fallback(self, model_config: Dict[str, Any]) -> bool:
        """Enable classifier fallback only for the explicitly opted-in Ultra model."""
        from config.config import get_model_provider, is_pangu_ultra_moe_compat_enabled

        model_name = model_config.get("model") or self.config.model
        provider = model_config.get("provider") or get_model_provider(model_name)
        return (
            is_pangu_ultra_moe_compat_enabled(
                model_name=model_name,
                provider=provider,
                enabled=model_config.get("pangu_ultra_moe_compat_enabled", False),
            )
            and bool(model_config.get("pangu_ultra_writer_fallback_enabled", False))
        )

    @staticmethod
    def _parse_classification_result(classification_result: str) -> List[Dict[str, Any]]:
        """Parse search_result_classifier output into ordered chapter outlines and file paths."""
        if not classification_result or not isinstance(classification_result, str):
            return []

        sections: List[Dict[str, Any]] = []
        outline_lines: List[str] = []
        file_paths: List[str] = []
        has_paragraph_markers = bool(re.search(r"(?im)^\s*paragraph\s+\d+\s*:", classification_result))

        def flush_section():
            outline = "\n".join(outline_lines).strip()
            if outline:
                sections.append({
                    "outline": outline,
                    "file_paths": file_paths[:],
                })

        def extend_file_paths(paths_text: str) -> None:
            if not paths_text:
                return
            paths = re.split(r"[,，]", paths_text)
            for path in paths:
                cleaned = path.strip()
                if cleaned and cleaned not in file_paths:
                    file_paths.append(cleaned)

        def is_subheading(heading_text: str) -> bool:
            number_match = re.match(r"^(\d+(?:\.\d+)*)", heading_text)
            if not number_match:
                return False
            return "." in number_match.group(1)

        for raw_line in classification_result.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if re.match(r"^paragraph\s*\d+\s*:", stripped, re.IGNORECASE):
                flush_section()
                outline_lines = []
                file_paths = []
                content = stripped.split(":", 1)[1].strip()
                if content:
                    outline_lines.append(content)
                continue
            heading_match = re.match(r"^#{1,6}\s+(.+)", stripped)
            if heading_match and (not has_paragraph_markers or not outline_lines):
                heading_text = heading_match.group(1).strip()
                if not outline_lines or not is_subheading(heading_text):
                    flush_section()
                    outline_lines = []
                    file_paths = []
                outline_lines.append(stripped)
                continue
            if stripped.lower().startswith("file_path_list"):
                _, _, paths_text = stripped.partition(":")
                extend_file_paths(paths_text)
                continue
            if outline_lines:
                outline_lines.append(line)

        flush_section()
        return sections

    @staticmethod
    def _build_sections_from_outline(outline: str) -> List[Dict[str, Any]]:
        """Fallback: split outline into chapter sections when classification output is empty."""
        if not outline or not isinstance(outline, str):
            return []

        sections: List[Dict[str, Any]] = []
        current_lines: List[str] = []
        prefix_lines: List[str] = []

        def flush_section():
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({
                    "outline": content,
                    "file_paths": [],
                })

        for raw_line in outline.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_lines:
                    flush_section()
                    current_lines = [stripped]
                else:
                    current_lines = prefix_lines + [stripped]
                    prefix_lines = []
                continue
            if current_lines:
                current_lines.append(line)
            else:
                prefix_lines.append(line)

        flush_section()

        if not sections and outline.strip():
            sections.append({
                "outline": "\n".join(prefix_lines).strip() or outline.strip(),
                "file_paths": [],
            })

        return sections

    @staticmethod
    def _select_key_files_by_paths(
        key_files: List[Dict[str, Any]],
        file_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter key_files by file_path list, falling back to all files if none match."""
        if not key_files:
            return []

        lookup = {
            file_info.get("file_path"): file_info
            for file_info in key_files
            if file_info.get("file_path")
        }
        selected: List[Dict[str, Any]] = []
        for path in file_paths:
            if not path:
                continue
            selected.append(lookup.get(path, {"file_path": path}))

        return selected if selected else list(key_files)

    @staticmethod
    def _assign_ultra_fallback_key_files(
        sections: List[Dict[str, Any]],
        key_files: List[Dict[str, Any]],
        max_files_per_chapter: int = 11,
    ) -> List[Dict[str, Any]]:
        """Deterministically distribute bounded context without another classifier call."""
        limit = max(1, int(max_files_per_chapter or 1))
        valid_files: List[Dict[str, Any]] = []
        seen_paths = set()
        for item in key_files or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("file_path") or "").strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            valid_files.append(item)

        assigned: List[Dict[str, Any]] = []
        file_count = len(valid_files)
        for chapter_index, section in enumerate(sections or []):
            copied = dict(section)
            if file_count:
                take = min(limit, file_count)
                start = (chapter_index * take) % file_count
                selected = [valid_files[(start + offset) % file_count] for offset in range(take)]
                copied["file_paths"] = [item["file_path"] for item in selected]
            else:
                copied["file_paths"] = []
            assigned.append(copied)
        return assigned

    def _run_ultra_classifier_fallback(
        self,
        *,
        outline: str,
        task_input: WriterAgentTaskInput,
        model_config: Dict[str, Any],
        base_iteration: int,
    ) -> Dict[str, Any]:
        """Build trusted sections locally and reuse the program-driven writer."""
        max_files = int(model_config.get("pangu_ultra_fallback_max_files_per_chapter", 11))
        section_retries = int(model_config.get("pangu_ultra_fallback_section_retries", 1))
        sections = self._build_sections_from_outline(outline)
        sections = self._assign_ultra_fallback_key_files(
            sections,
            getattr(task_input, "key_files", []) or [],
            max_files_per_chapter=max_files,
        )
        self.logger.warning(
            "Pangu Ultra classifier repair exhausted; entering program-driven fallback "
            "with %s sections.",
            len(sections),
        )
        return self._run_program_driven_writing(
            sections=sections,
            overall_outline=outline,
            task_input=task_input,
            base_iteration=base_iteration,
            max_files_per_chapter=max_files,
            section_retry_limit=section_retries,
            retryable_error_types={
                "transport_error",
                "llm_empty_content",
                "llm_output_truncated",
            },
        )

    def _run_program_driven_writing(
        self,
        *,
        sections: List[Dict[str, Any]],
        overall_outline: str,
        task_input: WriterAgentTaskInput,
        base_iteration: int,
        max_files_per_chapter: int = None,
        section_retry_limit: int = 0,
        retryable_error_types: set = None,
    ) -> Dict[str, Any]:
        """Sequentially run section_writer + concat for DeepSeek without model-driven ordering."""
        # [FIX-P1] 写作开始前清理旧的 part_*.md 文件，防止多轮运行时旧文件残留
        try:
            report_dir = self._workspace_report_dir()
            if report_dir and report_dir.exists():
                import glob as _glob
                for old_file in _glob.glob(str(report_dir / "part_*.md")):
                    os.remove(old_file)
                    self.logger.info(f"[清理] 删除旧章节文件: {old_file}")
        except Exception as e:
            self.logger.warning(f"[清理] 清理旧章节文件失败: {e}")

        chapter_summaries: List[str] = []
        action_iteration = base_iteration
        user_query = getattr(task_input, "user_query", "")
        task_content = getattr(task_input, "task_content", "")
        key_files = getattr(task_input, "key_files", []) or []

        if not sections:
            return {"success": False, "error": "No sections parsed from classification result."}

        for index, section in enumerate(sections, start=1):
            current_outline = section.get("outline", "").strip()
            if not current_outline:
                continue

            chapter_key_files = self._select_key_files_by_paths(
                key_files,
                section.get("file_paths", [])
            )
            if max_files_per_chapter is not None:
                chapter_key_files = chapter_key_files[:max(1, int(max_files_per_chapter))]
            tool_call = {
                "name": "section_writer",
                "arguments": {
                    "written_chapters_summary": "\n\n".join(chapter_summaries).strip(),
                    "task_content": task_content,
                    "user_query": user_query,
                    "current_chapter_outline": current_outline,
                    "overall_outline": overall_outline,
                    "target_file_path": f"./report/part_{index}.md",
                    "key_files": chapter_key_files,
                }
            }

            try:
                chapter_title = current_outline.split("\n")[0].strip()
                chapter_title = chapter_title.replace("#", "").strip()[:50]
                writing_prefix = "正在撰写: " if getattr(self, "_is_chinese_query", True) else "Writing: "
                self._send_progress(
                    "writing_chapter",
                    f"{writing_prefix}{chapter_title}",
                    {"chapter_title": chapter_title}
                )
            except Exception as e:
                self.logger.debug(f"Failed to send program-driven progress: {e}")

            key_files_count = len(chapter_key_files) if isinstance(chapter_key_files, list) else 0
            self.logger.info(
                "[SectionWriter] key_files=%s target=%s",
                key_files_count,
                tool_call["arguments"].get("target_file_path", "unknown")
            )
            retries_used = 0
            while True:
                tool_result = self.execute_tool_call(tool_call)
                action_iteration += 1
                self.log_action(action_iteration, "section_writer", tool_call["arguments"], tool_result)
                if tool_result.get("success"):
                    break
                metadata = tool_result.get("metadata") or {}
                error_type = metadata.get("error_type")
                retryable = bool(metadata.get("retryable")) or (
                    retryable_error_types is not None and error_type in retryable_error_types
                )
                if not retryable or retries_used >= max(0, int(section_retry_limit or 0)):
                    break
                retries_used += 1
                self.logger.warning(
                    "Ultra fallback retrying section %s after retryable error (%s/%s): %s",
                    index,
                    retries_used,
                    section_retry_limit,
                    error_type or "unspecified",
                )
            if not tool_result.get("success"):
                return {"success": False, "error": tool_result.get("error", "section_writer failed")}

            summary = (tool_result.get("data") or {}).get("chapter_summary", "")
            if summary:
                chapter_summaries.append(summary)

            self._written_chapters.add(index)
            self._expected_next_chapter = index + 1
            self._crash_test_part_count += 1
            try:
                saved_msg = f"已保存: part_{index}.md" if getattr(self, "_is_chinese_query", True) else f"Saved: part_{index}.md"
                self._send_progress("chapter_saved", saved_msg, {"chapter_num": index})
            except Exception:
                pass

        concat_call = {
            "name": "concat_section_files",
            "arguments": {
                # [FIX-P1] 基于实际写入的章节编号构造文件列表，而非 range(1, len+1) 假设连续
                "section_files": [
                    {"file_path": f"./report/part_{ch}.md"}
                    for ch in sorted(self._written_chapters)
                ],
                "final_file_path": "./report/final_report.md"
            }
        }
        concat_result = self.execute_tool_call(concat_call)
        action_iteration += 1
        self.log_action(action_iteration, "concat_section_files", concat_call["arguments"], concat_result)

        if not concat_result.get("success"):
            return {"success": False, "error": concat_result.get("error", "concat_section_files failed")}

        self._has_merged_final_report = True

        is_cn = getattr(self, "_is_chinese_query", True)
        summary_header = "各章摘要汇总" if is_cn else "Chapter Summaries"
        article_summary = "\n\n".join(
            [f"### {summary_header} {i}\n{summary}" for i, summary in enumerate(chapter_summaries, start=1)]
        ).strip()

        completion_analysis = (
            "程序驱动顺序写作已完成全部章节并合并最终报告。"
            if is_cn else
            "Program-driven sequential writing completed all chapters and merged the final report."
        )

        completion_payload = {
            "final_article_path": "./report/final_report.md",
            "article_summary": article_summary,
            "completion_status": "completed",
            "completion_analysis": completion_analysis,
        }

        return {
            "success": True,
            "completion_payload": completion_payload,
            "last_iteration": action_iteration,
        }

    def _prepare_sources_before_writing(self, task_input: WriterAgentTaskInput):
        """Use the session's server, including remote deployments; fail open."""
        try:
            response = self.mcp_tools.client.call_tool('prepare_writer_sources', {
                'key_files': getattr(task_input, 'key_files', []) or [],
                'user_query': getattr(task_input, 'user_query', ''),
            })
            if not response.success:
                raise RuntimeError(response.error)
            payload = json.loads(response.data['content'][0]['text'])
            if not payload.get('success', False):
                raise RuntimeError(payload.get('error', 'source preflight failed'))
            data = payload.get('data') or {}
            if isinstance(data.get('key_files'), list):
                task_input.key_files = data['key_files']
            self.logger.info('[SourcePreflight] %s', data.get('metadata', {}))
        except Exception as exc:
            self.logger.warning('[SourcePreflight] preserving original handoff: %s', exc)

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
                is_phase2 = get_thread_human_in_loop_phase2()
                if is_phase2:
                    # 从 workspace 文件读取大纲
                    workspace_root = self._workspace_root()
                    if workspace_root:
                        outline_file = workspace_root / '.user_outline'
                        if outline_file.exists():
                            with open(outline_file, 'r', encoding='utf-8') as f:
                                user_outline = f.read().strip()
                    self.logger.info(f"[HITL] Phase 2 detected via env var, user_outline length={len(user_outline)}")

            if not is_phase2:
                self._prepare_sources_before_writing(task_input)
            self._reset_tool_argument_guard(task_input, overall_outline=user_outline)
            
            # Build system prompt for writing
            system_prompt = self._build_system_prompt(is_phase2=is_phase2, user_outline=user_outline)

            # Build initial user message from TaskInput
            user_message = self._build_initial_message_from_task_input(task_input)

            # Add to conversation
            conversation_history.append({"role": "system", "content": system_prompt})
            conversation_history.append({"role": "user", "content": user_message + " /no_think"})

            iteration = 0
            task_completed = False
            outline_confirmation_pending = False

            self.logger.debug("Checking conversation history before model call")
            self.logger.debug(f"Conversation history: {conversation_history}")
            # ReAct Loop for Writing: Research → Plan → Write → Refine → Complete
            # Get model configuration from config
            from config.config import get_config
            config = get_config()
            model_config = config.get_custom_llm_config()
            tool_call_format = get_tool_call_format_instruction()
            program_driven_enabled = self._should_use_program_driven_writing(model_config)
            program_driven_completed = False
            classification_processed = False
            validation_circuit_open = False
            classifier_circuit_open = False
            truncation_correction_count = 0
            terminal_error = None
            terminal_metadata: Dict[str, Any] = {}

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
                    validation_retry_requested = False
                    # Get LLM response via centralized client
                    llm_turn = llm_chat(
                        conversation_history,
                        model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        max_retries=10,
                        timeout=model_config.get("timeout", 180),
                        reject_truncated=True,
                        retry_truncated_same_request=False,
                        preserve_reasoning_on_argumentless_tool=True,
                        tool_schemas=self.tool_schemas,
                        tool_call_mode=model_config.get("tool_call_mode"),
                        return_tool_turn=True,
                    )
                    llm_turn = self._coerce_llm_tool_turn(llm_turn)
                    assistant_text = llm_turn.content

                    assistant_message = {"content": assistant_text}

                    try:
                        # if assistant_message["content"]:
                        #     reasoning_content = assistant_message["content"].split("[unused16]")[-1].split("[unused17]")[0]
                        reasoning_source = llm_turn.reasoning_content or assistant_message.get("content")
                        if reasoning_source:
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
                    #             tool_calls = json.loads(tool_call_str[0])
                    #         except:
                    #             return []
                    #     else:
                    #         return []
                    #     return tool_calls

                    # Add assistant message to conversation
                    self._append_assistant_tool_turn(
                        conversation_history,
                        llm_turn,
                        assistant_message["content"],
                    )

                    # tool_calls = extract_tool_calls(assistant_message["content"])
                    if llm_turn.tool_call_mode == "native":
                        tool_calls = llm_turn.tool_calls
                    else:
                        tool_calls = extract_tool_calls_from_response(
                            assistant_message["content"],
                            tool_schemas=self.tool_schemas,
                            api_profile=model_config.get("api_profile"),
                        )

                    # 增强日志：记录工具调用解析结果
                    if len(tool_calls) == 0:
                        self.logger.warning(f"[工具调用] 第{iteration}次迭代未解析到任何工具调用")
                        # 记录原始内容的前500字符用于调试
                        content_preview = assistant_message["content"][:500] if assistant_message.get("content") else "None"
                        self.logger.debug(f"[工具调用] 原始响应内容预览: {content_preview}")
                        
                        # 智能检测：如果Reasoning中提到section_writer但未成功调用，立即重试
                        content = assistant_message.get("content", "")
                        # has_tool_marker = "[unused11]" in content and "[unused12]" in content
                        has_tool_marker = (
                                ("[unused11]" in content and "[unused12]" in content)
                                or ("```json" in content and "```" in content)
                        )
                        mentions_section_writer = "section_writer" in content
                        
                        if mentions_section_writer and has_tool_marker:
                            # 确定：AI生成了工具调用标记但解析失败（JSON格式错误或不完整）
                            self.logger.error(f"[工具调用] 检测到工具调用标记但解析失败，JSON可能格式错误或不完整")
                            retry_prompt = (
                                "工具调用格式有误，未能成功解析。请重新生成section_writer工具调用，"
                                # "确保JSON格式正确且完整。格式示例：[unused11][{\"name\": \"section_writer\", \"arguments\": {...}}][unused12] /no_think"
                                f"确保JSON格式正确且完整。格式示例：{tool_call_format} /no_think"
                            )
                            conversation_history.append({"role": "user", "content": retry_prompt})
                            continue  # 立即进入下一次LLM调用，不增加迭代计数
                        elif mentions_section_writer and not has_tool_marker:
                            # 推断：AI在思考中提到section_writer但可能忘记生成工具调用
                            self.logger.warning(f"[工具调用] Reasoning中提到section_writer但未生成工具调用标记")
                            retry_prompt = (
                                "你在思考中提到要调用section_writer工具，但没有生成工具调用。"
                                # "请使用正确的格式生成工具调用：[unused11][{\"name\": \"section_writer\", \"arguments\": {...}}][unused12] /no_think"
                                f"请使用正确的格式生成工具调用：{tool_call_format} /no_think"
                            )
                            conversation_history.append({"role": "user", "content": retry_prompt})
                            continue
                    else:
                        self.logger.debug(f"[工具调用] 第{iteration}次迭代解析到{len(tool_calls)}个工具调用: {[tc.get('name') for tc in tool_calls]}")

                    # Execute tool calls if any (Acting phase)
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict) or "name" not in tool_call or "arguments" not in tool_call:
                            continue
                        tool_name = tool_call["name"]
                        # Str
                        arguments = tool_call["arguments"]
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except Exception:
                                arguments = {"raw_arguments": arguments}
                        tool_call["arguments"] = arguments
                        tool_name = tool_call["name"]
                        if tool_name == "search_result_classifier":
                            self._hydrate_search_result_classifier_tool_call(tool_call)
                            arguments = tool_call["arguments"]
                        elif tool_name == "section_writer":
                            self._hydrate_section_writer_tool_call(tool_call)
                            arguments = tool_call["arguments"]
                        self.logger.debug(f"Arguments is string: {isinstance(arguments, str)}")

                        # Check if planning is complete
                        if tool_name in ["writer_subjective_task_done"]:
                            validation_error = self._validate_tool_call_arguments(
                                tool_name,
                                arguments,
                            )
                            final_report_ready = self._is_final_report_ready()
                            hitl_partial = bool(
                                outline_confirmation_pending
                                and isinstance(arguments, dict)
                                and arguments.get("completion_status") == "partial"
                            )
                            if validation_error:
                                tool_result = validation_error
                            elif not final_report_ready and not hitl_partial:
                                tool_result = {
                                    "success": False,
                                    "error_code": "WRITER_FINAL_REPORT_NOT_READY",
                                    "error": (
                                        "最终报告尚未成功合并或文件不存在。请先完成章节并调用 "
                                        "concat_section_files 生成非空 final_report.md，再调用 "
                                        "writer_subjective_task_done。"
                                    ),
                                    "retryable": True,
                                }
                            else:
                                task_completed = True
                                self.log_action(iteration, tool_name, arguments, arguments)
                                break
                        elif program_driven_enabled and tool_name == "section_writer" and not classification_processed:
                            tool_result = {
                                "success": False,
                                "error": "程序驱动顺序写作模式下，请先完成 search_result_classifier 再开始章节撰写。"
                            }
                            self.log_action(iteration, tool_name, arguments, tool_result)
                            self._append_tool_result_turn(
                                conversation_history,
                                tool_call,
                                tool_result,
                                llm_turn.tool_call_mode,
                            )
                            continue
                        elif tool_name in ["think"]:
                            tool_result = {
                                "tool_results": "You can proceed to invoke other tools if needed. But the next step cannot call the reflect tool"}
                        elif tool_name == "search_result_classifier":
                            tool_result = self.execute_tool_call(tool_call)
                            classifier_error_text = " ".join([
                                str(tool_result.get("error_code", "")),
                                str(tool_result.get("error", "")),
                            ])
                            if "WAITING_FOR_OUTLINE_CONFIRMATION" in classifier_error_text:
                                outline_confirmation_pending = True
                            self._enforce_classification_contract(arguments, tool_result)
                            self._remember_classification_context(arguments, tool_result)
                            self.log_action(iteration, tool_name, arguments, tool_result)
                            self._append_tool_result_turn(
                                conversation_history,
                                tool_call,
                                tool_result,
                                llm_turn.tool_call_mode,
                            )
                            metadata = tool_result.get("metadata") or {}
                            if metadata.get("non_retryable_in_writer"):
                                if self._should_use_ultra_classifier_fallback(model_config):
                                    overall_outline = arguments.get("outline", "")
                                    driver_result = self._run_ultra_classifier_fallback(
                                        outline=overall_outline,
                                        task_input=task_input,
                                        model_config=model_config,
                                        base_iteration=iteration,
                                    )
                                    if driver_result.get("success"):
                                        completion_payload = driver_result.get("completion_payload", {})
                                        iteration = driver_result.get("last_iteration", iteration)
                                        self.log_action(
                                            iteration,
                                            "writer_subjective_task_done",
                                            completion_payload,
                                            completion_payload,
                                        )
                                        task_completed = True
                                        program_driven_completed = True
                                        break
                                    terminal_error = (
                                        "Pangu Ultra program-driven fallback failed: "
                                        f"{driver_result.get('error', 'unknown error')}"
                                    )
                                    terminal_metadata = {
                                        "error_type": "pangu_ultra_writer_fallback_exhausted",
                                        "retryable": False,
                                        "stage": "writer",
                                        "classifier_attempts": metadata.get("format_attempts", 2),
                                        "fallback_attempted": True,
                                    }
                                    classifier_circuit_open = True
                                    break
                                classifier_circuit_open = True
                                terminal_metadata = {
                                    "error_type": metadata.get(
                                        "error_type", "pangu_ultra_classifier_format_failure"
                                    ),
                                    "retryable": False,
                                    "stage": "writer",
                                    "classifier_attempts": metadata.get("format_attempts", 2),
                                    "fallback_attempted": False,
                                }
                                self.logger.error(
                                    "Writer stopped retrying deterministic classifier failure: %s",
                                    tool_result.get("error", "unknown classifier error"),
                                )
                                break
                            if self._record_tool_validation_failure(
                                tool_name, tool_result, generation_id=iteration
                            ):
                                validation_circuit_open = True
                                break
                            if tool_result.get("validation_retry_escalated"):
                                conversation_history.append({
                                    "role": "user",
                                    "content": f"[工具参数强制纠错] {json.dumps(tool_result, ensure_ascii=False)} /no_think"
                                })
                                validation_retry_requested = True
                                break
                            if program_driven_enabled and tool_result.get("success"):
                                classification_processed = True
                                classification_result = tool_result.get("data", "")
                                sections = self._parse_classification_result(classification_result)
                                outline_text = arguments.get("outline", "")
                                overall_outline = outline_text or "\n\n".join(
                                    [section.get("outline", "").strip() for section in sections if section.get("outline")]
                                )
                                if sections:
                                    self.logger.info(
                                        f"[ProgramDriven] Parsed {len(sections)} sections from classification result"
                                    )
                                    for idx, section in enumerate(sections, start=1):
                                        outline_preview = section.get("outline", "").split("\n", 1)[0].strip()
                                        file_paths = section.get("file_paths", [])
                                        preview_paths = ", ".join(file_paths[:5])
                                        self.logger.info(
                                            "[ProgramDriven] Section %s outline='%s' files=%s preview=%s",
                                            idx,
                                            outline_preview,
                                            len(file_paths),
                                            preview_paths
                                        )
                                    driver_result = self._run_program_driven_writing(
                                        sections=sections,
                                        overall_outline=overall_outline,
                                        task_input=task_input,
                                        base_iteration=iteration,
                                    )
                                    if driver_result.get("success"):
                                        completion_payload = driver_result.get("completion_payload", {})
                                        iteration = driver_result.get("last_iteration", iteration)
                                        self.log_action(iteration, "writer_subjective_task_done", completion_payload, completion_payload)
                                        task_completed = True
                                        program_driven_completed = True
                                        break
                                    else:
                                        self.logger.warning(
                                            f"Program-driven writing failed, fallback to model-driven flow: {driver_result.get('error')}"
                                        )
                                else:
                                    self.logger.warning("Program-driven writing skipped: empty classification result.")
                            continue
                        elif tool_name == "section_writer":
                            # [FIX-P1] 模型驱动流程中，首次调用 section_writer 时清理旧的 part_*.md
                            if not self._written_chapters:
                                try:
                                    report_dir = self._workspace_report_dir()
                                    if report_dir and report_dir.exists():
                                        import glob as _glob
                                        for old_file in _glob.glob(str(report_dir / "part_*.md")):
                                            os.remove(old_file)
                                            self.logger.info(f"[清理] 删除旧章节文件: {old_file}")
                                except Exception as e:
                                    self.logger.warning(f"[清理] 清理旧章节文件失败: {e}")

                            # 章节编号验证
                            write_file_path = ""
                            parsed_args = {}
                            if isinstance(arguments, dict):
                                parsed_args = arguments
                                write_file_path = arguments.get('target_file_path', '') or arguments.get('write_file_path', '')
                            elif isinstance(arguments, str):
                                parsed_args = json.loads(arguments)
                                write_file_path = parsed_args.get('target_file_path', '') or parsed_args.get('write_file_path', '')
                            
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
                                        outline = parsed_args.get('current_chapter_outline', '')

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
                                    key_files = parsed_args.get('key_files', []) if isinstance(parsed_args, dict) else []
                                    key_files_count = len(key_files) if isinstance(key_files, list) else 0
                                    self.logger.info(
                                        "[SectionWriter] key_files=%s target=%s",
                                        key_files_count,
                                        write_file_path or "unknown"
                                    )
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
                                    outline = parsed_args.get('current_chapter_outline', '')

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
                                key_files = parsed_args.get('key_files', []) if isinstance(parsed_args, dict) else []
                                key_files_count = len(key_files) if isinstance(key_files, list) else 0
                                self.logger.info(
                                    "[SectionWriter] key_files=%s target=%s",
                                    key_files_count,
                                    write_file_path or "unknown"
                                )
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

                        if tool_name == "section_writer" and tool_result.get("success"):
                            self._remember_section_writer_summary(tool_result)
                        if self._record_tool_validation_failure(
                            tool_name, tool_result, generation_id=iteration
                        ):
                            validation_circuit_open = True

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
                        # search_result_classifier already publishes its result
                        # above because several classifier-specific branches may
                        # exit early.  Preserve the historical duplicate text
                        # feedback, but native role=tool must be exactly one
                        # message per tool_call_id.
                        if not (
                            llm_turn.tool_call_mode == "native"
                            and tool_name == "search_result_classifier"
                        ):
                            self._append_tool_result_turn(
                                conversation_history,
                                tool_call,
                                tool_result,
                                llm_turn.tool_call_mode,
                            )

                        # [FIX-P2] section_writer 成功后注入明确的完成标记，减少 LLM 重复调用同一章节
                        if tool_name == "section_writer" and tool_result.get("success"):
                            chapter_match_ctx = re.search(r'part_(\d+)\.md', str(arguments))
                            if chapter_match_ctx:
                                done_ch = int(chapter_match_ctx.group(1))
                                completion_marker = (
                                    f"✅ 第{done_ch}章已成功写入并验证，文件为 part_{done_ch}.md。"
                                    f"下一章必须写第{done_ch + 1}章，不要重复调用第{done_ch}章的 section_writer。 /no_think"
                                )
                                conversation_history.append({"role": "user", "content": completion_marker})

                        if validation_circuit_open:
                            break
                        if tool_result.get("validation_retry_escalated"):
                            validation_retry_requested = True
                            break

                    if classifier_circuit_open:
                        terminal_error = terminal_error or (
                            "Writer stopped after pangu_ultra_moe classifier format repair was exhausted."
                        )
                        self.logger.error(terminal_error)
                        break
                    if validation_circuit_open:
                        terminal_error = (
                            "Writer stopped by repeated tool argument validation failures."
                        )
                        self.logger.error(terminal_error)
                        break
                    if validation_retry_requested:
                        self.logger.warning(
                            "Writer requested a fresh model generation for tool argument correction."
                        )
                        continue

                    # If no tool calls, encourage continued writing
                    if len(tool_calls) == 0:
                        # Add follow-up prompt to encourage action or completion
                        followup_prompt = (
                            "Continue your writing process. If you need to research more, use available tools. "
                            "If you need to write or edit content, use file operations. "
                            "If your writing is complete and meets requirements, call writer_subjective_task_done. /no_think"
                        )
                        conversation_history.append({"role": "user", "content": followup_prompt})

                    if program_driven_completed:
                        break

                except LLMOutputTruncatedError as e:
                    if (
                        truncation_correction_count < self.MAX_TRUNCATION_CORRECTIONS
                        and iteration < self.config.max_iterations
                    ):
                        truncation_correction_count += 1
                        self.logger.warning(
                            "Writer discarded truncated generation and requested correction "
                            "(%s/%s): %s",
                            truncation_correction_count,
                            self.MAX_TRUNCATION_CORRECTIONS,
                            e,
                        )
                        conversation_history.append({
                            "role": "user",
                            "content": self.TRUNCATION_CORRECTION_PROMPT,
                        })
                        continue

                    if truncation_correction_count >= self.MAX_TRUNCATION_CORRECTIONS:
                        terminal_error = (
                            "Writer output remained truncated after "
                            f"{self.MAX_TRUNCATION_CORRECTIONS} corrective generations"
                        )
                    else:
                        terminal_error = (
                            "Writer output was truncated and no iteration remained "
                            "for a corrective generation"
                        )
                    self.log_error(iteration, f"{terminal_error}: {e}")
                    break
                except Exception as e:
                    error_msg = f"Error in writing iteration {iteration}: {e}"
                    terminal_error = error_msg
                    self.log_error(iteration, error_msg)
                    break

            # 【降级兜底A】writer agent 异常退出或超时时，尝试自动合并已有的 part_*.md
            # 采用分级降级策略：根据章节数决定是否合并以及如何标注
            if not task_completed:
                try:
                    report_dir = self._workspace_report_dir()
                    if report_dir:
                        import re as _re
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
                                    final_content, marker_issues = normalize_and_validate_report(final_content)
                                    if marker_issues:
                                        raise ValueError(f"降级报告仍包含内部标记: {marker_issues[:10]}")
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
                                    final_content, marker_issues = normalize_and_validate_report(final_content)
                                    if marker_issues:
                                        raise ValueError(f"降级报告仍包含内部标记: {marker_issues[:10]}")
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
                response_kwargs = {
                    "success": False,
                    "error": terminal_error or f"Writing task not completed within {self.config.max_iterations} iterations",
                    "iterations": iteration,
                    "execution_time": execution_time,
                }
                # Preserve compatibility with existing create_response overrides for all
                # legacy paths; structured metadata is only needed for Ultra terminal errors.
                if terminal_metadata:
                    response_kwargs["metadata"] = terminal_metadata
                return self.create_response(**response_kwargs)

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
