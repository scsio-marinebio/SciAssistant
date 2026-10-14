# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
import os
import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from pathlib import Path
from dotenv import load_dotenv
from .thinking import normalize_thinking_settings, parse_legacy_think_mode


# Load .env file from config directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "y"}


def normalize_tool_call_mode(value: Optional[str]) -> str:
    """Validate the deployment-selected tool-call transport mode."""
    mode = (value or "text").strip().lower()
    if mode not in {"text", "native"}:
        raise ValueError(
            "MODEL_TOOL_CALL_MODE must be either 'text' or 'native', "
            f"got {value!r}"
        )
    return mode


@dataclass
class APIConfig:
    """Configuration class for API keys and settings"""
    
    # Custom LLM Service Configuration
    # Your own deployed LLM service accessed via requests
    model_request_url: Optional[str] = None
    model_request_token: Optional[str] = None
    # model_name: str = "pangu_auto"  # Default model name
    model_name: str = "deepseek-chat"  # Default model name
    model_provider: str = "auto"  # auto | pangu | deepseek | qwen | openai_compatible
    # Endpoint/template profile for observability and future endpoint-specific
    # adapters. Generic malformed-tag recovery is schema/signature driven.
    model_api_profile: str = "default"
    # Tool-call transport is a deployment capability, not a model-name rule.
    # Keep text as the compatibility default; enable native only after the
    # configured endpoint/model combination has passed a direct probe.
    model_tool_call_mode: str = "text"  # text | native
    model_chat_template: Optional[str] = None
    model_spaces_between_special_tokens: Optional[bool] = None
    model_think_mode: Optional[bool] = None  # Legacy boolean; implicit transport is retained only for Pangu.
    model_thinking_format: Optional[str] = None
    model_thinking_mode: Optional[str] = None
    model_reasoning_effort: Optional[str] = None
    # Exact-model compatibility switch. It never applies to legacy pangu models.
    pangu_ultra_moe_compat_enabled: bool = False
    pangu_ultra_classifier_max_format_retries: int = 1
    pangu_ultra_classifier_temperature: float = 0.0
    # pangu_ultra_moe 分类失败后的可选程序驱动兜底；策略 A 阶段保持关闭。
    pangu_ultra_writer_fallback_enabled: bool = False
    pangu_ultra_writer_max_invocations: int = 2
    pangu_ultra_fallback_max_files_per_chapter: int = 11
    pangu_ultra_fallback_section_retries: int = 1

    # Custom Planner Mode
    planner_mode: str = "auto"  # Default planner mode
    
    # MCP Server Configuration
    mcp_server_url: Optional[str] = None
    mcp_auth_token: Optional[str] = None
    mcp_use_stdio: bool = True  # Default to stdio for backward compatibility
    
    # Search Engine Configuration (Generic)
    search_engine_base_url: Optional[str] = None
    search_engine_api_keys: Optional[str] = None  # Can be comma-separated for rotation
    
    # Webpage Fetcher Configuration (Generic)
    url_crawler_base_url: Optional[str] = None
    url_crawler_api_keys: Optional[str] = None  # Can be comma-separated for rotation
    url_crawler_max_tokens: int = 100000
    
    # RAG Knowledge Base Configuration
    rag_api_url: Optional[str] = None
    rag_app_code: Optional[str] = None
    rag_default_repo_id: Optional[str] = None
    rag_default_page_size: int = 20  # 增加到 20，提高相关文档覆盖率
    search_source_rag: bool = True  # 是否全局启用 RAG 搜索源

    # Proxy Configuration
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: Optional[str] = None
    
    # Model Interaction Configuration
    model_temperature: float = 0.3
    model_max_tokens: int = 8192
    model_request_timeout: int = 900
    
    # Tool Trajectory and Output Configuration  
    trajectory_storage_path: str = "./workspace"
    report_output_path: str = "./report"
    document_analysis_path: str = "./doc_analysis"
    
    # Per-agent iteration controls (optional; resolved by agent factories)
    planner_max_iterations: Optional[int] = None
    information_seeker_max_iterations: Optional[int] = None
    writer_max_iterations: Optional[int] = None
    
    # General Settings
    debug_mode: bool = False
    max_retries: int = 3
    timeout: int = 30
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        self.load_from_env()
    
    def load_from_env(self):
        """Load API keys and settings from environment variables"""
        # Custom LLM Service
        self.model_request_url = os.getenv('MODEL_REQUEST_URL')
        self.model_request_token = os.getenv('MODEL_REQUEST_TOKEN')
        # self.model_name = os.getenv('MODEL_NAME', 'pangu-auto')
        self.model_name = os.getenv('MODEL_NAME')
        self.model_provider = os.getenv('MODEL_PROVIDER', self.model_provider)
        self.model_api_profile = os.getenv(
            'MODEL_API_PROFILE', self.model_api_profile
        ).strip().lower()
        self.model_tool_call_mode = normalize_tool_call_mode(
            os.getenv('MODEL_TOOL_CALL_MODE', self.model_tool_call_mode)
        )
        self.model_chat_template = os.getenv('MODEL_CHAT_TEMPLATE', self.model_chat_template)
        spaces_between_tokens = os.getenv('MODEL_SPACES_BETWEEN_SPECIAL_TOKENS')
        parsed_spaces = _parse_optional_bool(spaces_between_tokens)
        if parsed_spaces is not None:
            self.model_spaces_between_special_tokens = parsed_spaces
        think_mode = os.getenv('MODEL_THINK_MODE')
        parsed_think = parse_legacy_think_mode(think_mode)
        if parsed_think is not None:
            self.model_think_mode = parsed_think
        self.model_thinking_format, self.model_thinking_mode, self.model_reasoning_effort = normalize_thinking_settings(
            os.getenv('MODEL_THINKING_FORMAT', self.model_thinking_format),
            os.getenv('MODEL_THINKING_MODE', self.model_thinking_mode),
            os.getenv('MODEL_REASONING_EFFORT', self.model_reasoning_effort),
            self.model_think_mode,
        )
        self.pangu_ultra_moe_compat_enabled = (
            os.getenv(
                "PANGU_ULTRA_MOE_COMPAT_ENABLED",
                str(self.pangu_ultra_moe_compat_enabled),
            ).strip().lower() == "true"
        )
        self.pangu_ultra_classifier_max_format_retries = max(
            0,
            int(os.getenv(
                "PANGU_ULTRA_CLASSIFIER_MAX_FORMAT_RETRIES",
                self.pangu_ultra_classifier_max_format_retries,
            )),
        )
        self.pangu_ultra_classifier_temperature = float(os.getenv(
            "PANGU_ULTRA_CLASSIFIER_TEMPERATURE",
            self.pangu_ultra_classifier_temperature,
        ))
        self.pangu_ultra_writer_fallback_enabled = (
            os.getenv(
                "PANGU_ULTRA_WRITER_FALLBACK_ENABLED",
                str(self.pangu_ultra_writer_fallback_enabled),
            ).strip().lower() == "true"
        )
        self.pangu_ultra_writer_max_invocations = max(
            1,
            int(os.getenv(
                "PANGU_ULTRA_WRITER_MAX_INVOCATIONS",
                self.pangu_ultra_writer_max_invocations,
            )),
        )
        self.pangu_ultra_fallback_max_files_per_chapter = max(
            1,
            int(os.getenv(
                "PANGU_ULTRA_FALLBACK_MAX_FILES_PER_CHAPTER",
                self.pangu_ultra_fallback_max_files_per_chapter,
            )),
        )
        self.pangu_ultra_fallback_section_retries = max(
            0,
            int(os.getenv(
                "PANGU_ULTRA_FALLBACK_SECTION_RETRIES",
                self.pangu_ultra_fallback_section_retries,
            )),
        )
        
        # Custom Planner Mode
        self.planner_mode = os.getenv("PLANNER_MODE", self.planner_mode)
        
        # MCP Server
        self.mcp_server_url = os.getenv("MCP_SERVER_URL")
        self.mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
        self.mcp_use_stdio = os.getenv("MCP_USE_STDIO", "true").lower() == "true"
        
        # Search Engine Configuration
        self.search_engine_base_url = os.getenv("SEARCH_ENGINE_BASE_URL")
        self.search_engine_api_keys = os.getenv("SEARCH_ENGINE_API_KEYS")
        
        # Webpage Fetcher Configuration
        self.url_crawler_base_url = os.getenv("URL_CRAWLER_BASE_URL")
        self.url_crawler_api_keys = os.getenv("URL_CRAWLER_API_KEYS")
        self.url_crawler_max_tokens = int(os.getenv("URL_CRAWLER_MAX_TOKENS", self.url_crawler_max_tokens))
        
        # RAG Knowledge Base Configuration
        self.rag_api_url = os.getenv("RAG_API_URL")
        self.rag_app_code = os.getenv("RAG_APP_CODE")
        self.rag_default_repo_id = os.getenv("RAG_DEFAULT_REPO_ID")
        self.rag_default_page_size = int(os.getenv("RAG_DEFAULT_PAGE_SIZE", self.rag_default_page_size))
        self.search_source_rag = os.getenv("SEARCH_SOURCE_RAG", "true").lower() == "true"

        # Proxy Configuration
        self.http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        self.https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        self.no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
        
        # Model Interaction Configuration
        self.model_temperature = float(os.getenv("MODEL_TEMPERATURE", self.model_temperature))
        self.model_max_tokens = int(os.getenv("MODEL_MAX_TOKENS", self.model_max_tokens))
        self.model_request_timeout = int(os.getenv("MODEL_REQUEST_TIMEOUT", self.model_request_timeout))
        
        # Tool Trajectory and Output Configuration
        self.trajectory_storage_path = os.getenv("TRAJECTORY_STORAGE_PATH", self.trajectory_storage_path)
        self.report_output_path = os.getenv("REPORT_OUTPUT_PATH", self.report_output_path)
        self.document_analysis_path = os.getenv("DOCUMENT_ANALYSIS_PATH", self.document_analysis_path)
        
        # Per-agent iteration controls
        self.planner_max_iterations = (
            int(os.getenv("PLANNER_MAX_ITERATION")) if os.getenv("PLANNER_MAX_ITERATION") else None
        )
        self.information_seeker_max_iterations = (
            int(os.getenv("INFORMATION_SEEKER_MAX_ITERATION")) if os.getenv("INFORMATION_SEEKER_MAX_ITERATION") else None
        )
        self.writer_max_iterations = (
            int(os.getenv("WRITER_MAX_ITERATION")) if os.getenv("WRITER_MAX_ITERATION") else None
        )
        
        # General Settings
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.max_retries = int(os.getenv("MAX_RETRIES", self.max_retries))
        self.timeout = int(os.getenv("TIMEOUT", self.timeout))
    
    def get_custom_llm_config(self) -> Dict[str, Any]:
        """Get configuration for custom LLM service"""
        return {
            "url": self.model_request_url,
            "token": self.model_request_token,
            "model": self.model_name,
            "provider": self.model_provider,
            "api_profile": self.model_api_profile,
            "tool_call_mode": self.model_tool_call_mode,
            "chat_template": self.model_chat_template,
            "spaces_between_special_tokens": self.model_spaces_between_special_tokens,
            "think_mode": self.model_think_mode,
            "thinking_format": self.model_thinking_format,
            "thinking_mode": self.model_thinking_mode,
            "reasoning_effort": self.model_reasoning_effort,
            "pangu_ultra_moe_compat_enabled": self.pangu_ultra_moe_compat_enabled,
            "pangu_ultra_classifier_max_format_retries": self.pangu_ultra_classifier_max_format_retries,
            "pangu_ultra_classifier_temperature": self.pangu_ultra_classifier_temperature,
            "pangu_ultra_writer_fallback_enabled": self.pangu_ultra_writer_fallback_enabled,
            "pangu_ultra_writer_max_invocations": self.pangu_ultra_writer_max_invocations,
            "pangu_ultra_fallback_max_files_per_chapter": self.pangu_ultra_fallback_max_files_per_chapter,
            "pangu_ultra_fallback_section_retries": self.pangu_ultra_fallback_section_retries,
            "temperature": self.model_temperature,
            "max_tokens": self.model_max_tokens,
            "timeout": self.model_request_timeout,
            "base_url": self.model_request_url  # For backward compatibility with model_config.get('base_url')
        }
    
    def get_available_search_providers(self) -> list:
        """Get list of available search providers based on API keys"""
        providers = []
        if self.search_engine_api_keys:
            providers.append("custom")
        return providers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if "api_key" in key.lower() or "password" in key.lower():
                config_dict[key] = "***" if value else None
            else:
                config_dict[key] = value
        return config_dict

# Global configuration instance
config = APIConfig()


def get_config() -> APIConfig:
    """Get the global configuration instance"""
    return config


def is_pangu_ultra_moe_compat_enabled(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> bool:
    """Return True only for the explicitly enabled pangu_ultra_moe adapter."""
    resolved_name = (model_name or config.model_name or "").strip().lower()
    resolved_provider = (provider or config.model_provider or "").strip().lower()
    if not resolved_provider or resolved_provider == "auto":
        resolved_provider = get_model_provider(resolved_name)
    resolved_enabled = (
        config.pangu_ultra_moe_compat_enabled if enabled is None else bool(enabled)
    )
    return (
        resolved_enabled
        and resolved_provider == "pangu"
        and resolved_name == "pangu_ultra_moe"
    )


def reload_config():
    """Reload configuration from environment variables"""
    global config
    config = APIConfig()
    logger.info("Configuration reloaded")


def validate_api_key(api_key: Optional[str], service_name: str) -> bool:
    """Validate that an API key is present and not empty"""
    if not api_key or api_key.strip() == "":
        logger.error(f"Missing or empty API key for {service_name}")
        return False
    return True


def get_url_crawler_config() -> Dict[str, Any]:
    """Get generic Webpage fetcher configuration"""
    api_keys = config.url_crawler_api_keys
    base_url = config.url_crawler_base_url
    
    if not api_keys:
        return {}
    
    # Parse comma-separated API keys for rotation
    api_key_list = [key.strip() for key in api_keys.split(",")] if isinstance(api_keys, str) else [api_keys]
    
    return {
        "api_keys": api_key_list,
        "base_url": base_url,
        "max_tokens": config.url_crawler_max_tokens,
        "timeout": config.timeout
    }


def get_search_engine_config() -> Dict[str, Any]:
    """Get generic search engine configuration"""
    api_keys = config.search_engine_api_keys
    base_url = config.search_engine_base_url
    
    if not api_keys:
        return {}
    
    # Parse comma-separated API keys for rotation
    api_key_list = [key.strip() for key in api_keys.split(",")] if isinstance(api_keys, str) else [api_keys]
    
    return {
        "api_keys": api_key_list,
        "base_url": base_url,
        "timeout": config.timeout
    }


def get_model_config() -> Dict[str, Any]:
    """Get model interaction configuration for custom LLM service"""
    return config.get_custom_llm_config()

def get_model_provider(model_name: Optional[str] = None) -> str:
    """Resolve model provider from env override or model name."""
    provider = (config.model_provider or "").strip().lower()
    if provider and provider != "auto":
        return provider

    name = (model_name or config.model_name or "").strip().lower()
    if "pangu" in name:
        return "pangu"
    if "deepseek" in name:
        return "deepseek"
    if "qwen" in name:
        return "qwen"
    if "glm" in name:
        return "glm"
    return "openai_compatible"


def get_tool_call_mode() -> str:
    """Return the configured tool-call transport without model-name inference."""
    return normalize_tool_call_mode(config.model_tool_call_mode)


def get_tool_schemas_prompt(tool_schemas: List[Dict[str, Any]]) -> str:
    """Avoid duplicating schemas in prompts when the API carries native tools."""
    if get_tool_call_mode() == "native":
        return "Tool definitions are supplied through the native API tools field."
    return json.dumps(tool_schemas, ensure_ascii=False)


def get_model_type(model_name: Optional[str] = None) -> str:
    """Get model type for request/response handling."""
    provider = get_model_provider(model_name)
    if provider in {"deepseek", "pangu", "qwen"}:
        return provider
    return "openai_compatible"


def build_llm_request_body(
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        **kwargs
) -> Dict[str, Any]:
    """Build a standard chat-completions request body for DeepSeek/OpenAI-compatible APIs."""
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    for key, value in kwargs.items():
        if value is not None:
            body[key] = value
    return body


def build_model_request_headers(model_token: Optional[str] = None) -> Dict[str, str]:
    """
    Build request headers compatible with multiple OpenAI-compatible gateways.
    Includes Authorization/api-key/csb-token in parallel to maximize compatibility.
    """
    token = (model_token or config.model_request_token or "").strip()
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }
    if not token:
        return headers

    if token.lower().startswith("bearer "):
        raw_token = token.split(" ", 1)[1].strip()
        bearer_value = token
    else:
        raw_token = token
        bearer_value = f"Bearer {token}"

    headers["Authorization"] = bearer_value
    headers["api-key"] = raw_token
    headers["x-api-key"] = raw_token
    headers["csb-token"] = raw_token
    return headers


def extract_reasoning_from_response(content: str) -> str:
    """Extract reasoning content from DeepSeek style output."""
    if not content:
        return ""

    think_match = re.search(r"<think>([\s\S]*?)</think>", content)
    if think_match:
        return think_match.group(1).strip()

    if "<think>" in content:
        return content.split("<think>", 1)[-1].strip()[:800]

    return content.strip()[:800]


def get_tool_call_format_instruction() -> str:
    """Return tool call format instruction based on active model provider."""
    cfg = get_config()
    model_cfg = cfg.get_custom_llm_config()
    model_name = model_cfg.get("model") or cfg.model_name
    provider = (model_cfg.get("provider") or "").strip().lower()
    if not provider or provider == "auto":
        provider = get_model_provider(model_name)
    if normalize_tool_call_mode(model_cfg.get("tool_call_mode")) == "native":
        return (
            "Use only the native function-calling tools supplied by the API. "
            "Do not serialize tool calls into message content."
        )
    if provider == "pangu":
        # When chat_template is configured (legacy self-hosted pangu), the model
        # uses [unused11]/[unused12] special tokens. When chat_template is absent
        # (managed inference service / intranet deployment), the service applies
        # its own template and the model emits vLLM-native <|tool_call_start|> /
        # <|tool_call_end|> markers instead.
        chat_template = model_cfg.get("chat_template")
        if chat_template:
            return "[unused11][{\"name\": \"<function name>\", \"arguments\": {}}][unused12]"
        return (
            "<|tool_call_start|>[{\"name\": \"<function name>\", \"arguments\": {}}]<|tool_call_end|>\n"
            "IMPORTANT: Output tool calls wrapped within <|tool_call_start|> and <|tool_call_end|> markers."
        )
    if provider == "deepseek":
        return (
            "```json\n{\"name\": \"<function name>\", \"arguments\": {}}\n```\n"
            "IMPORTANT: Output ONLY the JSON code block for tool calls, no extra text."
        )
    if provider == "glm":
        return (
            "```json\n{\"name\": \"<function name>\", \"arguments\": {}}\n```\n"
            "IMPORTANT: Output ONLY the JSON code block for tool calls, no extra text."
        )
    return "```json\n{\"name\": \"<function name>\", \"arguments\": {}}\n```"


def _normalize_tool_calls(parsed: Any) -> List[Dict[str, Any]]:
    """Normalize parsed payload into a list of tool calls."""
    if isinstance(parsed, dict):
        if "name" in parsed:
            if "arguments" in parsed:
                return [{"name": parsed["name"], "arguments": parsed.get("arguments")}]
            arguments = {key: value for key, value in parsed.items() if key != "name"}
            return [{"name": parsed["name"], "arguments": arguments}]
        return []
    if isinstance(parsed, list):
        normalized: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            if "name" in item:
                if "arguments" in item:
                    normalized.append({"name": item["name"], "arguments": item.get("arguments")})
                else:
                    arguments = {key: value for key, value in item.items() if key != "name"}
                    normalized.append({"name": item["name"], "arguments": arguments})
                continue
            function_block = item.get("function") if isinstance(item.get("function"), dict) else None
            if function_block and function_block.get("name"):
                normalized.append({"name": function_block["name"], "arguments": function_block.get("arguments")})
        return normalized
    return []


def extract_tool_calls_from_response(
    content: str,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    api_profile: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract tool calls, using schemas to constrain ambiguous GLM recovery.

    A high-confidence, provider-independent post-parse pass recovers the
    malformed ``</arg_value>key</arg_key>value`` variant first observed from
    Ark Agent Plan. Existing parsers always run first, valid results are never
    rewritten, and recovered arguments must match the active tool schema.

    ``api_profile`` remains available as endpoint context for logging and for
    future truly platform-specific adapters; it does not gate this generic,
    signature-based recovery.
    """
    if not content:
        return []
    resolved_api_profile = (api_profile or "default").strip().lower()

    def _try_parse_payload(payload: str) -> Optional[Any]:
        cleaned = payload.strip()
        if not cleaned:
            return None
        for text in dict.fromkeys([cleaned]):
            try:
                return json.loads(text)
            except Exception:
                try:
                    import yaml
                    return yaml.safe_load(text)
                except Exception:
                    try:
                        import ast
                        return ast.literal_eval(text)
                    except Exception:
                        continue
        return None

    def _coerce_scalar(value: str) -> Any:
        cleaned = value.strip()
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if re.fullmatch(r"-?\d+", cleaned):
            try:
                return int(cleaned)
            except Exception:
                return cleaned
        if re.fullmatch(r"-?\d+\.\d+", cleaned):
            try:
                return float(cleaned)
            except Exception:
                return cleaned
        if cleaned[0] in "{[":
            parsed = _parse_payload(cleaned)
            if parsed is not None:
                return parsed
        if cleaned[0] in "\"'":
            try:
                import ast
                return ast.literal_eval(cleaned)
            except Exception:
                return cleaned.strip("\"'")
        return cleaned

    def _parse_kv_arguments(arg_text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not arg_text:
            return result
        text = arg_text.strip()
        idx = 0
        length = len(text)
        while idx < length:
            while idx < length and text[idx] in " ,\n\t":
                idx += 1
            if idx >= length:
                break
            key_start = idx
            while idx < length and re.match(r"[A-Za-z0-9_\-]", text[idx]):
                idx += 1
            key = text[key_start:idx].strip()
            if not key:
                break
            while idx < length and text[idx].isspace():
                idx += 1
            if idx < length and text[idx] in "=:":
                idx += 1
            while idx < length and text[idx].isspace():
                idx += 1
            if idx >= length:
                result[key] = ""
                break
            if text[idx] in "\"'":
                quote = text[idx]
                idx += 1
                value_start = idx
                escaped = False
                while idx < length:
                    ch = text[idx]
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == quote:
                        break
                    idx += 1
                value = text[value_start:idx]
                idx = idx + 1 if idx < length else idx
                result[key] = _coerce_scalar(f"{quote}{value}{quote}")
            else:
                next_match = re.search(r"[,\t\n]\s*[A-Za-z0-9_\-]+\s*[=:]", text[idx:])
                if next_match:
                    value = text[idx: idx + next_match.start()].strip()
                    idx = idx + next_match.start() + 1
                else:
                    value = text[idx:].strip()
                    idx = length
                result[key] = _coerce_scalar(value)
        return result

    def _parse_tagged_arguments(arg_text: str) -> Dict[str, Any]:
        """Parse GLM native ``<arg_key>/<arg_value>`` tool arguments.

        GLM-5.2 may emit a complete textual tool call without JSON or
        parentheses. Previously this format was reduced to an empty argument
        object even though every value was present in the model response.
        """
        if not arg_text or "<arg_key>" not in arg_text:
            return {}

        result: Dict[str, Any] = {}
        pairs = re.findall(
            r"<arg_key>\s*([\s\S]*?)\s*</arg_key>\s*"
            r"<arg_value>\s*([\s\S]*?)\s*</arg_value>",
            arg_text,
            re.IGNORECASE,
        )
        for raw_key, raw_value in pairs:
            key = raw_key.strip()
            if not key or not re.fullmatch(r"[A-Za-z0-9_\-]+", key):
                continue
            result[key] = _coerce_scalar(raw_value)
        return result

    def _schema_argument_names(tool_name: str) -> List[str]:
        for schema in tool_schemas or []:
            if not isinstance(schema, dict):
                continue
            function_schema = schema.get("function")
            if not isinstance(function_schema, dict) or function_schema.get("name") != tool_name:
                continue
            parameters = function_schema.get("parameters")
            properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            if isinstance(properties, dict):
                return [name for name in properties if isinstance(name, str) and name]
        return []

    def _schema_parameters(tool_name: str) -> Dict[str, Any]:
        for schema in tool_schemas or []:
            if not isinstance(schema, dict):
                continue
            function_schema = schema.get("function")
            if not isinstance(function_schema, dict) or function_schema.get("name") != tool_name:
                continue
            parameters = function_schema.get("parameters")
            return parameters if isinstance(parameters, dict) else {}
        return {}

    def _arguments_match_schema(tool_name: str, arguments: Any) -> bool:
        """Apply strict checks before accepting high-confidence recovery."""
        if not isinstance(arguments, dict):
            return False
        parameters = _schema_parameters(tool_name)
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        if not isinstance(properties, dict) or not properties:
            return False
        required = parameters.get("required", [])
        if not isinstance(required, list):
            required = []
        if any(name not in arguments for name in required if isinstance(name, str)):
            return False
        if any(name not in properties for name in arguments):
            return False

        expected_types = {
            "string": str,
            "array": list,
            "object": dict,
            "boolean": bool,
            "integer": int,
            "number": (int, float),
        }
        for name, value in arguments.items():
            property_schema = properties.get(name)
            if not isinstance(property_schema, dict):
                continue
            expected_type = property_schema.get("type")
            python_type = expected_types.get(expected_type)
            if python_type is not None:
                if expected_type in {"integer", "number"} and isinstance(value, bool):
                    return False
                if not isinstance(value, python_type):
                    return False
            allowed_values = property_schema.get("enum")
            if isinstance(allowed_values, list) and value not in allowed_values:
                return False
        return bool(arguments)

    def _tool_calls_need_schema_recovery(calls: List[Dict[str, Any]]) -> bool:
        if not calls:
            return True
        for call in calls:
            if not isinstance(call, dict):
                return True
            tool_name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                return True
            parameters = _schema_parameters(tool_name)
            required = parameters.get("required", []) if isinstance(parameters, dict) else []
            if not isinstance(required, list):
                required = []
            if not arguments or any(
                isinstance(name, str) and name not in arguments for name in required
            ):
                return True
        return False

    def _parse_displaced_arg_tag_block(block: str) -> List[Dict[str, Any]]:
        """Recover only schema-declared arguments from displaced arg tags."""
        if not tool_schemas:
            return []
        cleaned = block.strip()
        name_match = re.match(r"([A-Za-z0-9_\-]+)", cleaned)
        if not name_match:
            return []
        tool_name = name_match.group(1)
        allowed_names = _schema_argument_names(tool_name)
        if not allowed_names:
            return []

        remainder = cleaned[name_match.end():]
        # Deliberately narrow signature: a displaced arg tag must immediately
        # follow the tool name. Provider/model names never trigger recovery.
        if not re.match(r"\s*</?arg_(?:key|value)>", remainder, re.IGNORECASE):
            return []

        normalized = re.sub(
            r"^\s*(?:</?arg_(?:key|value)>\s*)+",
            "",
            remainder,
            flags=re.IGNORECASE,
        )
        for argument_name in sorted(allowed_names, key=len, reverse=True):
            marker = re.compile(
                rf"(^|[,\n\t])\s*\"?{re.escape(argument_name)}\"?\s*"
                rf"(?:</arg_key>)?\s*(?::|=)?\s*",
                re.IGNORECASE,
            )
            normalized = marker.sub(
                lambda match: f"{match.group(1)}{argument_name}: ",
                normalized,
            )
        normalized = re.sub(
            r"</?arg_(?:key|value)>", "", normalized, flags=re.IGNORECASE
        ).strip()

        parsed_arguments = _parse_kv_arguments(normalized)
        allowed_name_set = set(allowed_names)
        if any(name not in allowed_name_set for name in parsed_arguments):
            return []
        filtered_arguments = {
            name: value
            for name, value in parsed_arguments.items()
            if name in allowed_name_set
        }
        if not _arguments_match_schema(tool_name, filtered_arguments):
            return []
        logger.info(
            "Recovered displaced tool-argument tags: profile=%s tool=%s keys=%s",
            resolved_api_profile,
            tool_name,
            sorted(filtered_arguments),
        )
        return _normalize_tool_calls({
            "name": tool_name,
            "arguments": filtered_arguments,
        })
    def _schema_tool_names() -> List[str]:
        names: List[str] = []
        for schema in tool_schemas or []:
            if not isinstance(schema, dict):
                continue
            function_schema = schema.get("function")
            name = function_schema.get("name") if isinstance(function_schema, dict) else None
            if isinstance(name, str) and name:
                names.append(name)
        return names

    def _parse_schema_adjacent_call(call_text: str) -> Optional[Dict[str, Any]]:
        """Recover schema-unambiguous ``tool_namearg:value`` calls.

        GLM may remove every separator between the tool name, argument names,
        and successive scalar arguments.  Generic comma-based parsing is not
        safe for JSON arrays or prose containing ``name:`` fragments, so this
        path consumes structured values as balanced units and scalar values by
        their declared schema type.  The whole suffix must be consumed and the
        recovered object must pass the active tool schema.
        """

        def _property_schema(tool_name: str, argument_name: str) -> Dict[str, Any]:
            parameters = _schema_parameters(tool_name)
            properties = parameters.get("properties", {})
            if not isinstance(properties, dict):
                return {}
            schema = properties.get(argument_name)
            return schema if isinstance(schema, dict) else {}

        def _parse_adjacent_arguments(
            tool_name: str,
            remainder: str,
        ) -> Optional[Dict[str, Any]]:
            # Residual protocol or tool-result markers mean the boundary is
            # already corrupted.  Do not turn them into executable strings.
            if re.search(
                r"</?arg_(?:key|value)>|</?tool_call>|\[工具返回结果\]",
                remainder,
                re.IGNORECASE,
            ):
                return None

            argument_names = _schema_argument_names(tool_name)
            if not argument_names:
                return None
            name_order = sorted(argument_names, key=len, reverse=True)
            text = remainder.strip()

            def _coerce_typed_value(raw_value: str, expected_type: Any):
                """Decode quoted structured values without weakening Schema checks.

                GLM-5.2 sometimes serializes an array/object as a quoted shell-style
                value (for example ``key_files="[{'file_path': 'a.md'}]"``).  The
                ordinary scalar coercion correctly keeps this as a string, so make
                one schema-directed inner parse only for array/object properties.
                """
                coerced = _coerce_scalar(raw_value)
                expected_python_type = {
                    "array": list,
                    "object": dict,
                }.get(expected_type)
                if expected_python_type is None or isinstance(coerced, expected_python_type):
                    return True, coerced

                inner = raw_value.strip()
                if (
                    len(inner) >= 2
                    and inner[0] in "\"'"
                    and inner[-1] == inner[0]
                ):
                    inner = inner[1:-1].strip()
                elif (
                    len(inner) >= 2
                    and inner[0] in "\"'"
                    and inner[1] in "[{"
                ):
                    # Observed in task76: GLM opens a shell-style quote before
                    # a complete JSON array but omits only that outer quote.
                    # The inner array/object must still be balanced and parseable.
                    inner = inner[1:].strip()
                parsed_inner = _try_parse_payload(inner)
                if isinstance(parsed_inner, expected_python_type):
                    return True, parsed_inner
                return False, coerced

            def _skip_separators(position: int) -> int:
                while position < len(text) and text[position] in " ,\n\t/":
                    position += 1
                return position

            def _match_marker(position: int, used: set):
                position = _skip_separators(position)
                for candidate_name in name_order:
                    if candidate_name in used:
                        continue
                    marker = re.match(
                        rf"{re.escape(candidate_name)}\s*[:=]\s*",
                        text[position:],
                    )
                    if marker:
                        return candidate_name, position + marker.end()
                return None

            def _parse_from(position: int, used: set) -> Optional[Dict[str, Any]]:
                position = _skip_separators(position)
                if position == len(text):
                    return {}
                marker = _match_marker(position, used)
                if not marker:
                    return None
                argument_name, value_start = marker
                field_schema = _property_schema(tool_name, argument_name)
                expected_type = field_schema.get("type")
                allowed_values = field_schema.get("enum")
                value_candidates = []

                # Arrays/objects are consumed as one balanced value, so commas
                # and field-like prose inside them never become top-level args.
                if value_start < len(text) and text[value_start] in "[{":
                    balanced = _extract_first_balanced_block(text[value_start:])
                    if balanced and text[value_start:].startswith(balanced):
                        parsed = _try_parse_payload(balanced)
                        if parsed is not None:
                            value_candidates.append((parsed, value_start + len(balanced)))

                # Enum values provide an exact, schema-owned boundary.
                if isinstance(allowed_values, list):
                    for enum_value in sorted(
                        (value for value in allowed_values if isinstance(value, str)),
                        key=len,
                        reverse=True,
                    ):
                        if text.startswith(enum_value, value_start):
                            value_candidates.append((enum_value, value_start + len(enum_value)))

                if expected_type in {"boolean", None}:
                    bool_match = re.match(r"(?:true|false)", text[value_start:], re.IGNORECASE)
                    if bool_match:
                        raw_bool = bool_match.group(0)
                        value_candidates.append(
                            (raw_bool.lower() == "true", value_start + len(raw_bool))
                        )
                if expected_type in {"integer", "number", None}:
                    number_pattern = r"-?\d+" if expected_type == "integer" else r"-?(?:\d+(?:\.\d+)?|\.\d+)"
                    number_match = re.match(number_pattern, text[value_start:])
                    if number_match:
                        raw_number = number_match.group(0)
                        value_candidates.append(
                            (_coerce_scalar(raw_number), value_start + len(raw_number))
                        )

                # Find a following declared marker and accept the split only if
                # the complete suffix also parses.  This also handles GLM's
                # quoted array/object values, which are not balanced JSON until
                # their harmless outer quotes are removed schema-directly.
                if expected_type in {"string", "array", "object", None}:
                    next_markers = []
                    for next_name in name_order:
                        if next_name == argument_name or next_name in used:
                            continue
                        for next_match in re.finditer(
                            rf"{re.escape(next_name)}\s*[:=]\s*",
                            text[value_start:],
                        ):
                            next_position = value_start + next_match.start()
                            if next_position > value_start:
                                next_schema = _property_schema(tool_name, next_name)
                                separator_before = text[next_position - 1] in " ,\n\t/"
                                # Separator-free string-to-string splitting is
                                # inherently ambiguous.  Typed following fields
                                # and ordinary separated calls remain recoverable.
                                if (
                                    next_schema.get("type") == "string"
                                    and not separator_before
                                    and not isinstance(allowed_values, list)
                                ):
                                    continue
                                # In the slash-delimited GLM variant, keep the
                                # slash available to _skip_separators but do not
                                # include it in the preceding value. Ordinary
                                # slashes inside paths are ignored unless what
                                # follows is an exact Schema argument marker.
                                if text[next_position - 1] == "/":
                                    next_position -= 1
                                next_markers.append(next_position)
                    for next_position in sorted(set(next_markers)):
                        raw_value = text[value_start:next_position].strip(" ,\n\t")
                        if raw_value:
                            valid_value, coerced_value = _coerce_typed_value(
                                raw_value,
                                expected_type,
                            )
                            if valid_value:
                                value_candidates.append((coerced_value, next_position))

                    raw_tail = text[value_start:].strip(" ,\n\t")
                    if raw_tail:
                        valid_value, coerced_value = _coerce_typed_value(
                            raw_tail,
                            expected_type,
                        )
                        if valid_value:
                            value_candidates.append((coerced_value, len(text)))

                for value, value_end in value_candidates:
                    remaining_arguments = _parse_from(
                        value_end,
                        used | {argument_name},
                    )
                    if remaining_arguments is None:
                        continue
                    return {argument_name: value, **remaining_arguments}
                return None

            parsed_arguments = _parse_from(0, set())
            if not parsed_arguments or not _arguments_match_schema(
                tool_name,
                parsed_arguments,
            ):
                return None
            return parsed_arguments

        # A terminal orphan closing tag is a provider wrapper defect, but direct
        # filesystem/network mutation primitives remain fail-closed. Higher-level
        # workflow tools (notably search_result_classifier and section_writer)
        # are recoverable only when every field is consumed and the complete
        # argument object passes the active Schema below.
        orphan_closer_denied_tools = {
            "bash",
            "concat_section_files",
            "download_files",
            "file_write",
            "process_library_files",
            "process_user_uploaded_files",
            "rag_document_saver",
            "str_replace_based_edit_tool",
            "url_crawler",
        }
        relocated_closer_safe_tools = {
            "file_find_by_name",
            "file_grep_search",
            "file_read",
            "file_read_lines",
            "file_stats",
            "list_workspace",
            "load_json",
        }

        for tool_name in sorted(_schema_tool_names(), key=len, reverse=True):
            if not call_text.startswith(tool_name):
                continue
            remainder = call_text[len(tool_name):]
            parse_remainder = remainder
            orphan_closer_removed = False
            if (
                not re.search(r"<arg_(?:key|value)>|</arg_key>", remainder, re.IGNORECASE)
                and not re.search(r"</?tool_call>|\[工具返回结果\]", remainder, re.IGNORECASE)
            ):
                closing_tags = re.findall(r"</arg_value>", remainder, re.IGNORECASE)
                if len(closing_tags) == 1 and tool_name in relocated_closer_safe_tools:
                    parse_remainder = re.sub(
                        r"</arg_value>", "", remainder, count=1, flags=re.IGNORECASE
                    )
                    orphan_closer_removed = True
                elif tool_name not in orphan_closer_denied_tools:
                    orphan_match = re.fullmatch(
                        r"([\s\S]*?)\s*</arg_value>\s*",
                        remainder,
                        re.IGNORECASE,
                    )
                    if orphan_match:
                        parse_remainder = orphan_match.group(1)
                        orphan_closer_removed = True

            parsed_arguments = _parse_adjacent_arguments(tool_name, parse_remainder)
            if parsed_arguments:
                runtime_logger = globals().get("logger")
                if runtime_logger is not None:
                    runtime_logger.info(
                        "Recovered schema-adjacent tool arguments: "
                        "profile=%s tool=%s keys=%s orphan_closer_removed=%s",
                        resolved_api_profile,
                        tool_name,
                        sorted(parsed_arguments),
                        orphan_closer_removed,
                    )
                return {"name": tool_name, "arguments": parsed_arguments}

            # Preserve the already-supported separated key/value fallback, but
            # require the complete result to satisfy the schema before use.
            if re.search(
                r"</?arg_(?:key|value)>|</?tool_call>|\[工具返回结果\]",
                parse_remainder,
                re.IGNORECASE,
            ):
                continue
            allowed_names = _schema_argument_names(tool_name)
            parsed_arguments = _parse_kv_arguments(parse_remainder)
            filtered_arguments = {
                key: value
                for key, value in parsed_arguments.items()
                if key in set(allowed_names)
            }
            if _arguments_match_schema(tool_name, filtered_arguments):
                return {"name": tool_name, "arguments": filtered_arguments}
        return None

    def _parse_double_angle_arguments(arg_text: str, tool_name: str) -> Dict[str, Any]:
        """Parse ``<<key>>value`` only when key is declared by the tool schema."""
        allowed_names = _schema_argument_names(tool_name)
        if not arg_text or not allowed_names:
            return {}
        name_pattern = "|".join(
            re.escape(name) for name in sorted(allowed_names, key=len, reverse=True)
        )
        marker_pattern = re.compile(rf"<<\s*({name_pattern})\s*>>")
        markers = list(marker_pattern.finditer(arg_text))
        if not markers:
            return {}
        result: Dict[str, Any] = {}
        for index, marker in enumerate(markers):
            value_end = markers[index + 1].start() if index + 1 < len(markers) else len(arg_text)
            value = arg_text[marker.end():value_end].strip()
            if value:
                result[marker.group(1)] = _coerce_scalar(value)
        return result

    def _parse_named_wrapper_argument(arg_text: str) -> Dict[str, Any]:
        """Parse GLM's ``](key): value`` / ``(key): value`` fallback form."""
        if not arg_text:
            return {}
        cleaned = arg_text.strip().lstrip("]").strip()
        match = re.fullmatch(
            r"\(\s*([A-Za-z0-9_\-]+)\s*\)\s*:\s*([\s\S]+)",
            cleaned,
        )
        if not match:
            return {}
        return {match.group(1): _coerce_scalar(match.group(2))}

    def _parse_missing_open_arg_key_argument(
        arg_text: str,
        tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Recover ``key</arg_key><arg_value>value</arg_value>`` safely.

        Some GLM-5.2 endpoints omit only the opening ``<arg_key>`` token.  The
        argument name and decoded value must both match the active tool schema;
        otherwise this recovery deliberately declines instead of letting the
        generic key/value parser scan prose inside a JSON string.
        """
        if not arg_text or not tool_schemas:
            return None
        match = re.fullmatch(
            r"\s*([A-Za-z0-9_\-]+)\s*</arg_key>\s*"
            r"<arg_value>\s*([\s\S]*?)\s*</arg_value>\s*",
            arg_text,
            re.IGNORECASE,
        )
        if not match:
            return None
        argument_name = match.group(1)
        if argument_name not in _schema_argument_names(tool_name):
            return None
        arguments = {argument_name: _coerce_scalar(match.group(2))}
        return arguments if _arguments_match_schema(tool_name, arguments) else None

    def _wrap_single_schema_argument(
        tool_name: str,
        value: Any,
    ) -> Optional[Dict[str, Any]]:
        """Wrap a bare structured value only when one schema field fits it."""
        argument_names = _schema_argument_names(tool_name)
        if len(argument_names) != 1:
            return None
        arguments = {argument_names[0]: value}
        return arguments if _arguments_match_schema(tool_name, arguments) else None

    def _extract_first_balanced_block(text: str) -> Optional[str]:
        """从文本中抓取第一个配平的 {...} 或 [...] 块，正确处理字符串与转义。

        用于 GLM-5.2 退化容错：当工具名与参数之间夹杂垃圾串（如 ,function / "]( / : ）
        导致 arg_text 无法直接解析时，从剩余文本里定位第一个完整 JSON 块作为参数来源。
        """
        if not text:
            return None
        brace_idx = text.find("{")
        bracket_idx = text.find("[")
        candidates = [i for i in (brace_idx, bracket_idx) if i != -1]
        if not candidates:
            return None
        start = min(candidates)
        open_ch = text[start]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_str = False
        quote = ""
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    in_str = False
                continue
            if ch in "\"'":
                in_str = True
                quote = ch
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _parse_tool_call_block(block: str) -> List[Dict[str, Any]]:
        cleaned = block.strip()
        if not cleaned:
            return []
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()
        parsed = _parse_payload(cleaned)
        adjacent_call = _parse_schema_adjacent_call(cleaned)
        if adjacent_call:
            return _normalize_tool_calls(adjacent_call)
        if parsed is not None:
            normalized = _normalize_tool_calls(parsed)
            if normalized:
                return normalized
        match = re.match(r"([A-Za-z0-9_\-]+)\s*\]?\s*(?:\((.*)\))?", cleaned, re.DOTALL)
        if match:
            name = match.group(1)
            remaining_after_name = cleaned[len(name):]
            missing_open_tag_arguments = _parse_missing_open_arg_key_argument(
                remaining_after_name,
                name,
            )
            if missing_open_tag_arguments:
                return _normalize_tool_calls({
                    "name": name,
                    "arguments": missing_open_tag_arguments,
                })
            tagged_arguments = _parse_tagged_arguments(remaining_after_name)
            if tagged_arguments:
                return _normalize_tool_calls({
                    "name": name,
                    "arguments": tagged_arguments,
                })
            double_angle_arguments = _parse_double_angle_arguments(remaining_after_name, name)
            if double_angle_arguments:
                return _normalize_tool_calls({
                    "name": name,
                    "arguments": double_angle_arguments,
                })
            named_wrapper_arguments = _parse_named_wrapper_argument(remaining_after_name)
            if named_wrapper_arguments:
                return _normalize_tool_calls({
                    "name": name,
                    "arguments": named_wrapper_arguments,
                })
            arg_text = (match.group(2) or "").strip()
            if not arg_text:
                # GLM-5.2 sometimes outputs tool calls without parentheses:
                # e.g., "batch_web_search\tqueries: [...]" instead of "batch_web_search(queries: [...])"
                # or "assign_subjective_task_to_writer{\"key\": \"value\"}" without any separator
                remaining = cleaned[len(name):].lstrip(" \t]")
                if remaining and (re.match(r"[A-Za-z0-9_\-]+\s*[=:]", remaining) or remaining[0] in "[{"):
                    arg_text = remaining
            # Try JSON parsing first (GLM-5.2 outputs args as JSON objects like {"key": "value"})
            parsed_args = _parse_payload(arg_text) if arg_text else None
            if isinstance(parsed_args, dict):
                arguments = parsed_args
            else:
                # [FIX] GLM-5.2 退化容错：工具名后可能夹杂垃圾串（如 ,function / "]( / : ），
                # 直接从工具名之后的剩余文本抓第一个配平的 {...} 块作为参数，
                # 仅当解析为 dict 时采用，避免畸形串被 _parse_kv_arguments 解析成空 {} 而丢失参数。
                arguments = None
                json_block = _extract_first_balanced_block(cleaned[len(name):])
                if json_block:
                    block_args = _parse_payload(json_block)
                    if isinstance(block_args, dict):
                        arguments = block_args
                    elif not re.search(
                        r"</?arg_(?:key|value)>",
                        remaining_after_name,
                        re.IGNORECASE,
                    ):
                        arguments = _wrap_single_schema_argument(name, block_args)
                if arguments is None:
                    has_structured_fragment = bool(
                        re.search(r"</?arg_(?:key|value)>", remaining_after_name, re.IGNORECASE)
                        or json_block
                    )
                    arguments = (
                        {}
                        if has_structured_fragment
                        else _parse_kv_arguments(arg_text)
                    )
            return _normalize_tool_calls({"name": name, "arguments": arguments})
        if "," in cleaned:
            name_part, arg_text = cleaned.split(",", 1)
            name = name_part.strip().strip("]")
            if name and re.fullmatch(r"[A-Za-z0-9_\-]+", name):
                arguments = _parse_kv_arguments(arg_text)
                return _normalize_tool_calls({"name": name, "arguments": arguments})
        return []

    def _sanitize_marker_payload(payload: str) -> str:
        cleaned = payload.strip()
        lower_cleaned = cleaned.lower()
        for prefix in ("message|", "function_call|", "tool_call|", "tool_calls|"):
            if lower_cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip()
                break

        if cleaned and cleaned[0] not in "{[":
            brace_idx = cleaned.find("{")
            bracket_idx = cleaned.find("[")
            candidates = [idx for idx in (brace_idx, bracket_idx) if idx != -1]
            if candidates:
                first_json_idx = min(candidates)
                pipe_idx = cleaned.rfind("|", 0, first_json_idx)
                if pipe_idx == -1:
                    prefix = cleaned[:first_json_idx].strip()
                    if not prefix or not re.fullmatch(r"[A-Za-z0-9_\-]+", prefix):
                        cleaned = cleaned[first_json_idx:]
        if cleaned:
            last_brace = cleaned.rfind("}")
            last_bracket = cleaned.rfind("]")
            last_idx = max(last_brace, last_bracket)
            if last_idx != -1:
                cleaned = cleaned[: last_idx + 1]
        return cleaned

    def _split_tool_name(payload: str) -> (Optional[str], str):
        json_start = None
        brace_idx = payload.find("{")
        bracket_idx = payload.find("[")
        candidates = [idx for idx in (brace_idx, bracket_idx) if idx != -1]
        if candidates:
            json_start = min(candidates)
        if json_start is None:
            return None, payload
        pipe_idx = payload.rfind("|", 0, json_start)
        if pipe_idx != -1:
            return payload[:pipe_idx].strip(), payload[json_start:]
        prefix = payload[:json_start].strip()
        if prefix and re.fullmatch(r"[A-Za-z0-9_\-]+", prefix):
            return prefix, payload[json_start:]
        return None, payload

    def _extract_json_candidate(payload: str) -> Optional[str]:
        start_candidates = [idx for idx in (payload.find("{"), payload.find("[")) if idx != -1]
        if not start_candidates:
            return None
        start_idx = min(start_candidates)
        stack = []
        in_string = False
        escape = False
        for idx in range(start_idx, len(payload)):
            ch = payload[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
                continue
            if ch in "]}":
                if not stack:
                    continue
                opening = stack.pop()
                if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                    return None
                if not stack:
                    return payload[start_idx: idx + 1]
        return payload[start_idx:]

    def _balance_json(payload: str) -> str:
        stack = []
        in_string = False
        escape = False
        for ch in payload:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "]}":
                if stack:
                    opening = stack.pop()
                    if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                        stack.clear()
                        break
        if in_string or not stack:
            return payload
        closing = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        return payload + closing

    def _repair_payload(payload: str) -> str:
        cleaned = payload.strip()
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return _balance_json(cleaned)

    def _parse_payload(payload: str) -> Optional[Any]:
        candidate = _extract_json_candidate(payload) or payload
        repaired_candidate = _repair_payload(candidate)
        repaired_payload = _repair_payload(payload)
        for text in dict.fromkeys([repaired_candidate, repaired_payload, candidate, payload]):
            try:
                return json.loads(text)
            except Exception:
                try:
                    import yaml
                    return yaml.safe_load(text)
                except Exception:
                    try:
                        import ast
                        return ast.literal_eval(text)
                    except Exception:
                        continue
        return None

    # 1) Legacy marker format
    marker_blocks = re.findall(r"\[unused11\]([\s\S]*?)\[unused12\]", content)
    if not marker_blocks and "[unused11]" in content:
        marker_blocks = [seg for seg in content.split("[unused11]")[1:] if seg.strip()]
    for block in marker_blocks:
        try:
            cleaned_block = _sanitize_marker_payload(block)
            if not cleaned_block:
                continue
            tool_name, json_payload = _split_tool_name(cleaned_block)
            parsed = _parse_payload(json_payload)
            if parsed is None:
                continue
            if tool_name and isinstance(parsed, dict) and "name" not in parsed:
                normalized = _normalize_tool_calls({"name": tool_name, "arguments": parsed})
            else:
                normalized = _normalize_tool_calls(parsed)
            if normalized:
                return normalized
        except Exception:
            continue

    # 1.2) vLLM-style <|tool_call_start|>...<|tool_call_end|> markers
    #      Used by managed inference services (e.g. intranet pangu_ultra_moe)
    #      where the service applies its own chat template and the model emits
    #      vLLM special tokens instead of [unused11]/[unused12].
    vllm_start = "<|tool_call_start|>"
    vllm_end = "<|tool_call_end|>"
    vllm_blocks = re.findall(
        re.escape(vllm_start) + r"([\s\S]*?)" + re.escape(vllm_end),
        content,
    )
    if not vllm_blocks and vllm_start in content:
        # Model may omit the closing marker; split on opening marker and
        # strip any stray closing markers from each segment.
        raw_segments = content.split(vllm_start)[1:]
        vllm_blocks = []
        for seg in raw_segments:
            if vllm_end in seg:
                seg = seg[: seg.index(vllm_end)]
            if seg.strip():
                vllm_blocks.append(seg)
    for block in vllm_blocks:
        try:
            cleaned_block = _sanitize_marker_payload(block)
            if not cleaned_block:
                continue
            tool_name, json_payload = _split_tool_name(cleaned_block)
            parsed = _parse_payload(json_payload)
            if parsed is None:
                continue
            if tool_name and isinstance(parsed, dict) and "name" not in parsed:
                normalized = _normalize_tool_calls({"name": tool_name, "arguments": parsed})
            else:
                normalized = _normalize_tool_calls(parsed)
            if normalized:
                return normalized
        except Exception:
            continue

    # 1.5) Split on every opening marker first. GLM may omit the closing
    # marker; an EOF-based regex would then absorb the following call into the
    # previous call's last argument.
    tool_call_blocks = []
    tool_call_markers = list(re.finditer(r"<tool_call>", content, re.IGNORECASE))
    for index, marker in enumerate(tool_call_markers):
        block_end = (
            tool_call_markers[index + 1].start()
            if index + 1 < len(tool_call_markers)
            else len(content)
        )
        block = content[marker.end():block_end]
        close_match = re.search(r"</tool_call>", block, re.IGNORECASE)
        if close_match:
            block = block[:close_match.start()]
        if block.strip():
            tool_call_blocks.append(block)
    parsed_tool_calls: List[Dict[str, Any]] = []
    for block in tool_call_blocks:
        try:
            normalized = _parse_tool_call_block(block)
            if _tool_calls_need_schema_recovery(normalized):
                recovered = _parse_displaced_arg_tag_block(block)
                if recovered:
                    normalized = recovered
            if normalized:
                parsed_tool_calls.extend(normalized)
        except Exception:
            continue
    if parsed_tool_calls:
        return parsed_tool_calls

    # 1.6) Fallback: scan for function-style tool calls in plain text
    for line in content.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if not re.match(r"^\(?\s*[A-Za-z0-9_\-]+\s*(?:,|\()", candidate):
            continue
        try:
            normalized = _parse_tool_call_block(candidate)
            if normalized:
                return normalized
        except Exception:
            continue

    # 2) Markdown json block format
    json_blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
    for block in json_blocks:
        try:
            parsed = _parse_payload(block.strip())
            if parsed is None:
                continue
            normalized = _normalize_tool_calls(parsed)
            if normalized:
                return normalized
        except Exception:
            continue

    # 3) Bare JSON payload (no markers or code fences)
    parsed = _parse_payload(content) or _try_parse_payload(content)
    if parsed is not None:
        normalized = _normalize_tool_calls(parsed)
        if normalized:
            return normalized

    return []


def get_storage_config() -> Dict[str, Any]:
    """Get storage and trajectory configuration"""
    return {
        "trajectory_storage_path": config.trajectory_storage_path,
        "report_output_path": config.report_output_path,
        "document_analysis_path": config.document_analysis_path
    }


def get_mcp_config() -> Dict[str, Any]:
    """Get MCP server specific configuration"""
    return {
        "server_url": config.mcp_server_url,
        "auth_token": config.mcp_auth_token,
        "use_stdio": config.mcp_use_stdio,
        "timeout": config.timeout
    }

def get_rag_config() -> Dict[str, Any]:
    """Get RAG knowledge base configuration"""
    return {
        "api_url": config.rag_api_url,
        "app_code": config.rag_app_code,
        "default_repo_id": config.rag_default_repo_id,
        "default_page_size": config.rag_default_page_size,
        "search_source_rag": config.search_source_rag,
        "timeout": config.timeout
    }


def get_proxy_config() -> Dict[str, str]:
    """
    Get proxy configuration for requests library.
    Returns empty dict if no proxy is configured, allowing requests to use system proxy.
    
    Returns:
        Dict with 'http' and 'https' keys if proxy is configured, otherwise empty dict
    """
    # If environment variables are set, return empty dict to let requests auto-detect
    # This allows system proxy to work automatically
    if config.http_proxy or config.https_proxy:
        proxy_dict = {}
        if config.http_proxy:
            proxy_dict['http'] = config.http_proxy
        if config.https_proxy:
            proxy_dict['https'] = config.https_proxy
        return proxy_dict
    
    # Return empty dict to allow requests to use system proxy automatically
    return {}


# Example usage and testing
if __name__ == "__main__":
    print("=== Multi Agent System Configuration ===")
    print(f"Debug Mode: {config.debug_mode}")
    print(f"Custom LLM Service URL: {config.model_request_url}")
    print(f"Available Search Providers: {config.get_available_search_providers()}")
    print("\nConfiguration Summary:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
