# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
from pathlib import Path
from dotenv import load_dotenv


# Load .env file from config directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration class for API keys and settings"""
    
    # Custom LLM Service Configuration
    # Your own deployed LLM service accessed via requests
    model_request_url: Optional[str] = None
    model_request_token: Optional[str] = None
    model_name: str = "pangu_auto"  # Default model name
    
    # Custom Planner Mode
    planner_mode: str = "auto"  # Default planner mode
    
    # MCP Server Configuration
    mcp_server_url: Optional[str] = None
    mcp_auth_token: Optional[str] = None
    mcp_use_stdio: bool = True  # Default to stdio for backward compatibility
    
    # Search Engine Configuration (Generic)
    search_engine_base_url: Optional[str] = None
    search_engine_api_keys: Optional[str] = None  # Can be comma-separated for rotation
    
    # URL Crawler Configuration (Generic)
    url_crawler_base_url: Optional[str] = None
    url_crawler_api_keys: Optional[str] = None  # Can be comma-separated for rotation
    url_crawler_max_tokens: int = 100000
    
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
        self.model_name = os.getenv('MODEL_NAME', 'pangu-auto')
        
        # Custom Planner Mode
        self.planner_mode = os.getenv("PLANNER_MODE", self.planner_mode)
        
        # MCP Server
        self.mcp_server_url = os.getenv("MCP_SERVER_URL")
        self.mcp_auth_token = os.getenv("MCP_AUTH_TOKEN")
        self.mcp_use_stdio = os.getenv("MCP_USE_STDIO", "true").lower() == "true"
        
        # Search Engine Configuration
        self.search_engine_base_url = os.getenv("SEARCH_ENGINE_BASE_URL")
        self.search_engine_api_keys = os.getenv("SEARCH_ENGINE_API_KEYS")
        
        # URL Crawler Configuration
        self.url_crawler_base_url = os.getenv("URL_CRAWLER_BASE_URL")
        self.url_crawler_api_keys = os.getenv("URL_CRAWLER_API_KEYS")
        self.url_crawler_max_tokens = int(os.getenv("URL_CRAWLER_MAX_TOKENS", self.url_crawler_max_tokens))
        
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
    """Get generic URL crawler configuration"""
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