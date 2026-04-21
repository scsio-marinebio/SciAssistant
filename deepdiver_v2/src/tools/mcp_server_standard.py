# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#!/usr/bin/env python3
"""
Demo-Ready MCP Server - New Standard Implementation
Combines robust session management with comprehensive tool definitions.
Features: workspace isolation, tool call tracking, rate limiting, security, and full tool suite.
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
import yaml
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread, Event
from typing import Any, Dict, List, Optional

# Third-party imports
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
import uvicorn

# Add project root to Python path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.status_codes import JsonRpcErr
from http import HTTPStatus

# Handle both relative and absolute imports
try:
    from src.tools.mcp_tools import MCPTools, get_tool_schemas
    from .mcp_tools_async import AsyncMCPTools
except ImportError:
    # Fallback for direct script execution
    from src.tools.mcp_tools import MCPTools, get_tool_schemas
    try:
        from src.tools.mcp_tools_async import AsyncMCPTools
    except ImportError:
        AsyncMCPTools = None

# Workspace knowledge manager disabled
WORKSPACE_KNOWLEDGE_AVAILABLE = False

# Configure structured logging - write to logs directory like app.log
# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# Force configure root logger (basicConfig may be ignored if already configured)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers to avoid duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Add file handler
file_handler = logging.FileHandler(log_dir / 'mcp_server.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.info(f"MCP Server logging configured: {log_dir / 'mcp_server.log'}")

# ================ CONFIGURATION ================


@dataclass
class ServerConfig:
    """Server configuration with only actually implemented options"""
    # Server Core Settings
    host: str = "127.0.0.1"
    port: int = 6274
    debug_mode: bool = False
    
    # Session Management
    session_ttl_seconds: int = 3600  # 1 hour default
    max_sessions: int = 1000
    cleanup_interval_seconds: int = 300  # 5 minutes
    enable_session_keepalive: bool = True
    keepalive_touch_interval: int = 300
    
    # Request Handling
    request_timeout_seconds: int = 120
    max_request_size_mb: int = 10
    
    # Client Rate Limiting (per IP)
    rate_limit_requests_per_minute: int = 300
    
    # Workspace Management
    base_workspace_dir: str = "workspaces"
    
    # Tool Call Tracking & Logging
    enable_tool_tracking: bool = True
    max_tracked_calls_per_session: int = 1000
    track_detailed_errors: bool = True
    

    
    # Per-tool Rate Limiting Configuration
    tool_rate_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ServerConfig':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract configuration sections with defaults
            server_config = config_data.get('server', {})
            tracking_config = config_data.get('tracking', {})
            tool_rate_limits = config_data.get('tool_rate_limits', {})
            
            return cls(
                # Server Core Settings
                host=server_config.get('host', "127.0.0.1"),
                port=server_config.get('port', 6274),
                debug_mode=server_config.get('debug_mode', False),
                
                # Session Management
                session_ttl_seconds=server_config.get('session_ttl_seconds', 3600),
                max_sessions=server_config.get('max_sessions', 1000),
                cleanup_interval_seconds=server_config.get('cleanup_interval_seconds', 300),
                enable_session_keepalive=server_config.get('enable_session_keepalive', True),
                keepalive_touch_interval=server_config.get('keepalive_touch_interval', 300),
                
                # Request Handling
                request_timeout_seconds=server_config.get('request_timeout_seconds', 120),
                max_request_size_mb=server_config.get('max_request_size_mb', 10),
                
                # Client Rate Limiting
                rate_limit_requests_per_minute=server_config.get('rate_limit_requests_per_minute', 300),
                
                # Workspace Management
                base_workspace_dir=server_config.get('base_workspace_dir', "workspaces"),
                
                # Tool Call Tracking & Logging
                enable_tool_tracking=tracking_config.get('enable_tool_tracking', True),
                max_tracked_calls_per_session=tracking_config.get('max_tracked_calls_per_session', 1000),
                track_detailed_errors=tracking_config.get('track_detailed_errors', True),
                
                # Per-tool Rate Limiting
                tool_rate_limits=tool_rate_limits
            )
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()

# Global configuration instance - will be set during startup
config: Optional[ServerConfig] = None

# ================ GLOBAL PER-TOOL RATE LIMITING ================


@dataclass
class ToolRateLimit:
    """Rate limit configuration for a specific tool"""
    requests_per_minute: float
    requests_per_hour: float
    burst_limit: int


class GlobalToolRateLimiter:
    """
    Global rate limiter that controls QPS to external APIs per tool.
    This is shared across all sessions and clients to manage upstream service load.
    """
    
    def __init__(self, tool_rate_limits: Dict[str, Dict[str, int]]):
        self.tool_limits: Dict[str, ToolRateLimit] = {}
        self.tool_requests: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()
        
        # Initialize rate limits for each tool
        for tool_name, limits_config in tool_rate_limits.items():
            self.tool_limits[tool_name] = ToolRateLimit(
                requests_per_minute=limits_config.get('requests_per_minute', float('inf')),
                requests_per_hour=limits_config.get('requests_per_hour', float('inf')),
                burst_limit=limits_config.get('burst_limit', 10)
            )
            self.tool_requests[tool_name] = deque()
        
        logger.info(f"Initialized global tool rate limiter for {len(self.tool_limits)} tools")
    
    async def is_allowed(self, tool_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if a request to the specified tool is allowed based on global rate limits.
        
        Returns:
            tuple[bool, Optional[str]]: (allowed, reason_if_denied)
        """
        if tool_name not in self.tool_limits:
            # Tool not configured for rate limiting - allow
            return True, None
        
        async with self.lock:
            now = time.time()
            limits = self.tool_limits[tool_name]
            requests = self.tool_requests[tool_name]
            
            # Clean old requests outside the time windows
            self._cleanup_old_requests(requests, now)
            
            # Check various time window limits
            recent_requests = list(requests)
            
            # Check burst limit (rapid requests in last second) - only if specified
            if limits.burst_limit != float('inf'):
                burst_count = sum(1 for req_time in recent_requests if now - req_time < 1.0)
                if burst_count >= limits.burst_limit:
                    return False, f"Tool '{tool_name}' burst limit exceeded ({limits.burst_limit} requests/burst)"
            
            # Check per-minute limit - only if specified
            if limits.requests_per_minute != float('inf'):
                minute_count = sum(1 for req_time in recent_requests if now - req_time < 60.0)
                if minute_count >= limits.requests_per_minute:
                    return False, f"Tool '{tool_name}' per-minute limit exceeded ({limits.requests_per_minute} requests/minute)"
            
            # Check per-hour limit - only if specified
            if limits.requests_per_hour != float('inf'):
                hour_count = sum(1 for req_time in recent_requests if now - req_time < 3600.0)
                if hour_count >= limits.requests_per_hour:
                    return False, f"Tool '{tool_name}' per-hour limit exceeded ({limits.requests_per_hour} requests/hour)"
            
            return True, None
    
    async def record_request(self, tool_name: str):
        """Record a successful request for rate limiting tracking"""
        if tool_name not in self.tool_limits:
            return
        
        async with self.lock:
            now = time.time()
            self.tool_requests[tool_name].append(now)
            
            # Keep deque size manageable (only keep last hour of requests)
            self._cleanup_old_requests(self.tool_requests[tool_name], now)

    @staticmethod
    def _cleanup_old_requests(requests: deque, now: float):
        """Remove requests older than 1 hour to keep memory usage bounded"""
        while requests and now - requests[0] > 3600.0:  # 1 hour
            requests.popleft()
    
    async def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """Get current usage statistics for a tool"""
        if tool_name not in self.tool_limits:
            return {"error": f"Tool '{tool_name}' not configured for rate limiting"}
        
        async with self.lock:
            now = time.time()
            requests = self.tool_requests[tool_name]
            limits = self.tool_limits[tool_name]
            
            # Clean old requests first
            self._cleanup_old_requests(requests, now)
            
            recent_requests = list(requests)
            
            return {
                "tool_name": tool_name,
                "current_usage": {
                    "last_second": sum(1 for req_time in recent_requests if now - req_time < 1.0),
                    "last_minute": sum(1 for req_time in recent_requests if now - req_time < 60.0),
                    "last_hour": sum(1 for req_time in recent_requests if now - req_time < 3600.0)
                },
                "limits": {
                    "requests_per_minute": limits.requests_per_minute if limits.requests_per_minute != float('inf') else None,
                    "requests_per_hour": limits.requests_per_hour if limits.requests_per_hour != float('inf') else None,
                    "burst_limit": limits.burst_limit if limits.burst_limit != float('inf') else None
                },
                "utilization": {
                    "minute_utilization": sum(1 for req_time in recent_requests if now - req_time < 60.0) / limits.requests_per_minute if limits.requests_per_minute != float('inf') else 0,
                    "hour_utilization": sum(1 for req_time in recent_requests if now - req_time < 3600.0) / limits.requests_per_hour if limits.requests_per_hour != float('inf') else 0
                }
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools"""
        return {
            tool_name: self.get_tool_stats(tool_name)
            for tool_name in self.tool_limits.keys()
        }

# Global tool rate limiter instance - will be initialized during startup
global_tool_rate_limiter: Optional[GlobalToolRateLimiter] = None

# ================ TOOL DEFINITIONS ================

# Tool execution function mapping - maps tool names to their implementation functions


def get_tool_function(tool_name: str):
    """Get the actual function for a tool"""
    tool_map = {
        "batch_web_search": lambda tools, **kwargs: tools.batch_web_search(**kwargs),
        "url_crawler": lambda tools, **kwargs: tools.url_crawler(**kwargs),
        "download_files": lambda tools, **kwargs: tools.download_files(**kwargs),
        "list_workspace": lambda tools, **kwargs: tools.list_workspace(**kwargs),
        "str_replace_based_edit_tool": lambda tools, **kwargs: tools.str_replace_based_edit_tool(**kwargs),
        "file_stats": lambda tools, **kwargs: tools.file_stats(**kwargs),
        "file_read": lambda tools, **kwargs: tools.file_read(**kwargs),
        "file_read_lines": lambda tools, **kwargs: tools.file_read_lines(**kwargs),
        "content_preview": lambda tools, **kwargs: tools.content_preview(**kwargs),
        "file_write": lambda tools, **kwargs: tools.file_write(**kwargs),
        "file_grep_search": lambda tools, **kwargs: tools.file_grep_search(**kwargs),
        "file_grep_with_context": lambda tools, **kwargs: tools.file_grep_with_context(**kwargs),
        "file_find_by_name": lambda tools, **kwargs: tools.file_find_by_name(**kwargs),
        "bash": lambda tools, **kwargs: tools.bash(**kwargs),
        "task_done": lambda tools, **kwargs: tools.task_done(**kwargs),
        "think": lambda tools, **kwargs: tools.think(**kwargs),
        "reflect": lambda tools, **kwargs: tools.reflect(**kwargs),
        "document_qa": lambda tools, **kwargs: tools.document_qa(**kwargs),
        "extract_markdown_toc": lambda tools, **kwargs: tools.extract_markdown_toc(**kwargs),
        "extract_markdown_section": lambda tools, **kwargs: tools.extract_markdown_section(**kwargs),

        "document_extract": lambda tools, **kwargs: tools.document_extract(**kwargs),
        "search_result_classifier": lambda tools, **kwargs: tools.search_result_classifier(**kwargs),
        "info_seeker_subjective_task_done": None,
        "writer_subjective_task_done": None,
        "section_writer": lambda tools, **kwargs: tools.section_writer(**kwargs),
        "concat_section_files": lambda tools, **kwargs: tools.concat_section_files(**kwargs),
        
        # Internal tools - available to server but NOT exposed to agents via tool schemas
        "internal_file_read_unlimited": lambda tools, **kwargs: tools.internal_file_read_unlimited(**kwargs),
        # Resource Library - PubMed tools
        "search_pubmed_key_words": lambda tools, **kwargs: tools.search_pubmed_key_words(**kwargs),
        "search_pubmed_advanced": lambda tools, **kwargs: tools.search_pubmed_advanced(**kwargs),
        "get_pubmed_article": lambda tools, **kwargs: tools.get_pubmed_article(**kwargs),
        
        # Resource Library - arXiv tools
        "arxiv_search": lambda tools, **kwargs: tools.arxiv_search(**kwargs),
        "arxiv_read_paper": lambda tools, **kwargs: tools.arxiv_read_paper(**kwargs),
        
        # Resource Library - medRxiv tools
        "medrxiv_search": lambda tools, **kwargs: tools.medrxiv_search(**kwargs),
        "medrxiv_read_paper": lambda tools, **kwargs: tools.medrxiv_read_paper(**kwargs),
        
        # Resource Library - Google Scholar tools
        "google_scholar_search": lambda tools, **kwargs: tools.google_scholar_search(**kwargs),
        "advanced_google_scholar_search": lambda tools, **kwargs: tools.advanced_google_scholar_search(**kwargs),
        "google_scholar_get_paper": lambda tools, **kwargs: tools.google_scholar_get_paper(**kwargs),
        
        # Resource Library - Springer Nature tools
        "springer_search": lambda tools, **kwargs: tools.springer_search(**kwargs),
        "springer_get_article": lambda tools, **kwargs: tools.springer_get_article(**kwargs),
    }
    return tool_map.get(tool_name)


# ================ TOOL CALL TRACKING ================


@dataclass
class ToolCallLog:
    """Individual tool call log entry"""
    call_id: str
    timestamp: datetime
    tool_name: str
    input_args: Dict[str, Any]
    output_result: Dict[str, Any]
    success: bool
    duration_ms: float
    error_details: Optional[str] = None
    session_id: str = ""
    agent_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "call_id": self.call_id,
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "input_args": self.input_args,
            "output_result": self.output_result,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error_details": self.error_details,
            "session_id": self.session_id,
            "agent_info": self.agent_info
        }


class ToolCallTracker:
    """Tracks and saves tool calls to workspace-specific files"""
    
    def __init__(self, workspace_path: Path, session_id: str):
        self.workspace_path = workspace_path
        self.session_id = session_id
        self.logs_dir = workspace_path / "tool_call_logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create daily log file
        today = datetime.now().strftime("%Y-%m-%d")
        self.current_log_file = self.logs_dir / f"tool_calls_{today}.jsonl"
        self.summary_file = self.logs_dir / "session_summary.json"
        
        # Track call counts
        self.call_count = 0
        self.tool_usage_stats = defaultdict(int)
        
        # Initialize session summary
        self._initialize_session_summary()
    
    def _initialize_session_summary(self):
        """Initialize or update session summary file"""
        summary = {
            "session_id": self.session_id,
            "session_start": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_tool_calls": 0,
            "tool_usage_stats": {},
            "agent_activity": {},
            "workspace_path": str(self.workspace_path)
        }
        
        # Load existing summary if it exists
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r') as f:
                    existing_summary = json.load(f)
                    summary.update(existing_summary)
                    # Don't overwrite session_start if it already exists
                    if "session_start" in existing_summary:
                        summary["session_start"] = existing_summary["session_start"]
            except Exception as e:
                logger.warning(f"Could not load existing session summary: {e}")
        
        self._save_summary(summary)
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save session summary to file"""
        try:
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session summary: {e}")
    
    def log_tool_call(self, 
                     tool_name: str, 
                     input_args: Dict[str, Any], 
                     output_result: Dict[str, Any],
                     success: bool,
                     duration_ms: float,
                     error_details: Optional[str] = None,
                     agent_info: Optional[Dict[str, Any]] = None) -> str:
        """Log a tool call and return the call ID"""
        
        if not config.enable_tool_tracking:
            return ""
        
        # Respect max call limit per session
        if self.call_count >= config.max_tracked_calls_per_session:
            logger.warning(f"Max tracked calls reached for session {self.session_id}")
            return ""
        
        call_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Create log entry
        log_entry = ToolCallLog(
            call_id=call_id,
            timestamp=timestamp,
            tool_name=tool_name,
            input_args=self._sanitize_args(input_args),
            output_result=self._sanitize_result(output_result),
            success=success,
            duration_ms=duration_ms,
            error_details=error_details if config.track_detailed_errors else None,
            session_id=self.session_id,
            agent_info=agent_info
        )
        
        # Save to JSONL file (one JSON object per line)
        try:
            with open(self.current_log_file, 'a', encoding="utf-8") as f:
                f.write(json.dumps(log_entry.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to save tool call log: {e}")
        
        # Update session summary
        self._update_session_summary(log_entry)
        
        self.call_count += 1
        self.tool_usage_stats[tool_name] += 1
        
        return call_id

    @staticmethod
    def _sanitize_args(args: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize arguments for logging (remove sensitive data)"""
        sanitized = {}
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:1000] + "... [truncated]"
            elif key.lower() in ['password', 'token', 'secret', 'key']:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized
    
    def _sanitize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize result for logging (remove large content)"""
        if not isinstance(result, dict):
            return result
        
        sanitized = {}
        for key, value in result.items():
            if isinstance(value, str) and len(value) > 2000:
                sanitized[key] = value[:2000] + "... [truncated]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_result(value)
            else:
                sanitized[key] = value
        return sanitized
    
    def _update_session_summary(self, log_entry: ToolCallLog):
        """Update session summary with new tool call"""
        try:
            summary = {
                "session_id": self.session_id,
                "last_updated": datetime.now().isoformat(),
                "total_tool_calls": self.call_count + 1,
                "tool_usage_stats": dict(self.tool_usage_stats),
                "workspace_path": str(self.workspace_path)
            }
            
            # Load existing summary
            if self.summary_file.exists():
                with open(self.summary_file, 'r') as f:
                    existing_summary = json.load(f)
                    summary.update(existing_summary)
            
            # Update with new data
            summary["last_updated"] = datetime.now().isoformat()
            summary["total_tool_calls"] = self.call_count + 1
            summary["tool_usage_stats"] = dict(self.tool_usage_stats)
            summary["tool_usage_stats"][log_entry.tool_name] = self.tool_usage_stats[log_entry.tool_name] + 1
            
            # Track agent activity
            if log_entry.agent_info:
                agent_type = log_entry.agent_info.get('type', 'unknown')
                if 'agent_activity' not in summary:
                    summary['agent_activity'] = {}
                if agent_type not in summary['agent_activity']:
                    summary['agent_activity'][agent_type] = {
                        'tool_calls': 0,
                        'last_active': log_entry.timestamp.isoformat()
                    }
                summary['agent_activity'][agent_type]['tool_calls'] += 1
                summary['agent_activity'][agent_type]['last_active'] = log_entry.timestamp.isoformat()
            
            self._save_summary(summary)
            
        except Exception as e:
            logger.error(f"Failed to update session summary: {e}") 

# ================ SESSION KEEP-ALIVE FOR LONG OPERATIONS ================


class KeepAliveSessionWrapper:
    """Wrapper that keeps a session alive during long-running operations"""
    
    def __init__(self, session: 'Session', touch_interval: int = 300):  # Touch every 5 minutes
        self.session = session
        self.touch_interval = touch_interval
        self.keep_alive_thread = None
        self.stop_event = Event()
        self.active = False
    
    def start_keep_alive(self):
        """Start the keep-alive mechanism"""
        if self.active:
            return
        
        self.active = True
        self.stop_event.clear()
        
        def keep_alive_worker():
            while not self.stop_event.wait(self.touch_interval):
                try:
                    self.session.touch()
                    logger.debug("Keep-alive: Touched session {%s}", self.session.id)
                except Exception as e:
                    logger.error(f"Keep-alive error for session {self.session.id}: {e}")
                    break
        
        self.keep_alive_thread = Thread(target=keep_alive_worker, daemon=True)
        self.keep_alive_thread.start()
        logger.info(f"Started keep-alive for session {self.session.id}")
    
    def stop_keep_alive(self):
        """Stop the keep-alive mechanism"""
        if not self.active:
            return
        
        self.active = False
        self.stop_event.set()
        
        if self.keep_alive_thread and self.keep_alive_thread.is_alive():
            self.keep_alive_thread.join(timeout=1.0)
        
        # Final touch
        try:
            self.session.touch()
        except Exception as e:
            logger.error(f"Final keep-alive touch error for session {self.session.id}: {e}")
        
        logger.info(f"Stopped keep-alive for session {self.session.id}")
    
    def __enter__(self):
        self.start_keep_alive()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_keep_alive()

# ================ SESSION MANAGEMENT ================


@dataclass
class Session:
    """Thread-safe session data structure with workspace management"""
    id: str
    created_at: datetime
    last_accessed: datetime
    initialized: bool = False
    request_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    workspace_path: Optional[Path] = None
    mcp_tools: Optional[MCPTools] = None
    tool_tracker: Optional[ToolCallTracker] = None

    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_accessed > timedelta(seconds=ttl_seconds)
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
        self.request_count += 1
    
    def get_mcp_tools(self, prefer_async: bool = True) -> MCPTools:
        """Get or create MCP tools instance for this session"""
        if self.mcp_tools is None:
            # Use async tools if available and preferred
            if prefer_async and AsyncMCPTools is not None:
                self.mcp_tools = AsyncMCPTools(workspace_path=str(self.workspace_path) if self.workspace_path else None)
            else:
                self.mcp_tools = MCPTools(workspace_path=str(self.workspace_path) if self.workspace_path else None)
        return self.mcp_tools
    
    def get_tool_tracker(self) -> Optional[ToolCallTracker]:
        """Get or create tool call tracker for this session"""
        if config.enable_tool_tracking and self.workspace_path:
            if self.tool_tracker is None:
                self.tool_tracker = ToolCallTracker(self.workspace_path, self.id)
            return self.tool_tracker
        return None
    

    
class AsyncRLock:
    """异步可重入锁，模拟 threading.RLock 的异步版本"""
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner: Optional[asyncio.Task] = None  # 记录持有锁的协程任务
        self._count = 0  # 重入次数

    async def acquire(self):
        current_task = asyncio.current_task()
        # 如果当前协程已持有锁，直接增加重入次数
        if self._owner == current_task:
            self._count += 1
            return
        # 否则等待获取锁
        await self._lock.acquire()
        self._owner = current_task
        self._count = 1

    async def release(self):
        if self._owner != asyncio.current_task():
            raise RuntimeError("不能释放非当前协程持有的锁")
        self._count -= 1
        if self._count == 0:  # 重入次数归零时，真正释放锁
            self._owner = None
            self._lock.release()

    # 支持 async with 语法
    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()


class ThreadSafeSessionManager:
    """Thread-safe session manager with workspace management"""
    
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 1000, base_workspace_dir: str = "workspaces"):
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        
        # 获取项目根目录（包含 app.py 的目录）
        current_file = Path(__file__).resolve()
        project_root = None
        
        # 向上查找包含 app.py 的目录
        for parent in [current_file.parent] + list(current_file.parents):
            if (parent / "app.py").exists():
                project_root = parent
                break
        
        # 如果找不到，使用当前工作目录
        if project_root is None:
            project_root = current_file.parent.parent.parent.parent
            logger.warning(f"Could not find app.py, using fallback project root: {project_root}")
        
        # 将 base_workspace_dir 设置为项目根目录下的路径
        self.base_workspace_dir = project_root / base_workspace_dir
        self.base_workspace_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Workspace base directory initialized at: {self.base_workspace_dir}")
        
        # Thread-safe session storage
        self.sessions: Dict[str, Session] = {}
        self.lock = AsyncRLock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    async def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        
        async with self.lock:
            # Check session limits
            if len(self.sessions) >= self.max_sessions:
                await self._cleanup_oldest_sessions()
            
            # Create workspace directory
            workspace_path = self.base_workspace_dir / session_id
            workspace_path.mkdir(exist_ok=True, parents=True)
            
            # Create session
            session = Session(
                id=session_id,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                workspace_path=workspace_path
            )
            
            self.sessions[session_id] = session
            
            logger.info(f"Created session {session_id} with workspace {workspace_path}")
            return session_id
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID if it exists and is not expired"""
        async with self.lock:
            session = self.sessions.get(session_id)
            if session and not session.is_expired(self.ttl_seconds):
                session.touch()
                return session
            elif session:
                # Remove expired session
                del self.sessions[session_id]
                logger.info(f"Removed expired session {session_id}")
            return None
    
    async def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one using the provided session_id"""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session
            
            # Session doesn't exist but session_id was provided - create session with this ID
            # This allows external systems (like cli/a.py) to pre-create workspace directories
            async with self.lock:
                # Check session limits
                if len(self.sessions) >= self.max_sessions:
                    await self._cleanup_oldest_sessions()
                
                # Use the provided session_id instead of generating a new one
                workspace_path = self.base_workspace_dir / session_id
                workspace_path.mkdir(exist_ok=True, parents=True)
                
                # Create session with the provided ID
                session = Session(
                    id=session_id,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    workspace_path=workspace_path
                )
                
                self.sessions[session_id] = session
                logger.info(f"Created session {session_id} with existing workspace {workspace_path}")
                return session
        
        # No session_id provided - create new session with generated UUID
        new_session_id = await self.create_session()
        return self.sessions[new_session_id]
    
    async def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        async with self.lock:
            expired_sessions = []
            for session_id, session in self.sessions.items():
                if session.is_expired(self.ttl_seconds):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
                logger.info(f"Cleaned up expired session {session_id}")
    
    async def _cleanup_oldest_sessions(self):
        """Remove oldest sessions when limit is reached"""
        async with self.lock:
            if len(self.sessions) < self.max_sessions:
                return
            
            # Sort by last accessed time and remove oldest
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_accessed
            )
            
            sessions_to_remove = len(self.sessions) - self.max_sessions + 10  # Remove extra
            for i in range(sessions_to_remove):
                if i < len(sorted_sessions):
                    session_id = sorted_sessions[i][0]
                    del self.sessions[session_id]
                    logger.info(f"Removed old session {session_id} due to session limit")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(config.cleanup_interval_seconds)
                    # Run async method in sync context
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self._cleanup_expired_sessions())
                    loop.close()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        import threading
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Started session cleanup thread")
    

    
    async def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        async with self.lock:
            return {
                "total_sessions": len(self.sessions),
                "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl_seconds,
                "session_ids": list(self.sessions.keys())
            }

# ================ MIDDLEWARE AND SECURITY ================


class RateLimiter:
    """Simple rate limiter with time-window tracking"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        async with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]
            
            # Check rate limit
            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False
            
            # Add current request
            self.requests[client_id].append(now)
            return True


class RequestValidator:
    """Validates incoming MCP requests"""
    
    @staticmethod
    def validate_mcp_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate basic MCP request structure"""
        if not isinstance(data, dict):
            return False, "Request must be a JSON object"
        
        if "method" not in data:
            return False, "Missing 'method' field"
        
        if "id" not in data:
            return False, "Missing 'id' field"
        
        return True, None
    
    @staticmethod
    def validate_tool_call(params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate tool call parameters"""
        if not isinstance(params, dict):
            return False, "Tool parameters must be a JSON object"
        
        if "name" not in params:
            return False, "Missing tool 'name'"
        
        if "arguments" not in params:
            return False, "Missing tool 'arguments'"
        
        tool_name = params["name"]
        
        # Get detailed schemas
        detailed_schemas = get_tool_schemas()
        
        if tool_name not in detailed_schemas:
            return False, f"Unknown tool: {tool_name}. Available tools: {sorted(list(detailed_schemas.keys()))}"
        
        return True, None


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for basic protection"""
    
    async def dispatch(self, request: Request, call_next):
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > config.max_request_size_mb * 1024 * 1024:
            return JSONResponse(
                status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                content={"error": "Request too large"}
            )
        
        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, input_rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = input_rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"
        
        if not await self.rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=HTTPStatus.TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded"}
            )
        
        return await call_next(request) 

# Global session manager
session_manager = None
rate_limiter = None


@dataclass
class RateLimitViolation:
    """Represents a rate limit violation with standardized error information"""
    tool_name: str
    limit_type: str  # "burst", "second", "minute", "hour"
    current_usage: int
    limit_value: float
    retry_after_seconds: float
    
    def to_user_friendly_message(self) -> str:
        """Generate user-friendly error message"""
        if self.limit_type == "burst":
            return f"Service temporarily unavailable: Too many rapid requests to {self.tool_name}. Please wait {self.retry_after_seconds:.0f} seconds before trying again."
        elif self.limit_type == "second":
            return f"Service temporarily unavailable: {self.tool_name} request rate exceeded ({self.limit_value}/second). Please wait {self.retry_after_seconds:.0f} seconds before trying again."
        elif self.limit_type == "minute":
            return f"Service temporarily unavailable: {self.tool_name} quota exceeded ({self.limit_value}/minute). Please try again in {self.retry_after_seconds:.0f} seconds."
        elif self.limit_type == "hour":
            return f"Service temporarily unavailable: {self.tool_name} hourly quota exceeded ({self.limit_value}/hour). Please try again in {self.retry_after_seconds:.0f} minutes."
        else:
            return f"Service temporarily unavailable: {self.tool_name} rate limit exceeded. Please try again later."
    
    def to_technical_message(self) -> str:
        """Generate technical error message for debugging"""
        return f"Tool '{self.tool_name}' {self.limit_type} limit exceeded ({self.current_usage}/{self.limit_value} {self.limit_type})"


def _parse_rate_limit_denial(tool_name: str, denial_reason: str) -> RateLimitViolation:
    """Parse rate limit denial reason into structured violation information"""
    import re
    
    # Default values
    limit_type = "unknown"
    current_usage = 0
    limit_value = 0.0
    retry_after_seconds = 60.0  # Default retry after 1 minute
    
    # Parse different types of rate limit violations
    if "burst limit exceeded" in denial_reason:
        limit_type = "burst"
        retry_after_seconds = 1.0  # Burst limits reset quickly
        match = re.search(r'\((\d+) requests/burst\)', denial_reason)
        if match:
            limit_value = float(match.group(1))
            current_usage = int(limit_value)  # Approximation
    
    elif "per-second limit exceeded" in denial_reason:
        limit_type = "second"
        retry_after_seconds = 1.0  # Wait 1 second
        match = re.search(r'\(([0-9.]+) requests/second\)', denial_reason)
        if match:
            limit_value = float(match.group(1))
            current_usage = int(limit_value)  # Approximation
    
    elif "per-minute limit exceeded" in denial_reason:
        limit_type = "minute"
        retry_after_seconds = 10.0  # Wait 10 seconds for minute limits
        match = re.search(r'\(([0-9.]+) requests/minute\)', denial_reason)
        if match:
            limit_value = float(match.group(1))
            current_usage = int(limit_value)  # Approximation
    
    elif "per-hour limit exceeded" in denial_reason:
        limit_type = "hour"
        retry_after_seconds = 300.0  # Wait 5 minutes for hour limits
        match = re.search(r'\(([0-9.]+) requests/hour\)', denial_reason)
        if match:
            limit_value = float(match.group(1))
            current_usage = int(limit_value)  # Approximation
    
    return RateLimitViolation(
        tool_name=tool_name,
        limit_type=limit_type,
        current_usage=current_usage,
        limit_value=limit_value,
        retry_after_seconds=retry_after_seconds
    )


async def _call_session_tool_async(session: Session, tool_name: str, tool_args: Dict[str, Any], 
                                   client_ip: str = "unknown") -> Dict[str, Any]:
    """Execute a tool within a session context with full tracking, workspace management, and global rate limiting"""
    
    start_time = time.time()
    success = False
    error_details = None
    result_data = None
    
    # Touch session at start of tool execution to prevent expiry during long operations
    session.touch()
    
    try:
        # CHECK GLOBAL TOOL RATE LIMITS FIRST
        if global_tool_rate_limiter:
            allowed, deny_reason = await global_tool_rate_limiter.is_allowed(tool_name)
            if not allowed:
                # Parse the denial reason to create structured rate limit violation
                rate_limit_violation = _parse_rate_limit_denial(tool_name, deny_reason)
                
                # Create user-friendly error message
                user_message = rate_limit_violation.to_user_friendly_message()
                technical_message = rate_limit_violation.to_technical_message()
                
                logger.warning(f"Session {session.id}: {technical_message}")
                
                result_data = {
                    "success": False,
                    "error": user_message,
                    "error_code": "RATE_LIMIT_EXCEEDED", 
                    "error_type": "rate_limit",
                    "tool_name": tool_name,
                    "limit_type": rate_limit_violation.limit_type,
                    "retry_after_seconds": rate_limit_violation.retry_after_seconds,
                    "data": None,
                    "rate_limited": True,  # Keep for backward compatibility
                    "technical_details": technical_message  # For debugging
                }
                
                # Still log this for tracking purposes
                duration_ms = (time.time() - start_time) * 1000
                tracker = session.get_tool_tracker()
                if tracker:
                    try:
                        agent_info = {
                            "client_ip": client_ip,
                            "type": "unknown",
                            "session_request_count": session.request_count
                        }
                        
                        tracker.log_tool_call(
                            tool_name=tool_name,
                            input_args=tool_args,
                            output_result=result_data,
                            success=False,
                            duration_ms=duration_ms,
                            error_details=user_message,
                            agent_info=agent_info
                        )
                    except Exception as e:
                        logger.error(f"Failed to log rate-limited tool call: {e}")
                
                return result_data
        
        # Get MCP tools instance for this session (handles workspace isolation)
        mcp_tools = session.get_mcp_tools(prefer_async=True)
        
        # Get tool method directly from the mcp_tools instance
        if not hasattr(mcp_tools, tool_name):
            raise ValueError(f"Tool '{tool_name}' not implemented")
        
        tool_method = getattr(mcp_tools, tool_name)
        
        # Add session context to tool arguments for workspace-aware tools
        if hasattr(mcp_tools, 'set_session_context'):
            mcp_tools.set_session_context(session.id, str(session.workspace_path))
        
        # Execute tool with keep-alive for potentially long operations
        logger.info(f"Session {session.id}: Executing tool '{tool_name}' with args: {list(tool_args.keys())}")
        
        # Use keep-alive wrapper for tools that might take a long time
        long_running_tools = {'batch_web_search', 'url_crawler', 'document_qa', 'document_extract', 'bash'}
        
        # Check if the tool method is async
        import inspect
        is_async_tool = inspect.iscoroutinefunction(tool_method)
        
        # Execute tool based on whether it's async or sync
        if is_async_tool:
            # Tool is async - execute directly
            logger.debug("Executing async tool '{%s}'", tool_name)
            
            if config.enable_session_keepalive and tool_name in long_running_tools:
                # For long-running async tools, use keep-alive
                with KeepAliveSessionWrapper(session, touch_interval=config.keepalive_touch_interval):
                    result = await tool_method(**tool_args)
            else:
                # For regular async tools, execute directly
                result = await tool_method(**tool_args)
        else:
            # Tool is sync - execute in thread pool
            logger.debug("Executing sync tool '{%s}' in thread pool", tool_name)
            
            # Define the synchronous tool execution function
            def execute_tool_sync():
                """Synchronous tool execution to be run in thread pool"""
                return tool_method(**tool_args)
            
            # Execute tool asynchronously in thread pool for true non-blocking execution
            import asyncio
            import concurrent.futures
            
            # Create a thread pool executor for CPU-bound/blocking operations
            loop = asyncio.get_event_loop()
            
            if config.enable_session_keepalive and tool_name in long_running_tools:
                # For long-running tools, use keep-alive with async execution
                with KeepAliveSessionWrapper(session, touch_interval=config.keepalive_touch_interval):
                    # Run in thread pool to avoid blocking the event loop
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        result = await loop.run_in_executor(executor, execute_tool_sync)
            else:
                # For regular tools, use async execution without keep-alive
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    result = await loop.run_in_executor(executor, execute_tool_sync)
        
        # Touch session after tool execution to update activity
        session.touch()
        
        # Handle different result formats
        if hasattr(result, 'to_dict'):
            result_data = result.to_dict()
        elif isinstance(result, dict):
            result_data = result
        else:
            result_data = {"result": result}
        
        success = result_data.get('success', True)
        
        if success:
            logger.info(f"Session {session.id}: Tool '{tool_name}' completed successfully")
            
            # RECORD SUCCESSFUL REQUEST FOR RATE LIMITING
            if global_tool_rate_limiter:
                await global_tool_rate_limiter.record_request(tool_name)
            

            
        else:
            error_details = result_data.get('error', 'Unknown error')
            logger.warning(f"Session {session.id}: Tool '{tool_name}' failed: {error_details}")
        
    except Exception as e:
        success = False
        error_details = str(e)
        result_data = {
            "success": False,
            "error": error_details,
            "data": None
        }
        logger.error(f"Session {session.id}: Tool '{tool_name}' exception: {e}")
    
    # Calculate execution time
    duration_ms = (time.time() - start_time) * 1000
    
    # Log tool call if tracking is enabled
    tracker = session.get_tool_tracker()
    if tracker:
        try:
            agent_info = {
                "client_ip": client_ip,
                "type": "unknown",  # Could be enhanced to detect agent type
                "session_request_count": session.request_count
            }
            
            tracker.log_tool_call(
                tool_name=tool_name,
                input_args=tool_args,
                output_result=result_data,
                success=success,
                duration_ms=duration_ms,
                error_details=error_details,
                agent_info=agent_info
            )
        except Exception as e:
            logger.error(f"Failed to log tool call: {e}")
    
    return result_data



def create_sse_response(response_data: dict, session_id: str = None) -> StreamingResponse:
    """Create Server-Sent Events response with proper formatting"""
    def generate_sse():
        try:
            # Add session info to response if available
            if session_id:
                response_data["session_id"] = session_id
            
            json_data = json.dumps(response_data, ensure_ascii=False)
            yield f"event: message\n"
            yield f"data: {json_data}\n"
            yield f"\n"
        except Exception as e:
            error_data = {
                "jsonrpc": "2.0",
                "error": {"code": JsonRpcErr.INTERNAL_ERROR, "message": f"Internal error: {str(e)}"},
                "id": response_data.get("id")
            }
            json_data = json.dumps(error_data, ensure_ascii=False)
            yield f"event: error\n"
            yield f"data: {json_data}\n"
            yield f"\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


def create_error_response(request_id: Any, code: int, message: str, session_id: str = None) -> StreamingResponse:
    """Create error response in SSE format"""
    error_data = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": request_id
    }
    return create_sse_response(error_data, session_id)


def create_rate_limit_response(
    request_id: Any,
    tool_name: str,
    error_message: str,
    retry_after_seconds: float,
    limit_type: str,
    technical_details: str = "",
    session_id: str = None
) -> JSONResponse:
    """
    Create HTTP 429 Rate Limit Exceeded response with proper headers and error format.
    
    Returns proper HTTP status code instead of SSE for rate limiting errors.
    """
    
    # Calculate retry-after header value
    retry_after_header = int(max(1.0, retry_after_seconds))
    
    # Create standardized error response
    error_data = {
        "error": {
            "type": "rate_limit_exceeded",
            "code": "RATE_LIMIT_EXCEEDED",
            "message": error_message,
            "details": {
                "tool_name": tool_name,
                "limit_type": limit_type,
                "retry_after_seconds": retry_after_seconds,
                "technical_details": technical_details
            }
        },
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id
    }
    
    # Set appropriate headers
    headers = {
        "Retry-After": str(retry_after_header),  # HTTP standard header
        "X-RateLimit-Limit-Type": limit_type,
        "X-RateLimit-Tool": tool_name,
        "X-RateLimit-Retry-After": str(retry_after_seconds),
        "Content-Type": "application/json"
    }
    
    return JSONResponse(
        status_code=HTTPStatus.TOO_MANY_REQUESTS,  # Too Many Requests
        content=error_data,
        headers=headers
    )


async def handle_mcp_request(request: Request) -> StreamingResponse:
    """Main MCP request handler with session management and tool execution"""
    
    try:
        # Check content length before reading body
        content_length = request.headers.get("content-length")
        if content_length:
            content_size_mb = int(content_length) / (1024 * 1024)
            if content_size_mb > config.max_request_size_mb:
                logger.warning(f"Request too large: {content_size_mb:.2f}MB > {config.max_request_size_mb}MB")
                return create_error_response(None, JsonRpcErr.PARSE_ERROR, f"Request too large: {content_size_mb:.2f}MB")
        
        # Parse request with timeout protection
        try:
            body = await asyncio.wait_for(request.body(), timeout=config.request_timeout_seconds)
        except asyncio.TimeoutError:
            logger.error("Timeout while reading request body")
            return create_error_response(None, JsonRpcErr.REQUEST_TIMEOUT, "Request body read timeout")
        
        if not body:
            return create_error_response(None, JsonRpcErr.PARSE_ERROR, "Empty request body")
        
        try:
            data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError as e:
            return create_error_response(None, JsonRpcErr.PARSE_ERROR, f"Invalid JSON: {str(e)}")
        
        # Validate MCP request structure
        is_valid, error_msg = RequestValidator.validate_mcp_request(data)
        if not is_valid:
            return create_error_response(data.get("id"), JsonRpcErr.INVALID_REQUEST, error_msg)
        
        request_id = data["id"]
        method = data["method"]
        params = data.get("params", {})
        
        # Get or create session
        session_id = request.headers.get("X-Session-ID")
        client_ip = request.client.host if request.client else "unknown"
        
        session = await session_manager.get_or_create_session(session_id)
        logger.info(f"Processing {method} request for session {session.id} from {client_ip}")
        
        # Handle different MCP methods
        if method == "initialize":
            # MCP initialization
            response_data = {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {
                        "tools": {"supportsProgress": True},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": "DeepDiver-Demo-MCP",
                        "version": "1.0.0"
                    }
                },
                "id": request_id
            }
            
        elif method == "tools/list":
            # List available tools using detailed schemas from get_tool_schemas()
            tools_list = []
            detailed_schemas = get_tool_schemas()
            
            # Build tools list from schemas
            for _, detailed_schema in detailed_schemas.items():
                tools_list.append({
                    "name": detailed_schema["name"],
                    "description": detailed_schema["description"],
                    "inputSchema": detailed_schema["inputSchema"]
                })
            
            logger.info(f"Serving {len(tools_list)} tools with detailed schemas to client")
            
            response_data = {
                "jsonrpc": "2.0",
                "result": {"tools": tools_list},
                "id": request_id
            }
            
        elif method == "tools/call":
            # Execute tool call
            is_valid, error_msg = RequestValidator.validate_tool_call(params)
            if not is_valid:
                return create_error_response(request_id, JsonRpcErr.INVALID_PARAMS, error_msg, session.id)
            
            tool_name = params["name"]
            tool_arguments = params["arguments"]
            
            # Execute tool in session context asynchronously
            result = await _call_session_tool_async(session, tool_name, tool_arguments, client_ip)
            
            # CHECK FOR RATE LIMITING AND RETURN PROPER HTTP STATUS
            if result.get("rate_limited", False):
                return create_rate_limit_response(
                    request_id=request_id,
                    tool_name=tool_name,
                    error_message=result.get("error", "Rate limit exceeded"),
                    retry_after_seconds=result.get("retry_after_seconds", 60),
                    limit_type=result.get("limit_type", "unknown"),
                    technical_details=result.get("technical_details", ""),
                    session_id=session.id
                )
            
            # Format normal response
            response_data = {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2, ensure_ascii=False)
                        }
                    ]
                },
                "id": request_id
            }
            
        else:
            return create_error_response(request_id, JsonRpcErr.METHOD_NOT_FOUND, f"Method not found: {method}", session.id)
        
        return create_sse_response(response_data, session.id)
        
    except asyncio.TimeoutError:
        logger.warning("Request timeout - client may have disconnected")
        return create_error_response(None, JsonRpcErr.REQUEST_TIMEOUT, "Request timeout")
    except Exception as e:
        # Handle client disconnects gracefully
        if "ClientDisconnect" in str(e) or "ConnectionClosedError" in str(e):
            logger.warning(f"Client disconnected during request processing: {e}")
            return create_error_response(None, JsonRpcErr.REQUEST_TIMEOUT, "Client disconnected")
        
        logger.error(f"Unexpected error in MCP request handler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return create_error_response(None, JsonRpcErr.INTERNAL_ERROR, f"Internal server error: {str(e)}")


async def handle_health_check(request: Request) -> JSONResponse:
    """Health check endpoint"""
    try:
        stats = await session_manager.get_stats() if session_manager else {}
        
        # Get rate limiting summary
        rate_limit_summary = {}
        if global_tool_rate_limiter:
            all_stats = global_tool_rate_limiter.get_all_stats()
            rate_limit_summary = {
                "enabled": True,
                "tools_with_limits": len(all_stats),
                "total_configured_tools": list(all_stats.keys())
            }
        else:
            rate_limit_summary = {"enabled": False}
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "session_stats": stats,
            "features": {
                "workspace_isolation": True,
                "tool_call_tracking": config.enable_tool_tracking if config else False,
                "client_rate_limiting": True,
                "global_tool_rate_limiting": rate_limit_summary["enabled"],
                "security_middleware": True,
                "standardized_rate_limit_responses": True
            },
            "rate_limiting": rate_limit_summary,
            "error_formats": {
                "rate_limit_exceeded": {
                    "http_status": HTTPStatus.TOO_MANY_REQUESTS,
                    "headers": ["Retry-After", "X-RateLimit-*"],
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "response_format": "application/json"
                }
            }
        }
        
        return JSONResponse(content=health_data)
        
    except Exception as e:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"status": "unhealthy", "error": str(e)}
        )


async def handle_tracking_info(request: Request) -> JSONResponse:
    """Get tool call tracking information for a session"""
    try:
        session_id = request.query_params.get("session_id")
        if not session_id:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"error": "session_id parameter required"}
            )
        
        session = await session_manager.get_session(session_id)
        if not session:
            return JSONResponse(
                status_code=HTTPStatus.NOT_FOUND,
                content={"error": f"Session {session_id} not found"}
            )
        
        tracker = session.get_tool_tracker()
        if not tracker:
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "tracking_enabled": False,
                    "message": "Tool call tracking not enabled or no workspace"
                }
            )
        
        # Read session summary
        summary_data = {}
        if tracker.summary_file.exists():
            try:
                with open(tracker.summary_file, 'r') as f:
                    summary_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read session summary: {e}")
        
        return JSONResponse(content={
            "session_id": session_id,
            "tracking_enabled": True,
            "summary": summary_data,
            "logs_directory": str(tracker.logs_dir),
            "current_log_file": str(tracker.current_log_file)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )



async def handle_rate_limit_stats(request: Request) -> JSONResponse:
    """Get global tool rate limiting statistics"""
    try:
        if not global_tool_rate_limiter:
            return JSONResponse(
                status_code=HTTPStatus.NOT_FOUND,
                content={"error": "Global tool rate limiter not initialized"}
            )
        
        # Check if specific tool requested
        tool_name = request.query_params.get("tool")
        
        if tool_name:
            # Get stats for specific tool
            stats = await global_tool_rate_limiter.get_tool_stats(tool_name)
            return JSONResponse(content=stats)
        else:
            # Get stats for all tools
            all_stats = global_tool_rate_limiter.get_all_stats()
            return JSONResponse(content={
                "timestamp": datetime.now().isoformat(),
                "global_tool_rate_limiting": True,
                "tools": all_stats,
                "summary": {
                    "total_tools_with_limits": len(all_stats),
                    "tools_configured": list(all_stats.keys())
                }
            })
        
    except Exception as e:
        logger.error(f"Failed to get rate limit stats: {e}")
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )


def create_app() -> Starlette:
    """Create and configure the Starlette application"""
    global session_manager, rate_limiter, global_tool_rate_limiter
    
    if not config:
        raise RuntimeError("Server configuration not initialized")
    
    # Initialize global components
    session_manager = ThreadSafeSessionManager(
        ttl_seconds=config.session_ttl_seconds,
        max_sessions=config.max_sessions,
        base_workspace_dir=config.base_workspace_dir
    )
    rate_limiter = RateLimiter(config.rate_limit_requests_per_minute)
    
    # Initialize global tool rate limiter
    if config.tool_rate_limits:
        global_tool_rate_limiter = GlobalToolRateLimiter(config.tool_rate_limits)
        logger.info(f"Initialized global tool rate limiter with {len(config.tool_rate_limits)} tool limits")
    else:
        logger.info("No tool rate limits configured - tools will run without global rate limiting")
    
    # Create app
    app = Starlette(debug=config.debug_mode)
    
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(RateLimitMiddleware, input_rate_limiter=rate_limiter)
    
    # Add routes
    app.add_route("/mcp", handle_mcp_request, methods=["POST"])
    app.add_route("/health", handle_health_check, methods=["GET"])
    app.add_route("/tracking", handle_tracking_info, methods=["GET"])
    app.add_route("/rate-limits", handle_rate_limit_stats, methods=["GET"])
    
    return app


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Demo-Ready MCP Server with Per-Tool Rate Limiting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/tools/mcp_server_standard.py --config src/tools/server_config.yaml
  python src/tools/mcp_server_standard.py --host 127.0.0.1 --port 8080
  python src/tools/mcp_server_standard.py --config custom_config.yaml --debug
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        help='Server host (overrides config file)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        help='Server port (overrides config file)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (overrides config file)'
    )
    
    parser.add_argument(
        '--workspace-dir',
        type=str,
        help='Base workspace directory (overrides config file)'
    )
    
    return parser.parse_args()


def print_startup_info():
    """Print server startup information"""
    logger.info("🚀 DeepDiver Demo MCP Server")
    logger.info("=" * 50)
    logger.info(f"📊 Features:")
    logger.info(f"  • Session Management: ✅ (TTL: {config.session_ttl_seconds}s)")
    logger.info(f"  • Workspace Isolation: ✅ (Base: {config.base_workspace_dir})")
    logger.info(f"  • Tool Call Tracking: {'✅' if config.enable_tool_tracking else '❌'}")
    logger.info(f"  • Client Rate Limiting: ✅ ({config.rate_limit_requests_per_minute}/min)")
    logger.info(f"  • Global Tool Rate Limiting: {'✅' if config.tool_rate_limits else '❌'}")
    logger.info(f"  • Security Middleware: ✅")
    
    # Tool rate limiting information
    if config.tool_rate_limits:
        logger.info(f"🚦 Tool Rate Limits: {len(config.tool_rate_limits)} tools configured")
        for tool_name, limits in list(config.tool_rate_limits.items())[:3]:
            burst = limits.get('burst_limit', '∞')
            rpm = limits.get('requests_per_minute', '∞')
            logger.info(f"  • {tool_name}: {rpm}/min, burst: {burst}")
        if len(config.tool_rate_limits) > 3:
            logger.info(f"  • ... and {len(config.tool_rate_limits) - 3} more tools")
    
    # Tool information from schemas
    tool_schemas = get_tool_schemas()
    available_tools = list(tool_schemas.keys())
    
    logger.info(f"🔧 Tools Available: {len(available_tools)}")
    logger.info(f"  • All tools defined in schemas: {len(available_tools)} tools")
    logger.info(f"  • Sample tools: {', '.join(sorted(available_tools)[:5])}...")
    logger.info("=" * 50)


def main():
    """Main function to run the production MCP server"""
    global config
    
    # Parse command line arguments
    args = parse_arguments()

    config = ServerConfig.from_yaml("./src/tools/server_config.yaml")
    
    # Apply CLI overrides
    if args.host:
        config.host = args.host
        logger.info(f"🔧 Override: Host = {config.host}")
    
    if args.port:
        config.port = args.port
        logger.info(f"🔧 Override: Port = {config.port}")
    
    if args.debug:
        config.debug_mode = True
        logger.info(f"🔧 Override: Debug mode enabled")
    
    if args.workspace_dir:
        config.base_workspace_dir = args.workspace_dir
        logger.info(f"🔧 Override: Workspace directory = {config.base_workspace_dir}")
    
    print_startup_info()
    
    try:
        import os
        
        # Calculate optimal worker count for high-concurrency FIRST
        # Use CPU core count indirectly via uvicorn's defaults; no local variable needed
        
        # Override for high-concurrency scenarios
        if os.getenv('FORCE_HIGH_CONCURRENCY', '').lower() == 'true':
            pass  # Configuration handled elsewhere if needed
        
        app = create_app()
        
        logger.info(f"🌐 Starting server at http://{config.host}:{config.port}")
        logger.info(f"📡 MCP endpoint: http://{config.host}:{config.port}/mcp")
        logger.info(f"🏥 Health check: http://{config.host}:{config.port}/health")
        logger.info(f"📊 Tracking info: http://{config.host}:{config.port}/tracking?session_id=<id>")
        logger.info(f"🚦 Rate limit stats: http://{config.host}:{config.port}/rate-limits")
        
        uvicorn.run(
            app,  # Use app instance directly for single worker with async optimizations
            host=config.host,
            port=config.port,
            log_level="info",
            timeout_keep_alive=config.request_timeout_seconds,
            workers=1,  # Single worker with async optimizations
            backlog=1024,  # Larger backlog for high-concurrency
            access_log=False,  # Disable access logs for better performance
            limit_concurrency=None,  # No artificial concurrency limit
        )
        
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 