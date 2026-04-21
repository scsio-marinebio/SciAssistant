# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#!/usr/bin/env python3
"""
MCP Client for Agent-to-Server Communication
Provides a proper MCP client that uses the official MCP package
to connect to and communicate with MCP servers through the Model Context Protocol.
"""

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from ..utils.status_codes import JsonRpcErr
from http import HTTPStatus

try:
    import httpx
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("HTTP client dependencies not available. Falling back to direct tools.")

logger = logging.getLogger(__name__)


@dataclass
class MCPClientResult:
    """Standard result format for MCP client operations"""
    success: bool
    data: Any = None
    error: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class MCPTool:
    """Simple representation of an MCP tool"""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RetryConfig:
    """Configuration for retry behavior on rate limiting"""
    max_retries: int = 20              # Maximum number of retry attempts
    base_delay: float = 2.0            # Base delay between retries (seconds)
    max_delay: float = 60.0            # Maximum delay between retries (seconds)
    exponential_backoff: bool = True   # Use exponential backoff
    respect_retry_after: bool = True   # Respect server's Retry-After header
    retry_on_rate_limit: bool = True   # Enable automatic retry on rate limits


class MCPClient:
    """
    Simple HTTP-based MCP Client for dynamic tool discovery and execution.
    
    This client makes direct HTTP JSON-RPC calls to the MCP server,
    avoiding the complexity of streaming connections.
    
    Session management is handled entirely by the server:
    - Server assigns session IDs on connection
    - Server manages workspace creation and isolation
    - All tool operations use server-managed workspaces
    """
    
    def __init__(self, server_url: str = "http://localhost:6274/mcp", retry_config: Optional[RetryConfig] = None):
        self.server_url = server_url.rstrip('/')
        self.retry_config = retry_config or RetryConfig()
        self._tools: Dict[str, MCPTool] = {}
        self._connected = False
        self._request_id = 0
        self._session_id = None
        
        if not MCP_AVAILABLE:
            logger.warning("HTTP client not available. Some functionality may be limited.")
            return
        
        # Initialize connection and discover tools
        self._initialize_connection()
    
    def _get_next_id(self) -> int:
        """Get next request ID"""
        self._request_id += 1
        return self._request_id

    @staticmethod
    def _parse_sse_response(sse_text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response and extract JSON data"""
        try:
            # SSE format: "event: message\ndata: {json}\n\n"
            lines = sse_text.strip().split('\n')
            
            for line in lines:
                if line.startswith('data: '):
                    json_data = line[6:]  # Remove "data: " prefix
                    return json.loads(json_data)
            
            # If no data line found, try parsing entire response as JSON
            return json.loads(sse_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SSE response: {e}")
            logger.error(f"SSE text: {sse_text[:200]}...")
            return {"error": {"code": JsonRpcErr.PARSE_ERROR, "message": f"Parse error: {e}"}}
    
    def _make_request(self, method: str, params: Dict[str, Any] = None) -> MCPClientResult:
        """Make a JSON-RPC request to the MCP server with automatic retry on rate limits"""
        if not MCP_AVAILABLE:
            return MCPClientResult(success=False, error="HTTP client not available")
        
        # Prepare JSON-RPC request
        request_data = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
            "params": params or {}
        }
        
        # Make HTTP request with proper MCP headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Add session ID if available
        if self._session_id:
            headers["X-Session-ID"] = self._session_id
        
        last_error = None
        retry_count = 0
        
        while retry_count <= self.retry_config.max_retries:
            try:
                # Disable proxy for localhost/127.0.0.1 connections to avoid proxy interference
                import os
                from urllib.parse import urlparse
                parsed_url = urlparse(self.server_url)
                is_localhost = parsed_url.hostname in ['localhost', '127.0.0.1', '::1']
                
                # Add localhost to NO_PROXY for localhost connections
                original_no_proxy = None
                if is_localhost:
                    original_no_proxy = os.environ.get('NO_PROXY', os.environ.get('no_proxy', ''))
                    # Add localhost and 127.0.0.1 to NO_PROXY
                    no_proxy_hosts = ['localhost', '127.0.0.1', '::1']
                    if original_no_proxy:
                        existing_hosts = [h.strip() for h in original_no_proxy.split(',')]
                        no_proxy_hosts.extend(existing_hosts)
                    os.environ['NO_PROXY'] = ','.join(no_proxy_hosts)
                    os.environ['no_proxy'] = ','.join(no_proxy_hosts)
                
                try:
                    # Create client with connection pooling for high-concurrency
                    limits = httpx.Limits(
                        max_keepalive_connections=3000,  # Keep more connections alive
                        max_connections=3000,           # Allow more concurrent connections
                        keepalive_expiry=1000.0         # Keep connections alive longer
                    )
                    timeout = httpx.Timeout(
                        connect=100.0,
                        read=None,
                        write=60.0,
                        pool=30.0
                    )
                    with httpx.Client(
                        timeout=timeout,  # Higher timeout for high-concurrency scenarios
                        limits=limits,   # Connection pooling for better performance
                        trust_env=False,
                        http2=True      # Enable HTTP/2 for better multiplexing
                    ) as client:
                        response = client.post(
                            self.server_url,
                            json=request_data,
                            headers=headers
                        )

                    # Check for rate limiting (HTTP 429)
                    if response.status_code == 429:
                        if not self.retry_config.retry_on_rate_limit:
                            return MCPClientResult(
                                success=False,
                                error=f"Rate limit exceeded (HTTP 429) - retries disabled",
                                metadata={"status_code": 429, "retry_count": retry_count}
                            )
                        
                        if retry_count >= self.retry_config.max_retries:
                            return MCPClientResult(
                                success=False,
                                error=f"Rate limit exceeded (HTTP 429) - max retries ({self.retry_config.max_retries}) reached",
                                metadata={"status_code": 429, "retry_count": retry_count}
                            )
                        
                        # Calculate retry delay
                        delay = self._calculate_retry_delay(response, retry_count)
                        
                        logger.warning(f"Rate limit exceeded for {method} (attempt {retry_count + 1}/{self.retry_config.max_retries + 1}). Retrying in {delay:.1f}s...")
                        
                        # Wait before retry
                        time.sleep(delay)
                        retry_count += 1
                        continue
                    
                    # Handle other HTTP errors
                    if response.status_code != HTTPStatus.OK:
                        return MCPClientResult(
                            success=False,
                            error=f"HTTP {response.status_code}: {response.text}",
                            metadata={"status_code": response.status_code, "retry_count": retry_count}
                        )
                    
                    # Parse successful response (could be JSON or SSE format)
                    if response.headers.get("content-type", "").startswith("text/event-stream"):
                        # Parse SSE format
                        response_data = self._parse_sse_response(response.text)
                    else:
                        # Parse regular JSON
                        response_data = response.json()
                    
                    if "error" in response_data:
                        return MCPClientResult(
                            success=False,
                            error=f"MCP Error: {response_data['error']}",
                            metadata={"retry_count": retry_count}
                        )
                    
                    # Capture session ID from response data (for all methods, not just initialize)
                    if "session_id" in response_data:
                        self._session_id = response_data["session_id"]
                        logger.info(f"Captured session ID from response: {self._session_id}")
                    
                    # Success! Log retry info if this wasn't the first attempt
                    if retry_count > 0:
                        logger.info(f"Request {method} succeeded after {retry_count} retries")
                    
                    return MCPClientResult(
                        success=True,
                        data=response_data.get("result"),
                        metadata={
                            "method": method, 
                            "server_url": self.server_url,
                            "session_id": self._session_id,
                            "retry_count": retry_count
                        }
                    )
                finally:
                    # Restore original NO_PROXY environment variable
                    if is_localhost:
                        if original_no_proxy is not None:
                            if original_no_proxy:
                                os.environ['NO_PROXY'] = original_no_proxy
                                os.environ['no_proxy'] = original_no_proxy
                            else:
                                # Remove NO_PROXY if it wasn't set originally
                                os.environ.pop('NO_PROXY', None)
                                os.environ.pop('no_proxy', None)
                                   
            except Exception as e:
                last_error = str(e)
                logger.error(f"MCP request failed for {method} (attempt {retry_count + 1}): {e}")
                
                # Only retry on certain exceptions (network issues, timeouts)
                if not self._should_retry_exception(e) or retry_count >= self.retry_config.max_retries:
                    break
                
                # Calculate retry delay for exceptions
                delay = self._calculate_exception_retry_delay(retry_count)
                logger.warning(f"Request {method} failed, retrying in {delay:.1f}s... (attempt {retry_count + 1}/{self.retry_config.max_retries + 1})")
                
                time.sleep(delay)
                retry_count += 1
        
        # All retries exhausted
        return MCPClientResult(
            success=False,
            error=f"Request failed after {retry_count} retries. Last error: {last_error}",
            metadata={"retry_count": retry_count}
        )
    
    def _calculate_retry_delay(self, response, retry_count: int) -> float:
        """Calculate delay before retry based on server response and retry count"""
        delay = self.retry_config.base_delay
        
        # Respect server's Retry-After header if available
        if self.retry_config.respect_retry_after and "Retry-After" in response.headers:
            try:
                retry_after = float(response.headers["Retry-After"])
                delay = min(retry_after, self.retry_config.max_delay)
                logger.debug("Using server Retry-After: {%s}s", delay)
            except (ValueError, TypeError):
                logger.warning(f"Invalid Retry-After header: {response.headers.get('Retry-After')}")
        
        # Apply exponential backoff if enabled
        elif self.retry_config.exponential_backoff:
            delay = min(
                self.retry_config.base_delay * (2 ** retry_count),
                self.retry_config.max_delay
            )
        
        return delay
    
    def _calculate_exception_retry_delay(self, retry_count: int) -> float:
        """Calculate delay for exception-based retries"""
        if self.retry_config.exponential_backoff:
            return min(
                self.retry_config.base_delay * (2 ** retry_count),
                self.retry_config.max_delay
            )
        return self.retry_config.base_delay

    @staticmethod
    def _should_retry_exception(exception: Exception) -> bool:
        """Determine if an exception warrants a retry"""
        # Retry on network-related exceptions
        if isinstance(exception, (httpx.RequestError, httpx.TimeoutException, httpx.ConnectError)):
            return True
        
        # Don't retry on other exceptions (parsing errors, etc.)
        return False
    
    def _initialize_connection(self):
        """Initialize MCP client connection and fetch available tools"""
        if not MCP_AVAILABLE:
            return
        
        try:
            # Check if session ID is already set via environment variable
            import os
            env_session_id = os.environ.get('AGENT_SESSION_ID')
            if env_session_id:
                self._session_id = env_session_id
                logger.info(f"Using existing session ID from environment: {self._session_id}")
            
            # Initialize session (server will use existing session if X-Session-ID header is provided)
            init_result = self._make_request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "DeepDiver-MCP-Client",
                    "version": "1.0.0"
                }
            })
            print(init_result)
            if not init_result.success:
                logger.error(f"MCP initialization failed: {init_result.error}")
                return
            
            logger.info("MCP client initialized successfully")
            
            # Fetch available tools
            tools_result = self._make_request("tools/list")
            
            if tools_result.success and tools_result.data:
                tools_data = tools_result.data.get("tools", [])
                self._tools = {}
                
                for tool_data in tools_data:
                    tool = MCPTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {})
                    )
                    self._tools[tool.name] = tool
                
                logger.info(f"Discovered {len(self._tools)} tools from MCP server: {list(self._tools.keys())}")
            
            self._connected = True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            self._connected = False
    
    def _ensure_connection(self):
        """Ensure MCP client is connected"""
        if not MCP_AVAILABLE:
            raise RuntimeError("HTTP client not available")
        
        if not self._connected:
            self._initialize_connection()
        
        if not self._connected:
            raise RuntimeError("MCP client not connected to server")
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPClientResult:
        """
        Generic method to call any tool available on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            MCPClientResult with the tool execution result
        """
        try:
            self._ensure_connection()
            
            if tool_name not in self._tools:
                return MCPClientResult(
                    success=False,
                    error=f"Tool '{tool_name}' not available on server. Available tools: {list(self._tools.keys())}"
                )
            
            # Check if search tool is allowed based on user preferences
            search_filter_result = self._check_search_tool_allowed(tool_name)
            if not search_filter_result["allowed"]:
                logger.info(f"Tool '{tool_name}' blocked by search source filter: {search_filter_result['reason']}")
                return MCPClientResult(
                    success=False,
                    error=search_filter_result["reason"],
                    data={"disabled_by_user": True, "tool_name": tool_name}
                )
            
            # Call the tool via JSON-RPC
            result = self._make_request("tools/call", {
                "name": tool_name,
                "arguments": arguments
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}")
            return MCPClientResult(
                success=False,
                error=str(e)
            )
    
    def get_available_tools(self) -> Dict[str, MCPTool]:
        """Get dictionary of available tools from the server (filtered by search source preferences)"""
        all_tools = self._tools.copy()
        # Filter out disabled search tools
        filtered_tools = {}
        for name, tool in all_tools.items():
            if self._check_search_tool_allowed(name)["allowed"]:
                filtered_tools[name] = tool
        return filtered_tools
    
    def list_tools(self) -> List[str]:
        """Get list of available tool names (filtered by search source preferences)"""
        all_tools = list(self._tools.keys())
        # Filter out disabled search tools
        return [name for name in all_tools if self._check_search_tool_allowed(name)["allowed"]]
    
    def get_tool_info(self, tool_name: str) -> Optional[MCPTool]:
        """Get detailed information about a specific tool"""
        return self._tools.get(tool_name)
    
    def is_connected(self) -> bool:
        """Check if client is connected to MCP server"""
        return self._connected and MCP_AVAILABLE
    
    def _check_search_tool_allowed(self, tool_name: str) -> Dict[str, Any]:
        """
        Check if a search tool is allowed based on user's search source preferences.
        Returns dict with 'allowed' (bool) and 'reason' (str) keys.
        
        Configuration-driven approach: Add new search sources by:
        1. Adding to SEARCH_SOURCE_CONFIG below
        2. Setting corresponding environment variable in backend
        3. Adding UI checkbox in frontend
        """
        # Configuration: Define search sources and their associated tools
        SEARCH_SOURCE_CONFIG = {
            'WEBSEARCH': {
                'name': '全网搜索',
                'env_var': 'SEARCH_SOURCE_WEBSEARCH',
                'tools': ['batch_web_search', 'url_crawler']
            },
            'PUBMED': {
                'name': 'PubMed',
                'env_var': 'SEARCH_SOURCE_PUBMED',
                'tools': ['search_pubmed_key_words', 'search_pubmed_advanced', 
                         'medrxiv_search', 'get_pubmed_article', 'medrxiv_read_paper']
            },
            'ARXIV': {
                'name': 'arXiv',
                'env_var': 'SEARCH_SOURCE_ARXIV',
                'tools': ['arxiv_search', 'arxiv_read_paper']
            },
            'GOOGLE_SCHOLAR': {
                'name': 'Google Scholar',
                'env_var': 'SEARCH_SOURCE_GOOGLE_SCHOLAR',
                'tools': ['google_scholar_search', 'advanced_google_scholar_search', 'google_scholar_get_paper', 'url_crawler']
            },
            'SPRINGER': {
                'name': 'Springer Nature',
                'env_var': 'SEARCH_SOURCE_SPRINGER',
                'tools': ['springer_search', 'springer_get_article']
            },
        }
        
        # Build tool to source mapping from configuration
        # A tool can belong to multiple sources; it's allowed if ANY source is enabled
        tool_to_sources = {}
        for source_key, config in SEARCH_SOURCE_CONFIG.items():
            is_enabled = os.environ.get(config['env_var'], 'True').lower() == 'true'
            for tool in config['tools']:
                if tool not in tool_to_sources:
                    tool_to_sources[tool] = []
                tool_to_sources[tool].append((config['name'], is_enabled))
        
        # Check if tool is a search tool and if it's disabled
        if tool_name in tool_to_sources:
            sources = tool_to_sources[tool_name]
            # Tool is allowed if ANY of its associated sources is enabled
            if any(is_enabled for _, is_enabled in sources):
                return {"allowed": True, "reason": ""}
            source_names = ', '.join(name for name, _ in sources)
            return {
                "allowed": False,
                "reason": f"搜索源 '{source_names}' 未被用户启用。请使用已启用的搜索源，或要求用户在前端界面勾选该搜索源。当前工具 '{tool_name}' 已被禁用。"
            }
        
        # Tool is allowed (either not a search tool, or is enabled)
        return {"allowed": True, "reason": ""}
    
    def refresh_tools(self):
        """Refresh the list of available tools from the server"""
        try:
            # Fetch available tools
            tools_result = self._make_request("tools/list")
            
            if tools_result.success and tools_result.data:
                tools_data = tools_result.data.get("tools", [])
                self._tools = {}
                print(self._tools)

                for tool_data in tools_data:
                    tool = MCPTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {})
                    )
                    self._tools[tool.name] = tool
                
                logger.info(f"Refreshed {len(self._tools)} tools from MCP server")
            else:
                logger.error(f"Failed to refresh tools: {tools_result.error}")
                
        except Exception as e:
            logger.error(f"Error refreshing tools: {e}")
    
    def close(self):
        """Close MCP client connection"""
        # Since we create connections per request, just mark as disconnected
        self._connected = False


class MCPToolsAdapter:
    """
    Adapter class that provides the MCPTools interface while using the generic MCP client.
    
    This adapter provides backward compatibility with existing agents by mapping
    MCPTools method calls to generic MCP client tool calls.
    """
    
    def __init__(self, server_url: str = "http://localhost:6274/mcp", retry_config: Optional[RetryConfig] = None):
        self.client = MCPClient(server_url, retry_config)
    
    def _call_tool(self, tool_name: str, **kwargs) -> MCPClientResult:
        """Internal method to call tools through the MCP client"""
        return self.client.call_tool(tool_name, kwargs)
    
    def __getattr__(self, name: str):
        """
        Dynamic method creation for any tool available on the server.
        This allows calling tools like adapter.batch_web_search(...) or adapter.file_read(...)
        """
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Create a dynamic method that calls the tool
        def tool_method(**kwargs):
            result = self._call_tool(name, **kwargs)
            # For backward compatibility, return the data portion
            return result.data if result.success else {"error": result.error}
        
        return tool_method
    

    
    def is_connected(self) -> bool:
        """Check if the MCP client is connected to the server."""
        return self.client.is_connected()
    
    def get_available_tools(self) -> Dict[str, MCPTool]:
        """Get available tools from the MCP server."""
        return self.client.get_available_tools()
    
    def list_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.client.list_tools()
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get tool schemas for all available tools.
        This is the proper MCP way - schemas come from server, not direct imports.
        """
        schemas = []
        available_tools = self.get_available_tools()
        
        for tool_name, tool_info in available_tools.items():
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info.description,
                    "parameters": tool_info.input_schema
                }
            }
            schemas.append(schema)
        
        return schemas
    
    def refresh_tools(self):
        """Refresh the list of available tools from the server."""
        self.client.refresh_tools()
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get session information from the underlying MCP client."""
        try:
            if hasattr(self.client, '_session_id'):
                return {
                    "session_id": self.client._session_id,
                    "connected": self.client.is_connected(),
                    "server_url": getattr(self.client, 'server_url', 'unknown')
                }
            return None
        except Exception:
            return None
    
    def close(self):
        """Close the MCP client connection."""
        self.client.close()


class FilteredMCPToolsAdapter:
    """
    Filtered adapter that shares MCP client connection but restricts tool access per agent type.
    
    This allows agents to:
    - Share the same session/workspace (via shared client)
    - Have different tool sets appropriate for their role
    - Maintain proper separation of concerns
    """
    
    def __init__(self, shared_client: MCPClient, allowed_tools: List[str]):
        """
        Initialize with shared client and allowed tools list
        
        Args:
            shared_client: Shared MCPClient instance (same session)
            allowed_tools: List of tools this agent can access
        """
        self.client = shared_client
        self.allowed_tools = set(allowed_tools)
        
        # Validate that allowed tools exist on server
        available_tools = set(self.client.list_tools())
        invalid_tools = self.allowed_tools - available_tools
        if invalid_tools:
            logger.warning(f"Requested tools not available on server: {invalid_tools}")
            self.allowed_tools = self.allowed_tools & available_tools
    
    def _call_tool(self, tool_name: str, **kwargs) -> MCPClientResult:
        """Call tool if allowed, otherwise return error"""
        if tool_name not in self.allowed_tools:
            return MCPClientResult(
                success=False,
                error=f"Tool '{tool_name}' not allowed for this agent. Allowed tools: {list(self.allowed_tools)}"
            )
        
        # Remove any workspace_path if accidentally passed - server handles workspace
        kwargs.pop('workspace_path', None)
        return self.client.call_tool(tool_name, kwargs)
    
    def __getattr__(self, name: str):
        """
        Dynamic method resolution with tool filtering.
        
        Only allows access to tools in the allowed_tools list.
        """
        if name in self.allowed_tools:
            def tool_method(**kwargs):
                return self._call_tool(name, **kwargs)
            return tool_method
        
        if name in self.client.list_tools():
            # Tool exists but not allowed for this agent
            raise AttributeError(f"Tool '{name}' not allowed for this agent. Allowed tools: {list(self.allowed_tools)}")
        else:
            # Tool doesn't exist on server
            raise AttributeError(f"Tool '{name}' not available on server. Available tools: {self.client.list_tools()}")
    

    
    # ================ CLIENT MANAGEMENT ================


    def is_connected(self) -> bool:
        """Check if client is connected to MCP server"""
        return self.client.is_connected()
    
    def get_available_tools(self) -> Dict[str, MCPTool]:
        """Get filtered list of available tools for this agent"""
        all_tools = self.client.get_available_tools()
        return {name: tool for name, tool in all_tools.items() if name in self.allowed_tools}
    
    def list_tools(self) -> List[str]:
        """Get list of allowed tool names for this agent"""
        return list(self.allowed_tools)
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get tool schemas for tools allowed for this agent.
        This is the proper MCP way - schemas come from server, not direct imports.
        """
        schemas = []
        available_tools = self.get_available_tools()
        
        for tool_name, tool_info in available_tools.items():
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info.description,
                    "parameters": tool_info.input_schema
                }
            }
            schemas.append(schema)
        
        return schemas
    
    def refresh_tools(self):
        """Refresh the underlying client's tools"""
        self.client.refresh_tools()
        
        # Re-validate allowed tools after refresh
        available_tools = set(self.client.list_tools())
        invalid_tools = self.allowed_tools - available_tools
        if invalid_tools:
            logger.warning(f"Some allowed tools no longer available after refresh: {invalid_tools}")
            self.allowed_tools = self.allowed_tools & available_tools
    
    def close(self):
        """Close MCP client connection"""
        self.client.close()


# ================ AGENT TOOL SETS ================
# Define what tools each agent type should have access to

PLANNER_AGENT_TOOLS = [
    "download_files",
    "document_qa",

    "file_read",
    "file_write",
    "str_replace_based_edit_tool",

    "list_workspace",
    "file_find_by_name",
]


INFORMATION_SEEKER_TOOLS = [
    "batch_web_search",
    "url_crawler",
    "document_extract",
    "document_qa",
    "download_files",
    "file_read",
    "file_write",
    "str_replace_based_edit_tool",
    "list_workspace",
    "file_find_by_name",
    
    # Academic search tools - PubMed
    "search_pubmed_key_words",
    "search_pubmed_advanced",
    "get_pubmed_article",
    
    # Academic search tools - arXiv
    "arxiv_search",
    "arxiv_read_paper",
    
    # Academic search tools - medRxiv
    "medrxiv_search",
    "medrxiv_read_paper",
    
    # Academic search tools - Google Scholar
    "google_scholar_search",
    "advanced_google_scholar_search",
    "google_scholar_get_paper",
    
    # Academic search tools - Springer Nature
    "springer_search",
    "springer_get_article",
]

WRITER_AGENT_TOOLS = [
    "file_read",
    "list_workspace",
    "file_find_by_name",

    "search_result_classifier",
    "section_writer",
    "concat_section_files",
]


def create_filtered_mcp_tools_adapter(
    shared_client: MCPClient, 
    agent_type: str
) -> FilteredMCPToolsAdapter:
    """
    Create a filtered MCP tools adapter for specific agent type
    
    Args:
        shared_client: Shared MCPClient instance 
        agent_type: Type of agent ("planner", "information_seeker", "writer")
        
    Returns:
        FilteredMCPToolsAdapter with appropriate tools for agent type
    """
    tool_sets = {
        "planner": PLANNER_AGENT_TOOLS,
        "information_seeker": INFORMATION_SEEKER_TOOLS, 
        "writer": WRITER_AGENT_TOOLS
    }
    
    allowed_tools = tool_sets.get(agent_type, PLANNER_AGENT_TOOLS)
    
    return FilteredMCPToolsAdapter(
        shared_client=shared_client,
        allowed_tools=allowed_tools
    )


def create_agent_mcp_tools(
    agent_type: str,
    server_url: str = "http://localhost:6274/mcp",
    retry_config: Optional[RetryConfig] = None
) -> FilteredMCPToolsAdapter:
    """
    Convenience factory to create a filtered MCP tools adapter with retry support.
    This is the RECOMMENDED way to create MCP tools for agents.
    
    Args:
        agent_type: Type of agent ("planner", "information_seeker", "writer")
        server_url: URL of the MCP server (default: http://localhost:6274/mcp)
        retry_config: Optional retry configuration for handling rate limits
        
    Returns:
        FilteredMCPToolsAdapter with appropriate tools and retry support for the agent type
    """
    # Create client with retry support
    client = create_mcp_client(server_url=server_url, retry_config=retry_config)
    
    # Create filtered adapter for the agent type
    return create_filtered_mcp_tools_adapter(client, agent_type)


def create_mcp_client(
    server_url: str = "http://localhost:6274/mcp",
    retry_config: Optional[RetryConfig] = None
) -> MCPClient:
    """
    Factory function to create a generic MCP Client with optional retry configuration
    
    Args:
        server_url: URL of the MCP server (default: http://localhost:6274/mcp)
        retry_config: Optional retry configuration for handling rate limits
    
    Returns:
        MCPClient instance for direct tool calling with automatic retry on rate limits
    """
    return MCPClient(server_url=server_url, retry_config=retry_config)


def create_mcp_tools_adapter(
    server_url: str = "http://localhost:6274/mcp",
    retry_config: Optional[RetryConfig] = None
) -> MCPToolsAdapter:
    """
    Factory function to create an MCP Tools Adapter for backward compatibility with retry support.
    
    Args:
        server_url: URL of the MCP server (default: http://localhost:6274/mcp)
        retry_config: Optional retry configuration for handling rate limits
    
    Returns:
        MCPToolsAdapter instance that behaves like MCPTools but uses MCP client with automatic retries
    """
    return MCPToolsAdapter(server_url=server_url, retry_config=retry_config)


# Export for compatibility
__all__ = [
    'MCPClientResult',
    'MCPClient',
    'MCPTool',
    'RetryConfig',
    'MCPToolsAdapter',
    'FilteredMCPToolsAdapter',
    'create_mcp_client',
    'create_mcp_tools_adapter',
    'create_filtered_mcp_tools_adapter',
    'create_agent_mcp_tools',  # RECOMMENDED for agents
    'PLANNER_AGENT_TOOLS',
    'INFORMATION_SEEKER_TOOLS', 
    'WRITER_AGENT_TOOLS'
]
