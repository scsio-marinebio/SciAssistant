# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
import json
import logging
import time
import os
import sys
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import litellm

logger = logging.getLogger(__name__)

# Import MCP client instead of direct tools
try:
    from ..tools import mcp_client as _mcp_client_module  # noqa: F401
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False


@dataclass
class AgentConfig:
    """Configuration for agents - session management handled entirely by MCP server"""
    agent_name: str = "base_agent"
    planner_mode: str = "auto"
    model: Optional[str] = None
    max_iterations: int = 10
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Paths used by writer and other agents
    trajectory_storage_path: Optional[str] = None
    report_output_path: Optional[str] = None
    document_analysis_path: Optional[str] = None


@dataclass
class AgentResponse:
    """Standardized response format for all agents"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    iterations: int = 0
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    agent_name: str = ""
    execution_time: float = 0.0


@dataclass
class TaskInput:
    """Standardized task input format for all agents"""
    task_content: str                                    # The specific task content
    task_steps_for_reference: Optional[str] = None       # Reference steps for execution
    deliverable_contents: Optional[str] = None           # Format of final deliverable
    current_task_status: Optional[str] = None            # Description of current task status
    task_executor: str = "info_seeker"                  # Name of task executor (info_seeker, writer)
    workspace_id: Optional[str] = None                   # Workspace ID for stored files and memory
    acceptance_checking_criteria: Optional[str] = None   # Criteria for determining task completion and quality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TaskInput to dictionary format"""
        return {
            "task_content": self.task_content,
            "task_steps_for_reference": self.task_steps_for_reference,
            "deliverable_contents": self.deliverable_contents,
            "current_task_status": self.current_task_status,
            "task_executor": self.task_executor,
            "workspace_id": self.workspace_id,
            "acceptance_checking_criteria": self.acceptance_checking_criteria
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskInput':
        """Create TaskInput from dictionary"""
        return cls(
            task_content=data.get("task_content", ""),
            task_steps_for_reference=data.get("task_steps_for_reference"),
            deliverable_contents=data.get("deliverable_contents"),
            current_task_status=data.get("current_task_status"),
            task_executor=data.get("task_executor", "info_seeker"),
            workspace_id=data.get("workspace_id"),
            acceptance_checking_criteria=data.get("acceptance_checking_criteria")
        )
    
    def format_for_prompt(self) -> str:
        """Format the task input for use in prompts"""
        prompt = f"Task Content:\n{self.task_content}\n\n"
        
        if self.task_steps_for_reference:
            prompt += f"Task Steps for Reference:\n{self.task_steps_for_reference}\n\n"
        
        if self.deliverable_contents:
            prompt += f"Deliverable Contents:\n{self.deliverable_contents}\n\n"
        
        if self.current_task_status:
            prompt += f"Current Task Status:\n{self.current_task_status}\n\n"
        
        if self.acceptance_checking_criteria:
            prompt += f"Acceptance Checking Criteria:\n{self.acceptance_checking_criteria}\n\n"
        
        prompt += f"Task Executor: {self.task_executor}\n"
        
        if self.workspace_id:
            prompt += f"Workspace ID: {self.workspace_id}\n"
        
        return prompt


class SectionWriterTaskInput(TaskInput):
    """
    Specialized TaskInput for section writing tasks

    Only stores the essential parameters. The section_writer agent
    will handle prompt assembly internally.
    """

    def __init__(
        self,
        task_content: str,
        user_query: str,
        write_file_path: str,
        overall_outline: str,
        current_chapter_outline: str,
        key_files: List[Dict[str, Any]],
        written_chapters: str = "",
        workspace_id: Optional[str] = None
    ):
        # Store the section writer specific parameters
        self.write_file_path = write_file_path
        self.user_query = user_query
        self.current_chapter_outline = current_chapter_outline
        self.key_files = key_files
        self.written_chapters = written_chapters
        self.overall_outline = overall_outline

        # Initialize parent TaskInput with minimal required fields
        super().__init__(
            task_content=task_content,
            task_executor="section_writer",
            workspace_id=workspace_id,
        )


class WriterAgentTaskInput(TaskInput):
    """
    Specialized TaskInput for section writing tasks

    Only stores the 4 essential parameters. The section_writer agent
    will handle prompt assembly internally.
    """

    def __init__(
        self,
        task_content: str,
        user_query: str,
        key_files: List[Dict[str, Any]],
        workspace_id: Optional[str] = None
    ):
        # Store the section writer specific parameters
        self.user_query = user_query
        self.key_files = key_files

        # Initialize parent TaskInput with minimal required fields
        super().__init__(
            task_content=task_content,
            task_executor="writer_agent",
            workspace_id=workspace_id,
        )


class BaseAgent(ABC):
    """
    Base class for all agents with MCP server-managed sessions.
    
    Session management is now entirely handled by the MCP server:
    - Server assigns session IDs on connection
    - Server creates workspace folders with UUID names
    - All tool operations are performed in server-managed workspaces
    """
    
    def __init__(self, config: AgentConfig, shared_mcp_client=None):
        self.execution_stats = None
        self.reasoning_trace = None
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.agent_name}")
        
        # Session info is populated by the MCP server
        self.session_info = None
        
        # Tool management
        self.mcp_tools = None
        self.available_tools = {}

        self.reset_trace()
        
        # Initialize MCP tools (server will handle session creation or use shared client)
        self._initialize(shared_mcp_client)
    
    def _initialize(self, shared_mcp_client=None):
        """Initialize agent with MCP server connection or shared client"""
        try:
            self.logger.info(f"Initializing agent {self.config.agent_name}")
            
            if shared_mcp_client:
                # Use shared MCP client with agent-specific tool filtering
                agent_type = self._get_agent_type()
                self.mcp_tools = self._create_filtered_mcp_tools(shared_mcp_client, agent_type)
                self.logger.info(f"Agent {self.config.agent_name} using shared MCP client with {agent_type} tools")
            else:
                # Create MCP tools with agent-specific filtering (no more unfiltered access)
                self.mcp_tools = self._create_filtered_mcp_tools_standalone()
            
            # Discover available tools
            self.available_tools = self._discover_mcp_tools()
            
            # Build tool schemas for function calling
            self.tool_schemas = self._build_tool_schemas()
            
            self.logger.info(f"Agent {self.config.agent_name} initialized successfully")
            self.logger.info(f"Available tools: {list(self.available_tools.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.config.agent_name}: {e}")
            raise

    def _discover_mcp_tools(self) -> Dict[str, Any]:
        """Discover available tools from MCP server or fallback tools"""
        available_tools = {}
        
        # Try to get tools from MCP client first
        if hasattr(self.mcp_tools, 'get_available_tools'):
            try:
                mcp_tools_dict = self.mcp_tools.get_available_tools()
                for tool_name, tool_info in mcp_tools_dict.items():
                    # For proper MCP architecture, store tool info for direct client calls
                    # instead of creating wrapper lambda functions
                    available_tools[tool_name] = tool_info
                
                if available_tools:
                    self.logger.info(f"Discovered {len(available_tools)} tools from MCP server")
                    return available_tools
            except Exception as e:
                self.logger.warning(f"Failed to discover MCP tools: {e}")
        
        # Fallback: if MCP client not available, use direct method access
        # This should rarely be needed with proper MCP setup
        if hasattr(self.mcp_tools, '__dict__'):
            for attr_name in dir(self.mcp_tools):
                if not attr_name.startswith('_') and callable(getattr(self.mcp_tools, attr_name)):
                    available_tools[attr_name] = getattr(self.mcp_tools, attr_name)
        
        return available_tools
    
    def _get_agent_type(self) -> str:
        """Get agent type for tool filtering"""
        agent_name = self.config.agent_name.lower()
        if "planner" in agent_name:
            return "planner"
        elif "information" in agent_name or "seeker" in agent_name:
            return "information_seeker"
        elif "writer" in agent_name:
            return "writer"
        else:
            # Default to planner tools for unknown agent types
            return "planner"
    
    def _create_filtered_mcp_tools(self, shared_client, agent_type: str):
        """Create filtered MCP tools adapter using shared client"""
        try:
            from src.tools.mcp_client import create_filtered_mcp_tools_adapter
            return create_filtered_mcp_tools_adapter(shared_client, agent_type)
        except ImportError:
            # Fallback if FilteredMCPToolsAdapter not available
            self.logger.warning("FilteredMCPToolsAdapter not available, using regular adapter")
            from src.tools.mcp_client import MCPToolsAdapter
            adapter = MCPToolsAdapter.__new__(MCPToolsAdapter)
            adapter.client = shared_client
            return adapter
    
    def _create_filtered_mcp_tools_standalone(self):
        """Create filtered MCP tools adapter with its own client connection"""
        try:
            # Get agent type for filtering
            agent_type = self._get_agent_type()
            
            # Create a new MCP client
            client = self._create_new_mcp_client()
            
            # Apply filtering based on agent type
            from src.tools.mcp_client import create_filtered_mcp_tools_adapter
            filtered_adapter = create_filtered_mcp_tools_adapter(client, agent_type)
            
            self.logger.info(f"Agent {self.config.agent_name} created filtered MCP adapter with {agent_type} tools")
            return filtered_adapter
            
        except Exception as e:
            self.logger.error(f"Failed to create filtered MCP tools: {e}")
            raise RuntimeError(f"Failed to create filtered MCP client for {self.config.agent_name}: {e}")
    
    def _create_new_mcp_client(self):
        """Create a new MCP client connection"""
        try:
            # Get MCP configuration
            from config.config import get_mcp_config
            mcp_config = get_mcp_config()
            
            # Create MCP client
            from src.tools.mcp_client import MCPClient
            
            if mcp_config.get("server_url") and not mcp_config.get("use_stdio", True):
                # HTTP-based MCP server
                client = MCPClient(server_url=mcp_config["server_url"])
                self.logger.info(
                    f"Agent {self.config.agent_name} connected to HTTP MCP server: {mcp_config['server_url']}")
            else:
                # Default to the expected HTTP MCP server on port 6274
                client = MCPClient(server_url="http://localhost:6274/mcp")
                self.logger.info(
                    f"Agent {self.config.agent_name} connected to default HTTP MCP server: http://localhost:6274/mcp")
                
            return client
                
        except Exception as e:
            self.logger.error(f"Failed to create MCP client: {e}")
            raise RuntimeError(f"MCP client creation failed for {self.config.agent_name}: {e}")
        
    # NOTE: _create_mcp_tools() method removed to prevent unfiltered tool access.
    # All agents now use _create_filtered_mcp_tools_standalone() or _create_filtered_mcp_tools() 
    # to ensure proper tool isolation and security.
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current server-managed session"""
        try:
            # First check environment variables (set by cli/a.py)
            env_session_id = os.environ.get('AGENT_SESSION_ID')
            env_workspace_path = os.environ.get('AGENT_WORKSPACE_PATH')
            
            if env_session_id and env_workspace_path:
                return {
                    "session_id": env_session_id,
                    "workspace_path": env_workspace_path,
                    "server_managed": True,
                    "agent_name": self.config.agent_name,
                    "source": "environment"
                }
            
            # Then try the adapter's get_session_info method if available
            if hasattr(self.mcp_tools, 'get_session_info'):
                session_info = self.mcp_tools.get_session_info()
                if session_info:
                    # Add agent-specific information
                    session_info.update({
                        "server_managed": True,
                        "agent_name": self.config.agent_name
                    })
                    return session_info
            
            # Fallback: Check if we have an MCP tools adapter with a client
            if hasattr(self.mcp_tools, 'client'):
                client = self.mcp_tools.client
                
                # Check if client has session ID and connection status
                if hasattr(client, '_session_id') and hasattr(client, 'is_connected'):
                    return {
                        "session_id": client._session_id,
                        "server_managed": True,
                        "agent_name": self.config.agent_name,
                        "connected": client.is_connected()
                    }
            
            # Fallback: check if mcp_tools has session info directly
            if hasattr(self.mcp_tools, '_session_id'):
                return {
                    "session_id": self.mcp_tools._session_id,
                    "server_managed": True,
                    "agent_name": self.config.agent_name,
                    "connected": getattr(self.mcp_tools, 'is_connected', lambda: True)()
                }
            
            # If no session info available, return basic info
            return {
                "session_id": None,
                "server_managed": True,
                "agent_name": self.config.agent_name,
                "connected": hasattr(self.mcp_tools, 'client') and getattr(self.mcp_tools.client, 'is_connected',
                                                                           lambda: False)()
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get session info: {e}")
            return {
                "session_id": None,
                "server_managed": True,
                "agent_name": self.config.agent_name,
                "connected": False,
                "error": str(e)
            }
    
    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """Build tool schemas for function calling"""
        schemas = []
        
        # Get agent-specific tool schemas
        agent_schemas = self._build_agent_specific_tool_schemas()
        schemas.extend(agent_schemas)
        
        return schemas
    
    def _build_agent_specific_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Build agent-specific tool schemas using proper MCP architecture.
        Schemas come from MCP server via client, not direct imports.
        """
        schemas = []
        
        # Proper MCP way: Get schemas from MCP client (which got them from server)
        try:
            if hasattr(self.mcp_tools, 'get_tool_schemas'):
                # Use the MCP client to get schemas (proper MCP architecture)
                schemas = self.mcp_tools.get_tool_schemas()
                self.logger.info(f"Retrieved {len(schemas)} tool schemas from MCP server")
            else:
                # Fallback for adapters that don't have the new method yet
                self.logger.warning("MCP adapter doesn't support get_tool_schemas, using fallback")
                schemas = self._build_fallback_schemas()
        except Exception as e:
            self.logger.warning(f"Failed to get schemas from MCP client: {e}, using fallback")
            schemas = self._build_fallback_schemas()
        
        return schemas
    
    def _build_fallback_schemas(self) -> List[Dict[str, Any]]:
        """Fallback schema building if MCP client method fails"""
        schemas = []
        
        # Try to get tool info from MCP client
        if hasattr(self.mcp_tools, 'get_available_tools'):
            try:
                available_tools = self.mcp_tools.get_available_tools()
                for tool_name, tool_info in available_tools.items():
                    schema = {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": getattr(tool_info, 'description', f"Tool: {tool_name}"),
                            "parameters": getattr(tool_info, 'input_schema', {"type": "object", "properties": {}, "required": []})
                        }
                    }
                    schemas.append(schema)
                self.logger.info(f"Built {len(schemas)} schemas using fallback method")
            except Exception as e:
                self.logger.warning(f"Fallback schema building failed: {e}")
        
        return schemas
    
    def execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Execute a tool call and return results using proper MCP architecture"""
        tool_name = tool_call["name"]
        
        try:
            # Parse arguments
            arguments = tool_call["arguments"]
            
            # Check if tool is available
            if tool_name not in self.available_tools:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not available for this agent"
                }
            
            # Route tool execution based on tool type
            # Built-in tools (like assign_task_to_*) are callable methods, not MCP server tools
            if callable(self.available_tools[tool_name]):
                # Built-in tool: execute locally
                tool_function = self.available_tools[tool_name]
                result = tool_function(**arguments)
                
                # Convert result to standard format
                if hasattr(result, 'to_dict'):
                    return result.to_dict()
                elif isinstance(result, dict):
                    return result
                else:
                    return {
                        "success": True,
                        "data": result,
                        "error": None,
                        "metadata": {}
                    }
                    
            elif hasattr(self.mcp_tools, 'client') and hasattr(self.mcp_tools.client, 'call_tool'):
                # MCP server tool: execute via client
                result = self.mcp_tools.client.call_tool(tool_name, arguments)
                
                # Convert MCPClientResult to standard format
                if hasattr(result, 'success'):
                    return {
                        "success": result.success,
                        "data": result.data,
                        "error": result.error,
                        "metadata": getattr(result, 'metadata', {})
                    }
                else:
                    return result
            else:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' is not executable (neither built-in nor MCP)"
                }
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def log_reasoning(self, iteration: int, reasoning: str):
        """Log reasoning step in the trace"""
        self.reasoning_trace.append({
            "type": "reasoning",
            "iteration": iteration,
            "content": reasoning,
            "timestamp": time.time()
        })
        self.execution_stats["reasoning_steps"] += 1
        self.execution_stats["total_steps"] += 1
        self.logger.info(f"Reasoning (Iter {iteration}): {reasoning[:100]}...")
    
    def log_action(self, iteration: int, tool: str, arguments: Dict[str, Any], result: Dict[str, Any]):
        """Log action step in the trace"""
        self.reasoning_trace.append({
            "type": "action", 
            "iteration": iteration,
            "tool": tool,
            "arguments": arguments,
            "result": result,
            "timestamp": time.time()
        })
        self.execution_stats["action_steps"] += 1
        self.execution_stats["total_steps"] += 1
        
        # Log success/failure
        success = result.get("success", True)
        status = "Success" if success else "Failed"
        self.logger.info(f"Action (Iter {iteration}): {tool} -> {status} -> {str(arguments)[:400]}...")
    
    def log_error(self, iteration: int, error: str):
        """Log error in the trace"""
        self.reasoning_trace.append({
            "type": "error",
            "iteration": iteration,
            "error": error,
            "timestamp": time.time()
        })
        self.execution_stats["error_steps"] += 1
        self.execution_stats["total_steps"] += 1
        self.logger.error(f"Error (Iter {iteration}): {error}")
    
    def reset_trace(self):
        """Reset the reasoning trace for a new task"""
        self.reasoning_trace = []
        self.execution_stats = {
            "total_steps": 0,
            "reasoning_steps": 0,
            "action_steps": 0, 
            "error_steps": 0,
            "tool_usage": {},
            "success_rate": 1.0
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        # Calculate success rate
        if self.execution_stats["action_steps"] > 0:
            failed_actions = sum(1 for step in self.reasoning_trace 
                               if step.get("type") == "action" 
                               and not step.get("result", {}).get("success", True))
            self.execution_stats["success_rate"] = (
                (self.execution_stats["action_steps"] - failed_actions) / 
                self.execution_stats["action_steps"]
            )
        
        return self.execution_stats.copy()
    
    def create_response(self, success: bool, result: Dict[str, Any] = None, 
                       error: str = None, iterations: int = 0, 
                       execution_time: float = 0.0) -> AgentResponse:
        """Create a standardized agent response"""
        return AgentResponse(
            success=success,
            result=result,
            error=error,
            iterations=iterations,
            reasoning_trace=self.reasoning_trace.copy(),
            agent_name=self.config.agent_name,
            execution_time=execution_time
        )
    
    def validate_config(self) -> bool:
        """Validate agent configuration"""
        try:
            # Check required fields
            if not self.config.agent_name:
                return False
            if not self.config.model:
                return False
            if self.config.max_iterations <= 0:
                return False
            if not (0.0 <= self.config.temperature <= 2.0):
                return False
            if self.config.max_tokens <= 0:
                return False
            
            return True
        except Exception:
            return False
    
    @abstractmethod
    def execute_task(self, task_input: TaskInput) -> AgentResponse:
        """
        Execute a task using the standardized TaskInput format
        
        Args:
            task_input: TaskInput object with standardized task information
            
        Returns:
            AgentResponse with results and process trace
        """
        pass
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent"""
        pass


# Simple factory function for creating agent configurations

def create_agent_config(
    agent_name: str,
    model: Optional[str] = None,
    max_iterations: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> AgentConfig:
    """
    Create an AgentConfig instance for server-managed sessions.
    
    Args:
        agent_name: Name of the agent
        model: LLM model to use
        max_iterations: Maximum number of iterations
        temperature: LLM temperature setting
        max_tokens: Maximum tokens for LLM response
        
    Returns:
        Configured AgentConfig instance
    """
    # Load env-backed defaults
    try:
        from config.config import get_config
        api_cfg = get_config()
    except Exception as e:
        raise ValueError(f"Failed to load global configuration: {e}")
    
    planner_mode = getattr(api_cfg, "planner_mode", "auto")

    resolved_model = model if model is not None else getattr(api_cfg, "model_name", None)
    if not resolved_model:
        raise ValueError("Model is not specified and MODEL_NAME is not set in environment")

    resolved_temperature = temperature if temperature is not None else getattr(api_cfg, "model_temperature", None)
    if resolved_temperature is None:
        raise ValueError("Temperature is not specified and MODEL_TEMPERATURE is not set in environment")

    resolved_max_tokens = max_tokens if max_tokens is not None else getattr(api_cfg, "model_max_tokens", None)
    if resolved_max_tokens is None:
        raise ValueError("Max tokens is not specified and MODEL_MAX_TOKENS is not set in environment")

    # Optional paths used by writer and others
    trajectory_storage_path = getattr(api_cfg, "trajectory_storage_path", None)
    report_output_path = getattr(api_cfg, "report_output_path", None)
    document_analysis_path = getattr(api_cfg, "document_analysis_path", None)

    # Resolve max_iterations per agent type
    if max_iterations is None:
        agent_lower = (agent_name or "").lower()
        resolved_max_iterations = None
        if "planner" in agent_lower:
            resolved_max_iterations = getattr(api_cfg, "planner_max_iterations", None)
        elif "writer" in agent_lower:
            resolved_max_iterations = getattr(api_cfg, "writer_max_iterations", None)
        elif "information" in agent_lower or "seeker" in agent_lower:
            resolved_max_iterations = getattr(api_cfg, "information_seeker_max_iterations", None)
        # if not found in env, raise
        if resolved_max_iterations is None:
            raise ValueError("Max iterations not specified and no env override (PLANNER_MAX_ITERATION/WRITER_MAX_ITERATION/INFORMATION_SEEKER_MAX_ITERATION)")
        max_iterations = resolved_max_iterations

    return AgentConfig(
        agent_name=agent_name,
        planner_mode=planner_mode,
        model=resolved_model,
        max_iterations=int(max_iterations),
        temperature=resolved_temperature,
        max_tokens=resolved_max_tokens,
        trajectory_storage_path=trajectory_storage_path,
        report_output_path=report_output_path,
        document_analysis_path=document_analysis_path
    )