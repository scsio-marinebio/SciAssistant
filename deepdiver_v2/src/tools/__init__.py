# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""
Model Context Protocol (MCP) Integration

This package contains MCP server implementations, tools, and integrations
for the DeepDiver multi-agent system.
"""

from .mcp_tools import MCPTools

# Server imports
try:
    from .mcp_server_standard import create_app as create_standard_app
    from .mcp_server_simple import app as simple_app
    MCP_STANDARD_AVAILABLE = True
except ImportError:
    MCP_STANDARD_AVAILABLE = False
    create_standard_app = None
    simple_app = None

# For backward compatibility
try:
    standard_app = simple_app  # Keep simple app for basic compatibility
    MCP_AVAILABLE = MCP_STANDARD_AVAILABLE
except Exception as e:
    MCP_AVAILABLE = False
    standard_app = None

__all__ = [
    'MCPTools',
    'create_standard_app',
    'simple_app',
    'standard_app',  # Backward compatibility
    'MCP_AVAILABLE',
    'MCP_STANDARD_AVAILABLE'
] 