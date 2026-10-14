# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""
Multi-Agent System - Agent Module

This module provides the core agents for the multi-agent system:
- BaseAgent: Abstract base class with common functionality
- InformationSeekerAgent: Research and information gathering
- WriterAgent: Content creation and writing
- PlannerAgent: Top-level orchestrator

All agents follow the ReAct pattern and use standardized TaskInput format.
"""

from .base_agent import (
    BaseAgent,
    AgentConfig,
    AgentResponse,
    TaskInput,
    create_agent_config
)

from .subjective_information_seeker import (
    InformationSeekerAgent,
    create_subjective_information_seeker
)

from .objective_information_seeker import (
    InformationSeekerAgent,
    create_objective_information_seeker
)

from .writer_agent import (
    WriterAgent,
    create_writer_agent
)

from .planner_agent import (
    PlannerAgent,
    create_planner_agent
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentConfig",
    "AgentResponse",
    "TaskInput",
    "create_agent_config",

    # Specific agents
    "InformationSeekerAgent",
    "create_subjective_information_seeker",
    "create_objective_information_seeker",
    "WriterAgent",
    "create_writer_agent",
    "PlannerAgent",
    "create_planner_agent"
]

# Version info
__version__ = "0.1.0"
__author__ = "DeepDiver Multi-Agent System"
