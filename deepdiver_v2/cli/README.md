# CLI Demo for DeepDiver Long Writer Multi-Agent System

This CLI demo showcases the multi-agent system that coordinates between PlannerAgent, InformationSeekerAgent, and WriterAgent to handle complex queries and generate comprehensive long-form content.

## Features

- 🧠 **PlannerAgent**: Orchestrates the entire process and coordinates sub-agents
- 🔍 **InformationSeekerAgent**: Performs web research and gathers information  
- ✍️ **WriterAgent**: Creates comprehensive long-form content
- 📊 **Real-time Visualization**: Shows tool calls, reasoning traces, and sub-agent responses
- ⚙️ **Configuration Management**: Loads settings from .env files

## Setup

### 1. Install Dependencies

```bash
cd deepdiver_v2
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the `config/` directory:

```bash
# From the project root
cp env.template config/.env
```

Then edit `config/.env` with your settings:

```bash
# Custom LLM Service Configuration
MODEL_REQUEST_URL=http://your-llm-service-endpoint/v1/chat/completions
MODEL_REQUEST_TOKEN=your-service-token
MODEL_NAME=pangu_auto

# MCP Server Configuration
MCP_SERVER_URL=http://localhost:6274/mcp
MCP_AUTH_TOKEN=
MCP_USE_STDIO=true

# Agent Iteration Limits
PLANNER_MAX_ITERATION=20
INFORMATION_SEEKER_MAX_ITERATION=30
WRITER_MAX_ITERATION=20

# Mode
PLANNER_MODE=auto # auto, writing, qa

# Other settings...
```

### 3. Start Required Services

Make sure your MCP server is running:

```bash
# Start MCP server (if needed)
python src/tools/mcp_server_standard.py
```

## Usage

### Interactive Mode (Recommended)

```bash
python cli/demo.py
```

This will start an interactive session where you can enter queries and see the full execution flow.

### Single Query Mode

```bash
python cli/demo.py -q "Write a comprehensive analysis of artificial intelligence trends in 2024"
```

### Configuration Only

```bash
python cli/demo.py --config-only
```

### Debug Mode (Verbose Logging)

```bash
python cli/demo.py --debug -q "Debug a specific query"
```

### Quiet Mode (Clean Output)

```bash
python cli/demo.py --quiet -q "Run with minimal output"
```

### Create Sample Configuration

```bash
python cli/demo.py --create-env
```

## Example Queries

### For Information Seeking Tasks:
- "What are the latest developments in quantum computing?"
- "Research the current state of renewable energy adoption globally"
- "Find information about recent AI breakthroughs in healthcare"

### For Long-form Writing Tasks:
- "Write a comprehensive report on the impact of AI on education"
- "Create an in-depth analysis of climate change mitigation strategies"
- "Generate a detailed guide on sustainable business practices"

## Demo Flow Visualization

The demo provides rich visual feedback showing:

1. **🚀 Task Initiation**: Shows the user query and planner startup
2. **🧠 Agent Reasoning**: Displays the planner's reasoning at each step
3. **🔧 Tool Calls**: Shows what tools are being called with their arguments
4. **📋 Tool Results**: Displays the results from each tool execution
5. **🤝 Sub-Agent Execution**: Shows when sub-agents (InformationSeeker, Writer) are invoked
6. **📊 Sub-Agent Results**: Displays results from sub-agent executions
7. **🏁 Final Result**: Shows the complete execution summary
8. **🔍 Execution Trace**: Detailed step-by-step trace of the entire process

## Output Modes

The CLI demo supports different output modes for different use cases:

### Default Mode
Shows the full rich interface with welcome screen, progress bars, and detailed visualization of all agent interactions.

### Quiet Mode (`--quiet`)
Suppresses all non-essential output, showing only final results. Useful for:
- Integration with scripts or automation
- Focusing on results without process details
- Running in environments where rich output isn't needed

### Debug Mode (`--debug`)
Enables verbose logging with timestamps, showing all internal system messages. Useful for:
- Troubleshooting configuration issues
- Understanding detailed agent behavior
- Development and debugging

```bash
# Examples of different modes
python cli/demo.py --query "Test query"  # Default rich mode
python cli/demo.py --quiet --query "Test query"  # Minimal output
python cli/demo.py --debug --query "Test query"  # Verbose debugging
```

## Troubleshooting

### Configuration Issues

If you see configuration errors:

1. Ensure `config/.env` exists and is properly formatted
2. Check that all required environment variables are set
3. Verify your LLM service endpoint is accessible
4. Confirm MCP server is running and reachable
5. Use `--debug` mode to see detailed error messages

### Agent Initialization Issues

If agent initialization fails:

1. Check MCP server connectivity
2. Verify model configuration is correct
3. Ensure required permissions for workspace directories
4. Check log output for specific error messages

### Tool Execution Issues

If tool calls fail:

1. Verify MCP server is running and has the required tools
2. Check network connectivity for web search/parse tools
3. Ensure workspace directories exist and are writable
4. Review tool arguments for correctness

## Advanced Usage

### Custom Sub-Agent Configurations

You can customize sub-agent behavior by modifying the configurations in the demo script:

```python
sub_agent_configs = {
    "information_seeker": {
        "model": "your-model",
        "max_iterations": 30,
    },
    "writer": {
        "model": "your-model", 
        "max_iterations": 20,
        "temperature": 0.3,
        "max_tokens": 16384
    }
}
```

### Monitoring and Debugging

Enable debug mode in your `.env` file:

```bash
DEBUG_MODE=true
```

This will provide more detailed logging and error information.

## Architecture Overview

The demo showcases a sophisticated multi-agent architecture:

```
User Query
    ↓
PlannerAgent (Coordinator)
    ↓
├── InformationSeekerAgent (Research)
│   ├── Webpage Search Tools
│   ├── Webpage Parseing Tools  
│   ├── Document Analysis Tools
│   └── File Management Tools
│
└── WriterAgent (Content Generation)
    ├── File Reading Tools
    ├── Document QA Tools
    ├── Content Synthesis
    └── Long-form Writing
```

Each agent follows the ReAct pattern (Reasoning + Acting) with iterative refinement until task completion.
