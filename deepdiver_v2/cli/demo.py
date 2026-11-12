# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
#!/usr/bin/env python3
"""
CLI Demo for DeepDiver Long Writer Multi-Agent System

This demo showcases the multi-agent system that includes:
- PlannerAgent: Coordinates and orchestrates the entire process
- InformationSeekerAgent: Gathers and researches information  
- WriterAgent: Creates long-form content

Features:
- Loads configuration from config/.env file
- Shows real-time tool calls and reasoning traces
- Displays sub-agent responses and interactions
- Visualizes the complete execution flow
- Query preprocessing for safety and task suitability check
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Configure logging to keep the CLI clean
def setup_clean_logging(debug_mode: bool = False):
    """Configure logging to show only relevant information for the demo"""
    if debug_mode:
        # Debug mode: show all logs with timestamps
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        # Clean demo mode: suppress verbose logs
        
        # Suppress specific noisy loggers
        noisy_loggers = [
            'httpx',
            'httpcore', 
            'urllib3',
            'src.tools.mcp_client',
            # 'src.agents.base_agent',
            'config.config'  # Also suppress config messages in quiet mode
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

# Set up default clean logging before any imports
setup_clean_logging(debug_mode=False)

# Import the multi-agent system components
from config.config import get_config, reload_config
from src.agents.planner_agent import create_planner_agent
from src.agents.base_agent import AgentResponse

console = Console()


class DemoVisualizer:
    """Visualizes the execution of the multi-agent system"""
    
    def __init__(self, quiet_mode: bool = False):
        self.console = console
        self.execution_log = []
        self.quiet_mode = quiet_mode
        
    def _should_display(self, force: bool = False) -> bool:
        """Check if output should be displayed based on quiet mode"""
        return not self.quiet_mode or force
        
    def show_welcome(self):
        """Display welcome message and system info"""
        if not self._should_display():
            return
            
        welcome_text = """
# 🤖 DeepDiver Long Writer Multi-Agent System Demo

This demo showcases an advanced multi-agent system for research and long-form content generation.

## System Components:
- **🧠 PlannerAgent**: Orchestrates the entire process and coordinates sub-agents
- **🔍 InformationSeekerAgent**: Performs web research and gathers information
- **✍️  WriterAgent**: Creates comprehensive long-form content

## Features:
- Real-time tool execution visualization
- Sub-agent response tracking
- Complete reasoning trace display
- Configuration management
- Query safety and suitability pre-check
        """
        self.console.print(Panel(Markdown(welcome_text), title="[bold blue]Welcome", border_style="blue"))
    
    def show_config(self, config):
        """Display current configuration"""
        if not self._should_display():
            return
            
        config_table = Table(title="📋 System Configuration", show_header=True, header_style="bold magenta")
        config_table.add_column("Setting", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="green")
        
        # Safe config display (hide sensitive values)
        safe_config = config.to_dict()
        for key, value in safe_config.items():
            if value is not None and str(value) != "None":
                display_value = str(value)
                if len(display_value) > 60:
                    display_value = display_value[:57] + "..."
                config_table.add_row(key, display_value)
        
        self.console.print(config_table)
    
    def show_planner_start(self, query: str):
        """Show planner starting execution"""
        self.console.print(Panel(
            f"[bold yellow]User Query:[/bold yellow] {query}\n\n"
            f"[bold green]🚀 Starting PlannerAgent execution...[/bold green]",
            title="[bold blue]Task Initiation",
            border_style="green"
        ))
    
    def show_reasoning_step(self, iteration: int, reasoning: str):
        """Display reasoning step"""
        self.console.print(Panel(
            Markdown(f"**Iteration {iteration} - Reasoning:**\n\n{reasoning}"),
            title=f"[bold yellow]🧠 Agent Reasoning (Step {iteration})",
            border_style="yellow"
        ))
    
    def show_tool_call(self, iteration: int, tool_name: str, arguments: Dict[str, Any]):
        """Display tool call"""
        args_json = json.dumps(arguments, indent=2, ensure_ascii=False)
        
        self.console.print(Panel(
            f"[bold cyan]Tool:[/bold cyan] {tool_name}\n\n"
            f"[bold cyan]Arguments:[/bold cyan]\n{Syntax(args_json, 'json', theme='monokai', line_numbers=True)}",
            title=f"[bold cyan]🔧 Tool Call (Step {iteration})",
            border_style="cyan"
        ))
    
    def show_tool_result(self, iteration: int, tool_name: str, result: Dict[str, Any]):
        """Display tool result"""
        success = result.get("success", True)
        status_icon = "✅" if success else "❌"
        status_color = "green" if success else "red"
        
        # Format result for display
        if success and "data" in result:
            display_result = result["data"]
        elif "error" in result:
            display_result = {"error": result["error"]}
        else:
            display_result = result
        
        result_text = json.dumps(display_result, indent=2, ensure_ascii=False)
        if len(result_text) > 1000:
            result_text = result_text[:997] + "..."
        
        self.console.print(Panel(
            f"[bold {status_color}]Status:[/bold {status_color}] {status_icon} {'Success' if success else 'Failed'}\n\n"
            f"[bold {status_color}]Result:[/bold {status_color}]\n{Syntax(result_text, 'json', theme='monokai', line_numbers=True)}",
            title=f"[bold {status_color}]📋 Tool Result: {tool_name} (Step {iteration})",
            border_style=status_color
        ))
    
    def show_sub_agent_execution(self, agent_name: str, task_content: str):
        """Show sub-agent starting execution"""
        self.console.print(Panel(
            f"[bold magenta]Agent:[/bold magenta] {agent_name}\n\n"
            f"[bold magenta]Task:[/bold magenta] {task_content[:500]}{'...' if len(task_content) > 500 else ''}",
            title="[bold magenta]🤝 Sub-Agent Execution",
            border_style="magenta"
        ))
    
    def show_sub_agent_result(self, agent_name: str, result: Dict[str, Any]):
        """Show sub-agent execution result"""
        success = result.get("success", True)
        status_icon = "✅" if success else "❌"
        status_color = "green" if success else "red"
        
        # Extract key information
        iterations = result.get("iterations", 0)
        execution_time = result.get("execution_time", 0)
        
        summary = f"[bold {status_color}]Status:[/bold {status_color}] {status_icon} {'Success' if success else 'Failed'}\n"
        summary += f"[bold blue]Iterations:[/bold blue] {iterations}\n"
        summary += f"[bold blue]Execution Time:[/bold blue] {execution_time:.2f}s\n\n"
        
        if success and "data" in result:
            data = result["data"]
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 200:
                        summary += f"[bold blue]{key}:[/bold blue] {value[:197]}...\n"
                    else:
                        summary += f"[bold blue]{key}:[/bold blue] {value}\n"
        elif "error" in result:
            summary += f"[bold red]Error:[/bold red] {result['error']}\n"
        
        self.console.print(Panel(
            summary,
            title=f"[bold {status_color}]📊 Sub-Agent Result: {agent_name}",
            border_style=status_color
        ))
    
    def show_final_result(self, response: AgentResponse):
        """Display final execution result"""
        # Always show final results, even in quiet mode
        if not self._should_display(force=True):
            return
            
        success = response.success
        status_icon = "✅" if success else "❌"
        status_color = "green" if success else "red"
        
        summary = f"[bold {status_color}]Final Status:[/bold {status_color}] {status_icon} {'Completed Successfully' if success else 'Failed'}\n"
        summary += f"[bold blue]Total Iterations:[/bold blue] {response.iterations}\n"
        summary += f"[bold blue]Total Execution Time:[/bold blue] {response.execution_time:.2f}s\n"
        summary += f"[bold blue]Agent:[/bold blue] {response.agent_name}\n\n"
        
        if success and response.result:
            if isinstance(response.result, dict):
                for key, value in response.result.items():
                    if isinstance(value, str) and len(value) > 3000:
                        summary += f"[bold blue]{key}:[/bold blue] {value[:2997]}...\n\n"
                    else:
                        summary += f"[bold blue]{key}:[/bold blue] {value}\n\n"
        elif response.error:
            summary += f"[bold red]Error:[/bold red] {response.error}\n"
        
        self.console.print(Panel(
            summary,
            title=f"[bold {status_color}]🏁 Final Result",
            border_style=status_color
        ))
    
    def show_reasoning_trace(self, trace: List[Dict[str, Any]]):
        """Display detailed reasoning trace"""
        if not trace:
            return
        
        trace_table = Table(title="🔍 Detailed Execution Trace", show_header=True, header_style="bold cyan")
        trace_table.add_column("Step", style="cyan", width=8)
        trace_table.add_column("Type", style="magenta", width=12)
        trace_table.add_column("Details", style="white")
        
        for i, step in enumerate(trace, 1):
            step_type = step.get("type", "unknown")
            
            if step_type == "reasoning":
                content = step.get("content", "")[:100] + ("..." if len(step.get("content", "")) > 100 else "")
                trace_table.add_row(str(i), "🧠 Reasoning", content)
            
            elif step_type == "action":
                tool = step.get("tool", "")
                result_status = "✅" if step.get("result", {}).get("success", True) else "❌"
                trace_table.add_row(str(i), "🔧 Tool Call", f"{result_status} {tool}")
            
            elif step_type == "error":
                error = step.get("error", "")[:100] + ("..." if len(step.get("error", "")) > 100 else "")
                trace_table.add_row(str(i), "❌ Error", error)
        
        self.console.print(trace_table)

    def show_unsupported_response(self):
        """Display the fixed response for unsupported queries"""
        # Always show this response, even in quiet mode
        self.console.print(Panel(
            "Sorry, your question is not within the current scope of tasks for DeepDiver-V2. Please try asking a question related to long-form writing or complex knowledge Q&A instead.",
            title="[bold yellow]❌ Unsupported Query",
            border_style="yellow"
        ))


class AgentExecutionMonitor:
    """Monitors agent execution and provides real-time feedback"""
    
    def __init__(self, visualizer: DemoVisualizer):
        self.visualizer = visualizer
        self.current_iteration = 0
    
    def on_reasoning_step(self, iteration: int, reasoning: str):
        """Called when agent performs reasoning"""
        self.visualizer.show_reasoning_step(iteration, reasoning)
    
    def on_tool_call(self, iteration: int, tool_name: str, arguments: Dict[str, Any]):
        """Called when agent makes a tool call"""
        self.visualizer.show_tool_call(iteration, tool_name, arguments)
        
        # Check for sub-agent assignments
        if "assign_" in tool_name and "task" in tool_name:
            if "tasks" in arguments:
                for task in arguments.get("tasks", []):
                    task_content = task.get("task_content", "")
                    self.visualizer.show_sub_agent_execution("InformationSeeker", task_content)
            elif "task_content" in arguments:
                task_content = arguments.get("task_content", "")
                self.visualizer.show_sub_agent_execution("Writer", task_content)
    
    def on_tool_result(self, iteration: int, tool_name: str, result: Dict[str, Any]):
        """Called when tool execution completes"""
        self.visualizer.show_tool_result(iteration, tool_name, result)
        
        # Show sub-agent results if this was an assignment
        if "assign_" in tool_name and "task" in tool_name:
            if "data" in result and "tasks" in result["data"]:
                for task_result in result["data"]["tasks"]:
                    agent_name = task_result.get("agent_name", "InformationSeeker")
                    self.visualizer.show_sub_agent_result(agent_name, task_result)
            elif "data" in result:
                agent_name = result["data"].get("agent_name", "Writer")
                self.visualizer.show_sub_agent_result(agent_name, result["data"])


def classify_query(query: str, config) -> Dict[str, Any]:
    """
    Classify user query into one of three categories using LLM:
    1. SAFE_SENSITIVE: Contains unsafe content (insults, political risks, etc.)
    2. NON_KNOWLEDGE: Non-knowledge intensive (no need for research, e.g., greetings, simple calculations)
    3. NORMAL: Requires processing (long-form writing or complex knowledge Q&A)
    
    Returns:
        Dict with 'category' (str) and 'reasoning' (str)
    """
    logger = logging.getLogger(__name__)
    
    # Get model configuration
    model_config = config.get_custom_llm_config()
    pangu_url = model_config.get('url') or os.getenv('MODEL_REQUEST_URL', '')
    model_token = model_config.get('token') or os.getenv('MODEL_REQUEST_TOKEN', '')
    
    # Validate model configuration
    if not pangu_url:
        logger.error("Model URL not configured for query classification")
        # Fallback to NORMAL category if model config is missing
        return {
            "category": "NORMAL",
            "reasoning": "模型配置不完整，跳过分类检查，默认按正常任务处理"
        }
    
    headers = {'Content-Type': 'application/json', 'csb-token': model_token}
    
    # Classification prompt (detailed instructions for accurate categorization)
    prompt_template = """
你是一个Query分类器，需要将用户输入的查询分为以下三类，并给出明确的分类理由：

1. 【SAFE_SENSITIVE - 安全敏感内容】：包含以下任何一种情况的查询
   - 辱骂、侮辱性语言（如脏话、人身攻击）
   - 涉及政治敏感内容（如国家领导人、敏感政治事件、舆情风险话题）
   - 违法违规内容（如暴力、色情、恐怖主义相关）
   - 歧视性言论（种族、性别、宗教等歧视）

2. 【NON_KNOWLEDGE - 非知识密集型任务】：不需要进行信息搜索的简单查询
   - 问候语（如"你好"、"早上好"、"嗨"）
   - 简单计算（如"1+1等于几"、"25乘以4是多少"）
   - 基础闲聊（如"你是谁"）
   - 指令性语句（如"退出"、"帮助"、"开始"）
   - 不需要信息收集的简单问题

3. 【NORMAL - 正常任务】：不包含安全敏感内容，需要进行信息搜索或长文写作的任务
   - 简单的信息收集任务 （如"华为成立时间是什么时候"）
   - 复杂知识问答（如"ACL2025举办地有什么美食推荐"）
   - 长文写作任务（如"写一篇关于气候变化影响的5000字报告"）
   - 需要数据支持的分析（如"2023年全球经济增长数据及分析"）
   - 专业领域研究（如"机器学习在医疗诊断中的应用案例"）

分类要求：
- 严格按照上述定义进行分类，不要遗漏任何关键特征
- 优先判断是否为SAFE_SENSITIVE，其次判断是否为NON_KNOWLEDGE，最后才是NORMAL
- 必须提供清晰的分类理由，说明为什么属于该类别
- 输出格式必须严格遵循：先输出分类理由的思考，然后换行输出分类结果（SAFE_SENSITIVE/NON_KNOWLEDGE/NORMAL）

示例1（SAFE_SENSITIVE）：
该查询包含辱骂性语言"XXX"，符合安全敏感内容的定义，属于需要拦截的内容
SAFE_SENSITIVE

示例2（NON_KNOWLEDGE）：
该查询是简单的问候语"你好"，不需要进行信息搜索，属于非知识密集型任务
NON_KNOWLEDGE

示例3（NORMAL）：
该查询不包含安全敏感内容，要求撰写关于"区块链技术在金融领域的应用"的长文，需要进行信息收集、案例研究和深度分析，属于正常的长文写作任务
NORMAL

用户输入query：$query"""
    
    # Prepare conversation history
    conversation_history = [
        {"role": "user", "content": prompt_template.replace("$query", query) + " /no_think"}
    ]
    
    try:
        # Call LLM with retry logic
        retry_num = 1
        max_retry_num = 3
        while retry_num <= max_retry_num:
            try:
                response = requests.post(
                    url=pangu_url,
                    headers=headers,
                    json={
                        "model": config.model_name,
                        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<s>[unused9]系统：[unused10]' }}{% endif %}{% if message['role'] == 'system' %}{{'<s>[unused9]系统：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'assistant' %}{{'[unused9]助手：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'tool' %}{{'[unused9]工具：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'function' %}{{'[unused9]方法：' + message['content'] + '[unused10]'}}{% endif %}{% if message['role'] == 'user' %}{{'[unused9]用户：' + message['content'] + '[unused10]'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '[unused9]助手：' }}{% endif %}",
                        "spaces_between_special_tokens": False,
                        "messages": conversation_history,
                        "temperature": 0.1,  # Low temperature for deterministic classification
                        "max_tokens": 5000,
                    },
                    timeout=model_config.get("timeout", 60)
                )
                
                response_json = response.json()
                logger.debug(f"Classification API response: {json.dumps(response_json, indent=2)}")
                
                # Extract and parse result
                assistant_message = response_json["choices"][0]["message"]["content"].strip()
                lines = assistant_message.split('\n', 1)
                
                if len(lines) < 2:
                    raise ValueError(f"Invalid response format: {assistant_message}")

                reasoning = lines[0].strip()
                category = lines[1].strip() if len(lines) > 1 else "NORMAL"
                
                # Validate category
                valid_categories = ["SAFE_SENSITIVE", "NON_KNOWLEDGE", "NORMAL"]
                if category not in valid_categories:
                    logger.warning(f"Invalid category '{category}', using fallback NORMAL")
                    category = "NORMAL"
                    reasoning = f"模型返回无效分类 '{category}'，默认按正常任务处理。原始理由：{reasoning}"
                
                return {
                    "category": category,
                    "reasoning": reasoning
                }
                
            except Exception as e:
                logger.error(f"Classification attempt {retry_num} failed: {str(e)}")
                if retry_num == max_retry_num:
                    raise
                time.sleep(2)  # Wait before retry
                retry_num += 1
    
    except Exception as e:
        logger.error(f"Query classification failed: {str(e)}")
        # Fallback to NORMAL category if classification fails
        return {
            "category": "NORMAL",
            "reasoning": f"分类服务暂时不可用（错误：{str(e)[:100]}...），默认按正常任务处理"
        }


def load_environment_config(quiet: bool = False):
    """Load configuration from .env file"""
    try:
        # Check for .env file in config directory
        config_dir = Path(__file__).parent.parent / "config"
        env_file = config_dir / ".env"
        
        if not env_file.exists():
            if not quiet:
                console.print(f"[yellow]⚠️ No .env file found at {env_file}[/yellow]")
                console.print(f"[yellow]💡 Please copy env.template to config/.env and configure your settings[/yellow]")
            return None
        
        # Reload configuration to pick up .env file
        reload_config()
        config = get_config()
        
        if not quiet:
            console.print("[green]✅ Configuration loaded successfully[/green]")
        return config
        
    except Exception as e:
        if not quiet:
            console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
        return None


def create_sample_env_file():
    """Create a sample .env file for demo purposes"""
    config_dir = Path(__file__).parent.parent / "config"
    env_file = config_dir / ".env"
    
    if env_file.exists():
        return
    
    # Copy from template
    template_file = Path(__file__).parent.parent / "env.template"
    if template_file.exists():
        import shutil
        shutil.copy2(template_file, env_file)
        console.print(f"[green]✅ Created .env file from template at {env_file}[/green]")
        console.print("[yellow]⚠️ Please edit the .env file with your actual configuration values[/yellow]")
    else:
        console.print(f"[red]❌ Could not find env.template to copy[/red]")


def run_demo_query(planner, query: str, visualizer: DemoVisualizer, config) -> Optional[AgentResponse]:
    """Run a demo query through the planner with preprocessing"""
    
    # Step 1: Show query information
    visualizer.show_planner_start(query)
    
    # Step 2: Query classification (preprocessing)
    classification_result = classify_query(query, config)
    
    # Step 3: Branch processing based on classification
    unsupported_categories = ["SAFE_SENSITIVE", "NON_KNOWLEDGE"]
    if classification_result["category"] in unsupported_categories:
        # Show fixed response for unsupported queries
        visualizer.show_unsupported_response()
        return None
    
    # Step 4: Process normal query (original flow)
    try:
        # Execute the query
        with console.status("[bold green]Executing planner task...", spinner="dots"):
            response = planner.execute_task(query)
        
        # Show final results
        visualizer.show_final_result(response)
        
        # Show detailed trace if available
        if hasattr(response, 'reasoning_trace') and response.reasoning_trace:
            visualizer.show_reasoning_trace(response.reasoning_trace)
        
        return response
        
    except Exception as e:
        console.print(f"[red]❌ Error during execution: {e}[/red]")
        return None


def main():
    """Main CLI demo function"""
    parser = argparse.ArgumentParser(description="DeepDiver Multi-Agent System Demo")
    parser.add_argument("--query", "-q", type=str, help="Query to execute (interactive mode if not provided)")
    parser.add_argument("--config-only", "-c", action="store_true", help="Only show configuration and exit")
    parser.add_argument("--create-env", "-e", action="store_true", help="Create sample .env file from template")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode with verbose logging")
    parser.add_argument("--quiet", help="Suppress all non-essential output")
    
    args = parser.parse_args()
    
    # Setup logging based on arguments (re-configure if debug mode is requested)
    if args.debug:
        setup_clean_logging(debug_mode=True)
    
    # Initialize visualizer
    visualizer = DemoVisualizer(quiet_mode=args.quiet)
    if not args.quiet:
        visualizer.show_welcome()
    
    # Create sample .env file if requested
    if args.create_env:
        create_sample_env_file()
        return 0
    
    # Load configuration
    config = load_environment_config(quiet=args.quiet)
    if not config:
        if not args.quiet:
            console.print("[red]❌ Cannot proceed without valid configuration[/red]")
            console.print("[yellow]💡 Use --create-env to create a sample configuration file[/yellow]")
        return 1
    
    # Show configuration
    visualizer.show_config(config)
    
    if args.config_only:
        return 0
    
    # Initialize planner agent
    try:
        if not args.quiet:
            console.print("[blue]🔄 Initializing PlannerAgent...[/blue]")
        
        # Create planner with sub-agent configurations
        sub_agent_configs = {
            "information_seeker": {
                "model": config.model_name,
                "max_iterations": config.information_seeker_max_iterations or 30,
            },
            "writer": {
                "model": config.model_name, 
                "max_iterations": config.writer_max_iterations or 30,
                "temperature": config.model_temperature,
                "max_tokens": config.model_max_tokens
            }
        }
        
        planner = create_planner_agent(
            model=config.model_name,
            max_iterations=config.planner_max_iterations or 40,
            sub_agent_configs=sub_agent_configs
        )
        
        if not args.quiet:
            console.print("[green]✅ PlannerAgent initialized successfully[/green]")
        
    except Exception as e:
        if not args.quiet:
            console.print(f"[red]❌ Failed to initialize PlannerAgent: {e}[/red]")
        return 1
    
    # Handle query execution
    if args.query:
        # Single query mode
        run_demo_query(planner, args.query, visualizer, config)
    else:
        # Interactive mode
        if not args.quiet:
            console.print("\n[bold blue]🎯 Interactive Mode[/bold blue]")
            console.print("Enter your queries below. Type 'quit' or 'exit' to leave.")
        
        while True:
            try:                
                prompt_text = "\n[bold cyan]Enter your query:[/bold cyan] " if not args.quiet else "Query: "
                query = console.input(prompt_text).strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    if not args.quiet:
                        console.print("[green]👋 Goodbye![/green]")
                    break
                
                if not query:
                    continue
                
                if not args.quiet:
                    console.print("\n" + "="*80 + "\n")
                run_demo_query(planner, query, visualizer, config)
                if not args.quiet:
                    console.print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                if not args.quiet:
                    console.print("\n[yellow]⚠️ Interrupted by user[/yellow]")
                break
            except EOFError:
                if not args.quiet:
                    console.print("\n[green]👋 Goodbye![/green]")
                break
    
    return 0


if __name__ == "__main__":
    sys.exit(main())