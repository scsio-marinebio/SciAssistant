"""
PlannerAgent HTTP Server
基于FastAPI实现的PlannerAgent服务器，提供RESTful API接口
支持单查询处理、批量查询处理等功能
本文件配置项：
	app="a:app",
	host="0.0.0.0",
	port=8000,		# a.py对外提供服务端口号
	reload=False,
	workers=1
"""
import asyncio
import os
import sys
import time
import json
import uuid
import signal
import threading
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import requests

# 【重要】先调整Python路径，再导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加 new_deepdiver 到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # 添加项目根目录到路径

# 导入日志配置
from config.logging_config import get_logger, quick_setup

# 导入核心模块
from src.agents.planner_agent import PlannerAgent
from src.agents.base_agent import AgentConfig
from src.tools.mcp_tools import MCPTools
from src.utils.task_manager import task_manager, TaskStatus

# 配置日志 - 捕获所有日志到文件
import logging

# 使用绝对路径，确保日志写入项目根目录的logs文件夹
log_dir = Path(__file__).parent.parent.parent / 'logs'
quick_setup(environment='production', log_dir=str(log_dir))
logger = get_logger(__name__)

# 确保第三方库的日志也写入文件
logging.getLogger('config.config').setLevel(logging.INFO)
logging.getLogger('faiss.loader').setLevel(logging.INFO)
logging.getLogger('litellm').setLevel(logging.WARNING)

# 导入FastAPI相关模块
from typing import List, Dict, Optional, Any, cast
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import queue

# 导入原有核心模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agents.planner_agent import PlannerAgent, create_planner_agent
from src.agents.base_agent import AgentConfig
from config.config import get_config
from fastapi.middleware.cors import CORSMiddleware
from typing import cast, Any
# a.py 新增会话管理工具类
import uuid
from pathlib import Path
from typing import Dict, Optional
# a.py 应用初始化改造
from concurrent.futures import ThreadPoolExecutor
import asyncio
# a.py 定时清理任务
from fastapi import BackgroundTasks
import time

# 全局变量
query_history: List[Dict[str, Any]] = []  # 仅记录查询历史，无会话关联
batch_results: Dict[str, Any] = {}
executor = None  # 线程池将在lifespan中初始化

# task_id 到 session_id 的映射（用于查找workspace）
task_session_mapping: Dict[str, str] = {}

# Human in the loop: 存储等待大纲确认的任务状态
pending_outline_confirmations: Dict[str, Dict[str, Any]] = {}
pending_outline_lock = threading.Lock()  # 线程锁保护
progress_queues: Dict[str, queue.Queue] = {}  # 进度队列管理器：task_id -> Queue
progress_history: Dict[str, List[dict]] = {}  # 历史进度消息存储：task_id -> List[progress_data]
queue_executor = None  # 队列处理线程池
cancel_executor = None  # 专用取消操作线程池，避免被任务执行阻塞
MAX_CONCURRENT_TASKS = 4  # 最大并发任务数


# 数据模型（请求/响应格式）
class UserFile(BaseModel):
    """用户上传的文件信息（简化版：只包含必需字段）"""
    file_id: str  # 文件ID，用于从Flask后端下载文件
    filename: str  # 文件名，用于显示和保存


class SearchSources(BaseModel):
    """搜索源选择"""
    websearch: bool = True
    pubmed: bool = True
    arxiv: bool = True
    google_scholar: bool = True
    scihub: bool = True


class SingleQueryRequest(BaseModel):
    query: str  # 查询文本
    taskId: str  # 任务ID
    frontend_session_id: Optional[str] = None  # 前端会话ID，用于存储消息到数据库
    user_files: Optional[List[UserFile]] = []  # 强制使用的文件列表（直接上传）
    reference_files: Optional[List[UserFile]] = []  # 可选参考的文件列表（从文档库选择）
    use_web_search: bool = True  # [DEPRECATED] 已废弃，请使用 search_sources 参数控制搜索源
    prioritize_user_files: bool = True  # 是否优先使用用户文件
    username: Optional[str] = "用户"  # 用户名，用于生成报告署名
    search_sources: Optional[SearchSources] = None  # 搜索源选择（控制 websearch/pubmed/arxiv/google_scholar/scihub）
    human_in_loop: bool = False  # Human in the loop 模式：大纲生成后等待用户确认


class BatchQueryRequest(BaseModel):
    queries: List[str]  # 批量查询列表
    max_workers: Optional[int] = None  # 可选：指定进程数


class AgentSubResponse(BaseModel):
    """子 Agent（如 information_seeker）的响应结构"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    reasoning_trace: Optional[List[Dict[str, Any]]] = []
    iterations: int
    execution_time: float
    agent_name: str


# 章节写作Agent的响应（特殊处理，因由Writer调用）
class SectionWriterSubResponse(BaseModel):
    section_task: Optional[Dict[str, Any]] = None  # 章节任务参数
    section_result: Optional[Dict[str, Any]] = None  # 章节写作结果
    execution_time: float = 0.0


# 最终接口响应模型
class QueryResponse(BaseModel):
    # 1. 基础请求信息
    success: bool
    query: str
    timestamp: str
    session_id: str
    task_id: Optional[str] = None  # 新增：任务ID用于跟踪和取消
    # PlannerAgent信息
    planner_result: Optional[Dict[str, Any]] = None
    planner_error: Optional[str] = None
    planner_reasoning_trace: List[Dict[str, Any]] = []
    planner_iterations: int = 0
    planner_execution_time: float = 0.0
    planner_agent_name: str = ""

    # 子Agent响应
    section_writer_responses: List[SectionWriterSubResponse] = []
    
    # 最终报告内容
    final_report: Optional[str] = None  # Markdown格式的最终报告内容
    report_path: Optional[str] = None  # 报告文件路径（相对于workspace）
    
    # Human in the loop 相关字段
    waiting_for_outline_confirm: bool = False  # 是否等待用户确认大纲
    outline_content: Optional[str] = None  # 大纲内容（仅在 waiting_for_outline_confirm=True 时有值）
    reasoning_content: Optional[str] = None  # reasoning内容（仅在 waiting_for_outline_confirm=True 时有值）


# Human in the loop 大纲确认请求模型
class OutlineConfirmRequest(BaseModel):
    task_id: str  # 任务ID
    session_id: str  # 会话ID
    action: str  # "confirm" 或 "cancel"
    outline: Optional[str] = None  # 用户修改后的大纲（如果有修改）
    modified: bool = False  # 是否修改了大纲


class BatchResponse(BaseModel):
    batch_id: str
    status: str  # "processing" 或 "completed"
    total_queries: int
    completed_count: int
    results: Optional[List[QueryResponse]] = None


# 服务器生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务器启动和关闭时的处理逻辑"""
    import random

    # 【多进程兼容性修复】添加随机延迟，避免多个 worker 同时初始化
    # 延迟 0-2 秒，错开资源初始化时间
    delay = random.uniform(0, 2)
    await asyncio.sleep(delay)

    # 启动时初始化环境变量
    if not os.environ.get('MCP_SERVER_URL'):
        os.environ['MCP_SERVER_URL'] = 'http://localhost:6274/mcp/'
        os.environ['MCP_USE_STDIO'] = 'false'

    # 初始化全局线程池
    global executor, queue_executor, cancel_executor
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)
    queue_executor = ThreadPoolExecutor(max_workers=1)  # 队列处理器单线程
    cancel_executor = ThreadPoolExecutor(max_workers=8)  # 专用取消操作线程池，支持多个并发取消请求
    
    # 启动队列处理器
    queue_executor.submit(process_task_queue)

    logger.info(f"PlannerAgent服务器初始化成功（PID: {os.getpid()}, 延迟: {delay:.2f}s）")
    yield  # 运行期间

    # 关闭线程池
    logger.info(f"服务器正在关闭... (PID: {os.getpid()})")
    if executor:
        executor.shutdown(wait=True)
    if queue_executor:
        queue_executor.shutdown(wait=True)
    if cancel_executor:
        cancel_executor.shutdown(wait=True)


# 初始化FastAPI应用
app = FastAPI(
    title="PlannerAgent Server (Stateless)",
    description="无状态PlannerAgent服务器，支持并发查询处理",
    version="1.0.0",
    lifespan=lifespan
)

# 配置跨域
app.add_middleware(
    cast(Any, CORSMiddleware),
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# 辅助函数：下载用户文件到工作区
def _download_user_files(user_files_data: List[Dict[str, str]], workspace_path: Path) -> None:
    """下载用户上传的文件到工作区，按类型分目录存储"""
    if not user_files_data:
        return

    # 分离强制文件和可选文件
    mandatory_files = [f for f in user_files_data if f.get('type') == 'mandatory']
    optional_files = [f for f in user_files_data if f.get('type') == 'optional']

    logger.info(f"检测到 {len(user_files_data)} 个文件：强制使用 {len(mandatory_files)} 个，可选参考 {len(optional_files)} 个")

    try:
        mcp_tools = MCPTools(workspace_path=workspace_path)

        # 下载强制使用的文件到 user_uploads/
        if mandatory_files:
            file_ids = [f['file_id'] for f in mandatory_files]
            logger.info(f"下载强制文件到 user_uploads/: {file_ids}")

            download_result = mcp_tools.process_user_uploaded_files(
                file_ids=file_ids,
                backend_url="http://localhost:5000",
                target_subdir="user_uploads"  # 强制文件目录
            )

            if download_result.success:
                downloaded_files = download_result.data.get('files', [])
                logger.info(f"成功下载 {len(downloaded_files)} 个强制文件")
                for f in downloaded_files:
                    logger.debug(f"  - {f.get('filename')} -> {f.get('local_path')}")
            else:
                logger.error(f"强制文件下载失败: {download_result.error}")

        # 下载可选参考的文件到 library_refs/
        if optional_files:
            file_ids = [f['file_id'] for f in optional_files]
            logger.info(f"下载可选文件到 library_refs/: {file_ids}")

            download_result = mcp_tools.process_user_uploaded_files(
                file_ids=file_ids,
                backend_url="http://localhost:5000",
                target_subdir="library_refs"  # 可选文件目录
            )

            if download_result.success:
                downloaded_files = download_result.data.get('files', [])
                logger.info(f"成功下载 {len(downloaded_files)} 个可选文件")
                for f in downloaded_files:
                    logger.debug(f"  - {f.get('filename')} -> {f.get('local_path')}")
            else:
                logger.error(f"可选文件下载失败: {download_result.error}")

    except Exception as e:
        logger.error(f"预下载用户文件时发生异常: {e}", exc_info=True)


# 辅助函数：构建增强的查询文本
def _build_enhanced_query(query_text: str, user_files_data: List[Dict[str, str]]) -> str:
    """构建包含用户文件信息的增强查询文本"""
    if not user_files_data:
        return query_text

    # 检测用户query的语言(使用30%阈值)
    import re
    _zh_count = len(re.findall(r'[\u4e00-\u9fff]', query_text))
    _total_chars = len(query_text.strip())
    _is_chinese = (_zh_count / max(_total_chars, 1)) > 0.3

    # 分离强制文件和可选文件
    mandatory_files = [f for f in user_files_data if f.get('type') == 'mandatory']
    optional_files = [f for f in user_files_data if f.get('type') == 'optional']

    file_info_text = ""

    # 添加强制使用的文件信息(根据用户query语言选择提示语言)
    if mandatory_files:
        if _is_chinese:
            file_info_text += "\n\n【用户强制要求使用的文件（必须在报告中使用）】：\n"
        else:
            file_info_text += "\n\n[User-Required Files (Must be used in the report)]:\n"
        for i, file_info in enumerate(mandatory_files, 1):
            if _is_chinese:
                file_info_text += f"{i}. ./user_uploads/{file_info['filename']} (文件ID: {file_info['file_id']})\n"
            else:
                file_info_text += f"{i}. ./user_uploads/{file_info['filename']} (File ID: {file_info['file_id']})\n"
        if _is_chinese:
            file_info_text += "\n这些文件必须被分析和引用。"
        else:
            file_info_text += "\nThese files must be analyzed and cited."

    # 添加可选参考的文件信息(根据用户query语言选择提示语言)
    if optional_files:
        if _is_chinese:
            file_info_text += "\n\n【用户提供的可选参考文件（根据相关性自行判断是否使用）】：\n"
        else:
            file_info_text += "\n\n[Optional Reference Files (Use based on relevance)]:\n"
        for i, file_info in enumerate(optional_files, 1):
            if _is_chinese:
                file_info_text += f"{i}. ./library_refs/{file_info['filename']} (文件ID: {file_info['file_id']})\n"
            else:
                file_info_text += f"{i}. ./library_refs/{file_info['filename']} (File ID: {file_info['file_id']})\n"
        if _is_chinese:
            file_info_text += "\n这些文件可以作为参考资料，与网络检索结果一起评估相关性后选择使用。"
        else:
            file_info_text += "\nThese files can be used as references, evaluate relevance with web search results before using."

    if _is_chinese:
        file_info_text += "\n\n请使用 document_extract 工具分析文件内容，并进行网络检索来补充最新信息，确保报告内容的全面性和时效性。\n"
    else:
        file_info_text += "\n\nPlease use the document_extract tool to analyze file content and perform web searches to supplement with latest information, ensuring comprehensive and up-to-date report content.\n"
    
    return file_info_text + query_text


def process_single_query(query_data, task_id: Optional[str] = None, username: str = "用户",
                         skip_task_creation: bool = False, frontend_session_id: Optional[str] = None,
                         human_in_loop: bool = False, user_outline: Optional[str] = None):
    """处理单个查询（独立进程，使用持久化工作区）"""
    query_text, query_index, user_files_data, search_sources_dict = query_data
    process_id = os.getpid()
    if not task_id:
        task_id = f"req_{int(time.time() * 1000)}_{query_index}"  # 生成唯一请求ID

    # 创建并注册任务（如果尚未在调用方创建）
    if not skip_task_creation:
        task_manager.create_task(task_id, query_text)
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)

    # 使用持久化工作区（而非临时目录）
    # 统一使用项目根目录的 workspaces
    current_file = Path(__file__).resolve()  # cli/a.py
    project_root = None

    # 向上查找包含 app.py 的目录（项目根目录）
    for parent in [current_file.parent] + list(current_file.parents):
        if (parent / "app.py").exists():
            project_root = parent
            break

    # 如果找不到 app.py，使用当前工作目录
    if project_root is None:
        project_root = Path.cwd()

    # 统一使用项目根目录下的 workspaces
    base_workspaces = project_root / "workspaces"
    base_workspaces.mkdir(exist_ok=True, parents=True)

    # 生成 session_id（使用 UUID）
    session_id = str(uuid.uuid4())
    workspace_path = base_workspaces / session_id
    workspace_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"[WORKSPACE] session_id: {session_id}")
    logger.info(f"[WORKSPACE] workspace initialized at: {workspace_path.resolve()}")
    
    # 保存 task_id 到 session_id 的映射（用于查找workspace）
    if task_id:
        task_session_mapping[task_id] = session_id

    try:
        app_config = get_config()
        sub_agent_configs = {
            "information_seeker": {
                "model": app_config.model_name,
                **({"max_iterations": app_config.information_seeker_max_iterations} if app_config.information_seeker_max_iterations else {})
            },
            "writer": {
                "model": app_config.model_name,
                **({"max_iterations": app_config.writer_max_iterations} if app_config.writer_max_iterations else {})
            }
        }

        # 设置环境变量，让 Agent 使用已创建的 workspace
        os.environ['AGENT_SESSION_ID'] = session_id
        os.environ['AGENT_WORKSPACE_PATH'] = str(workspace_path)
        
        # Human in the loop 模式：通过 workspace 文件传递状态
        os.environ['HUMAN_IN_LOOP'] = 'true' if human_in_loop else 'false'
        os.environ.pop('HUMAN_IN_LOOP_PHASE2', None)  # 阶段1 确保不设置 PHASE2 标志
        
        # 创建 .human_in_loop 标记文件，供 MCP 工具读取
        human_in_loop_file = workspace_path / '.human_in_loop'
        if human_in_loop:
            with open(human_in_loop_file, 'w', encoding='utf-8') as f:
                f.write('true')
            logger.info(f"[HITL DEBUG] 已创建标记文件: {human_in_loop_file}")
            logger.info(f"[HITL DEBUG] workspace_path={workspace_path}, session_id={session_id}")
        else:
            # 确保不存在旧的标记文件
            if human_in_loop_file.exists():
                human_in_loop_file.unlink()
        
        if user_outline:
            # 如果有用户确认的大纲，写入文件供 Agent 使用
            outline_file = workspace_path / '.user_outline'
            with open(outline_file, 'w', encoding='utf-8') as f:
                f.write(user_outline)
            os.environ['USER_OUTLINE_PATH'] = str(outline_file)
        else:
            os.environ.pop('USER_OUTLINE_PATH', None)
        
        # 设置搜索源偏好
        if search_sources_dict:
            os.environ['SEARCH_SOURCE_WEBSEARCH'] = str(search_sources_dict.get('websearch', False))
            os.environ['SEARCH_SOURCE_PUBMED'] = str(search_sources_dict.get('pubmed', False))
            os.environ['SEARCH_SOURCE_ARXIV'] = str(search_sources_dict.get('arxiv', False))
            os.environ['SEARCH_SOURCE_GOOGLE_SCHOLAR'] = str(search_sources_dict.get('google_scholar', False))
            os.environ['SEARCH_SOURCE_SCIHUB'] = str(search_sources_dict.get('scihub', False))
            logger.info(f"[SEARCH_SOURCES] WebSearch: {search_sources_dict.get('websearch', False)}, "
                       f"PubMed: {search_sources_dict.get('pubmed', False)}, "
                       f"arXiv: {search_sources_dict.get('arxiv', False)}, "
                       f"GoogleScholar: {search_sources_dict.get('google_scholar', False)}, "
                       f"SciHub: {search_sources_dict.get('scihub', False)}, "
                      )

        agent = create_planner_agent(
            agent_name=f"PlannerAgent",
            model=app_config.model_name,
            max_iterations=app_config.planner_max_iterations or 40,
            sub_agent_configs=sub_agent_configs,
            task_id=task_id
        )
        # 设置取消令牌
        cancellation_token = task_manager.get_cancellation_token(task_id)
        if cancellation_token:
            agent.set_cancellation_token(cancellation_token)
        
        # 设置进度推送回调
        if hasattr(agent, 'set_progress_callback'):
            agent.set_progress_callback(send_progress_update)

        # 下载用户文件到工作区
        _download_user_files(user_files_data, workspace_path)

        # 将username写入workspace的配置文件，避免环境变量冲突
        username_file = workspace_path / '.username'
        with open(username_file, 'w', encoding='utf-8') as f:
            f.write(username)

        # 在开始执行前检查任务是否已被取消
        if task_manager.is_task_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled before execution")
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            raise HTTPException(status_code=499, detail="Task was cancelled by user")

        # 【关键修复】先基于原始query检测语言，避免enhanced_query中的文件提示文本影响语言判断
        import re
        _zh_count = len(re.findall(r'[\u4e00-\u9fff]', query_text))
        _total_chars = len(query_text.strip())
        _is_chinese_query = (_zh_count / max(_total_chars, 1)) > 0.3
        # 将语言标志传递给agent，确保在execute_task之前就设置好
        agent._is_chinese_query = _is_chinese_query
        logger.info(f"[语言检测] 原始query: {query_text[:100]}, 中文字符数: {_zh_count}, 总字符数: {_total_chars}, 判定为中文: {_is_chinese_query}")
        
        # 构建增强的查询文本
        enhanced_query = _build_enhanced_query(query_text, user_files_data)

        # 再次检查取消状态
        if task_manager.is_task_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled before agent execution")
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            raise HTTPException(status_code=499, detail="Task was cancelled by user")

        start_time = time.time()
        response = agent.execute_task(enhanced_query + " /no_think")
        execution_time = time.time() - start_time

        # 检查是否被取消（通过agent响应或直接检查状态）
        if (task_manager.is_task_cancelled(task_id) or 
            (hasattr(response, 'error') and response.error and "cancelled" in str(response.error).lower())):
            logger.info(f"Task {task_id} was cancelled during or after execution")
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED, error=getattr(response, 'error', 'Task cancelled'))
            raise HTTPException(status_code=499, detail="Task was cancelled by user")

        # Human in the loop: 检查是否需要等待用户确认大纲
        outline_pending_file = workspace_path / '.outline_pending'
        if human_in_loop and outline_pending_file.exists():
            logger.info(f"[HITL] 检测到 .outline_pending 文件，任务 {task_id} 等待用户确认大纲")
            
            # 读取大纲内容
            outline_content = ""
            outline_file = workspace_path / '.outline_content'
            if outline_file.exists():
                with open(outline_file, 'r', encoding='utf-8') as f:
                    outline_content = f.read()
            
            # 读取 reasoning 内容
            reasoning_content = ""
            reasoning_file = workspace_path / '.reasoning_content'
            if reasoning_file.exists():
                with open(reasoning_file, 'r', encoding='utf-8') as f:
                    reasoning_content = f.read()
            
            # 保存待确认状态（加锁保护）
            with pending_outline_lock:
                pending_outline_confirmations[task_id] = {
                    'session_id': session_id,
                    'workspace_path': str(workspace_path),
                    'outline_content': outline_content,
                    'reasoning_content': reasoning_content,
                    'query_data': query_data,
                    'username': username,
                    'frontend_session_id': frontend_session_id,
                    'task_id': task_id,
                    'status': 'waiting_for_confirm',
                    # HITL耗时拆分：Phase1（到大纲待确认为止）与确认等待时长
                    'phase1_execution_time': execution_time,
                    'outline_ready_at': time.time()
                }
            
            # 更新任务进度，通知前端等待大纲确认
            task_manager.update_task_progress(task_id, {
                'status': 'waiting_for_outline_confirm',
                'outline_content': outline_content,
                'reasoning_content': reasoning_content,
                'session_id': session_id
            })
            
            # 通过 SSE 发送进度更新
            send_progress_update(task_id, {
                'type': 'outline_pending',
                'message': '大纲已生成，等待用户确认',
                'outline_content': outline_content,
                'reasoning_content': reasoning_content,
                'session_id': session_id
            })
            
            logger.info(
                f"[HITL] 任务 {task_id} 已暂停，等待用户确认大纲 "
                f"(outline长度={len(outline_content)}, phase1耗时={execution_time:.2f}s)"
            )
            return  # 暂停执行，等待用户确认后通过 Phase 2 继续

        # 读取最终报告内容（在更新任务状态之前读取，以便保存到result中）
        final_report_content = None
        report_relative_path = None
        try:
            # 尝试读取 final_report.md
            final_report_path = workspace_path / "report" / "final_report.md"
            if final_report_path.exists():
                with open(final_report_path, 'r', encoding='utf-8') as f:
                    final_report_content = f.read()
                report_relative_path = "report/final_report.md"
                logger.info(f"成功读取最终报告: {final_report_path} (大小: {len(final_report_content)} 字符)")
                
                # 计算报告字数（中文字符+英文单词数）
                word_count = 0
                try:
                    import re
                    # 统计中文字符数
                    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', final_report_content))
                    # 统计英文单词数
                    english_words = len(re.findall(r'\b[a-zA-Z]+\b', final_report_content))
                    word_count = chinese_chars + english_words
                    logger.info(f"报告字数统计: 中文字符={chinese_chars}, 英文单词={english_words}, 总计={word_count}")
                except Exception as e:
                    logger.warning(f"计算报告字数失败: {e}")

                # 统计 InfoSeeker 全部检索次数（网页搜索 + 学术搜索 + 网页抓取，失败的也算）
                total_search_count = 0
                try:
                    import json
                    tool_call_logs_dir = workspace_path / "tool_call_logs"
                    if tool_call_logs_dir.is_dir():
                        for log_file in tool_call_logs_dir.glob("tool_calls_*.jsonl"):
                            try:
                                with open(log_file, 'r', encoding='utf-8') as lf:
                                    for line in lf:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        try:
                                            rec = json.loads(line)
                                        except Exception:
                                            continue

                                        tool_name = rec.get('tool_name')
                                        
                                        # 统计 batch_web_search 的查询次数
                                        if tool_name == 'batch_web_search':
                                            input_args = rec.get('input_args') or {}
                                            queries = input_args.get('queries') or []
                                            if isinstance(queries, list):
                                                total_search_count += len(queries)
                                        
                                        # 统计 url_crawler 的抓取次数
                                        elif tool_name == 'url_crawler':
                                            input_args = rec.get('input_args') or {}
                                            documents = input_args.get('documents') or []
                                            if isinstance(documents, list):
                                                total_search_count += len(documents)
                                        
                                        # 统计学术搜索工具（每次调用计1次）
                                        elif tool_name in ['arxiv_search', 'search_pubmed_key_words', 'search_pubmed_advanced', 
                                                          'medrxiv_search', 'springer_search', 'get_pubmed_article', 
                                                          'arxiv_read_paper', 'medrxiv_read_paper', 'springer_get_article',
                                                          'scihub_search', 'scihub_get_paper',
                                                          'google_scholar_search', 'advanced_google_scholar_search', 'google_scholar_get_paper']:
                                            total_search_count += 1
                            except Exception as e:
                                logger.warning(f"解析tool_call_logs失败: {log_file} - {e}")

                    logger.info(f"InfoSeeker检索统计: 总检索次数={total_search_count} (网页搜索+学术搜索+网页抓取)")
                except Exception as e:
                    logger.warning(f"统计InfoSeeker检索次数失败: {e}")
                
                # 在报告末尾添加统计信息
                try:
                    # 检测用户查询语言，决定统计信息的语言
                    import re
                    query_zh_count = len(re.findall(r'[\u4e00-\u9fff]', query_text))
                    query_total_chars = len(query_text.strip())
                    # 只有当中文字符占比超过30%时才判定为中文查询
                    is_chinese_query = (query_zh_count / max(query_total_chars, 1)) > 0.3
                    
                    # 添加换页符，使统计信息显示在新的一页
                    if is_chinese_query:
                        stats_section = f"\n\n<div style=\"page-break-before: always;\"></div>\n\n## 报告统计信息\n\n"
                        stats_section += f"- 报告字数: {word_count:,} 字\n"
                        stats_section += f"- 生成耗时: {execution_time:.2f} 秒 ({execution_time/60:.1f} 分钟)\n"
                        stats_section += f"- 网站检索: {total_search_count:,} 次\n"
                    else:
                        stats_section = f"\n\n<div style=\"page-break-before: always;\"></div>\n\n## Report Statistics\n\n"
                        stats_section += f"- Word Count: {word_count:,} words\n"
                        stats_section += f"- Generation Time: {execution_time:.2f} seconds ({execution_time/60:.1f} minutes)\n"
                        stats_section += f"- Web Searches: {total_search_count:,} times\n"
                    
                    # 将统计信息追加到报告内容
                    final_report_content_with_stats = final_report_content + stats_section
                    
                    # 写回文件
                    with open(final_report_path, 'w', encoding='utf-8') as f:
                        f.write(final_report_content_with_stats)
                    
                    # 更新final_report_content为包含统计信息的版本
                    final_report_content = final_report_content_with_stats
                    logger.info(f"已在报告末尾添加统计信息")
                    
                    # 异步生成PDF文件（避免阻塞主流程）
                    def generate_pdf_async():
                        try:
                            from src.tools.mcp_tools import generate_pdf_with_reportlab
                            
                            pdf_path = Path(final_report_path).parent.parent / "final_report.pdf"
                            success = generate_pdf_with_reportlab(final_report_content_with_stats, pdf_path)
                            
                            if success:
                                logger.info(f"[异步] 成功生成PDF文件（包含统计信息）: {pdf_path}")
                            else:
                                logger.warning(f"[异步] PDF生成失败")
                        except Exception as pdf_error:
                            logger.warning(f"[异步] 生成PDF失败: {pdf_error}")
                    
                    # 提交到线程池异步执行
                    import threading
                    pdf_thread = threading.Thread(target=generate_pdf_async, daemon=True)
                    pdf_thread.start()
                    logger.info(f"已提交PDF生成任务到后台线程")
                        
                except Exception as e:
                    logger.warning(f"添加统计信息到报告失败: {e}")
                
                # 自动存储报告到数据库
                try:
                    if frontend_session_id:
                        # 从报告内容中提取标题
                        report_title = "研究报告"  # 默认值
                        try:
                            import re
                            # 方法1: 查找第一个 # 标题
                            title_match = re.search(r'^#\s+(.+?)$', final_report_content, re.MULTILINE)
                            if title_match:
                                report_title = title_match.group(1).strip()
                            else:
                                # 方法2: 查找第一行非空内容作为标题
                                lines = [line.strip() for line in final_report_content.split('\n') if line.strip()]
                                if lines:
                                    report_title = re.sub(r'^#+\s*', '', lines[0]).strip()
                            
                            # 限制标题长度（最多50个字符）
                            if len(report_title) > 50:
                                report_title = report_title[:50] + '...'
                        except Exception as e:
                            logger.warning(f"提取报告标题失败: {e}")
                        
                        store_url = "http://localhost:5000/api/chat/messages"
                        store_data = {
                            "session_id": frontend_session_id,
                            "from_who": "ai",
                            "content": final_report_content,
                            "round": 1,
                            "uuid": session_id,  # 使用workspace session_id作为uuid
                            "has_report": 1,
                            "report_title": report_title
                        }
                        
                        http_response = requests.post(store_url, json=store_data, timeout=30)
                        if http_response.status_code == 201:
                            logger.info(f"成功存储报告到数据库: session_id={frontend_session_id}, uuid={session_id}")
                        else:
                            logger.error(f"存储报告失败: {http_response.status_code}, {http_response.text}")
                    else:
                        logger.warning("未提供frontend_session_id，跳过自动存储")
                except Exception as store_error:
                    logger.error(f"自动存储报告到数据库失败: {store_error}")
            else:
                logger.warning(f"最终报告文件不存在: {final_report_path}")
                # 【降级兜底B】尝试从 part_*.md 文件自动合并生成 final_report.md
                # 采用分级降级策略：根据章节数决定是否合并以及如何标注
                try:
                    import re as _re_fallback
                    report_dir = final_report_path.parent
                    if report_dir.exists():
                        part_files = sorted(
                            report_dir.glob("part_*.md"),
                            key=lambda p: int(_re_fallback.search(r'part_(\d+)', p.name).group(1))
                            if _re_fallback.search(r'part_(\d+)', p.name) else 0
                        )
                        part_count = len(part_files)
                        
                        if part_count == 0:
                            logger.warning("[降级兜底B] 无可用章节，跳过合并")
                        elif part_count < 3:
                            # 内容太少，标注为"草稿"并建议重试
                            logger.warning(f"[降级兜底B] 仅 {part_count} 个章节，标注为草稿（建议用户重试）")
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
                                final_report_path.write_text(final_content, encoding='utf-8')
                                final_report_content = final_content
                                report_relative_path = "report/final_report.md"
                                logger.info(f"[降级兜底B] 已保存草稿 ({len(final_content)} 字符)")
                        else:
                            # >=3个章节，基本可用，添加警告说明
                            logger.info(f"[降级兜底B] 成功合并 {part_count} 个章节（添加警告说明）")
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
                                final_report_path.write_text(final_content, encoding='utf-8')
                                final_report_content = final_content
                                report_relative_path = "report/final_report.md"
                                logger.info(f"[降级兜底B] 成功合并为 final_report.md ({len(final_content)} 字符)")
                except Exception as fallback_err:
                    logger.warning(f"[降级兜底B] 自动合并 part_*.md 失败: {fallback_err}")
                
                # 如果降级兜底B成功合并了报告，自动存储到数据库
                if final_report_content:
                    try:
                        if frontend_session_id:
                            import re as _re_title
                            report_title = "研究报告"
                            title_match = _re_title.search(r'^#\s+(.+?)$', final_report_content, _re_title.MULTILINE)
                            if title_match:
                                report_title = title_match.group(1).strip()
                            if len(report_title) > 50:
                                report_title = report_title[:50] + '...'
                            store_url = "http://localhost:5000/api/chat/messages"
                            store_data = {
                                "session_id": frontend_session_id,
                                "from_who": "ai",
                                "content": final_report_content,
                                "round": 1,
                                "uuid": session_id,
                                "has_report": 1,
                                "report_title": report_title + "（降级恢复）"
                            }
                            http_response = requests.post(store_url, json=store_data, timeout=30)
                            if http_response.status_code == 201:
                                logger.info(f"[降级兜底B] 成功存储降级恢复报告到数据库")
                            else:
                                logger.error(f"[降级兜底B] 存储降级恢复报告失败: {http_response.status_code}")
                    except Exception as store_err:
                        logger.error(f"[降级兜底B] 存储降级恢复报告异常: {store_err}")

                # 没有完整报告时，尝试保存 task_summary（简单问题的回复）或错误提示
                if not final_report_content:
                    try:
                        content_to_store = None
                        
                        # 情况1: planner成功完成，有task_summary
                        if frontend_session_id and response.success and response.result:
                            task_summary = response.result.get('task_summary', '')
                            if task_summary:
                                import re
                                # 提取"完整答案"部分
                                answer_match = re.search(r'## 完整答案[\s\S]*?(?=任务已圆满完成)', task_summary)
                                if answer_match:
                                    content_to_store = answer_match.group(0).replace('## 完整答案\n', '').strip() + '\n\n任务已圆满完成'
                                else:
                                    content_to_store = task_summary
                                
                                # 清理task_summary中不适合展示的内容
                                if content_to_store:
                                    # 移除"文件存储位置"段落（从标题到下一个标题或末尾）
                                    content_to_store = re.sub(
                                        r'(?:#{1,4}\s*)?文件存储位置[\s\S]*?(?=(?:#{1,4}\s|\Z))',
                                        '', content_to_store
                                    ).strip()
                                    # 替换"最终文章路径"相关内容为友好提示
                                    content_to_store = re.sub(
                                        r'最终文章路径[：:]\s*.+',
                                        '⚠️ 由于写作阶段未完成，最终文章尚未生成。建议重新提问以获取完整报告。',
                                        content_to_store
                                    )
                        
                        # 情况2: planner未成功（如超时/超过最大迭代次数），存储错误提示
                        if not content_to_store and frontend_session_id and not response.success:
                            error_msg = response.error or '任务执行未完成'
                            content_to_store = f"⚠️ 任务未能完成：{error_msg}\n\n信息收集阶段可能已部分完成，但由于处理步骤限制，未能生成最终报告。请尝试重新提问。"
                            logger.warning(f"[降级兜底C] planner未成功完成 (success=False, error={error_msg})，存储错误提示到数据库")
                        
                        if content_to_store and frontend_session_id:
                            store_url = "http://localhost:5000/api/chat/messages"
                            store_data = {
                                "session_id": frontend_session_id,
                                "from_who": "ai",
                                "content": content_to_store,
                                "round": 1,
                                "uuid": session_id,
                                "has_report": 0,
                                "report_title": ""
                            }
                            
                            http_response = requests.post(store_url, json=store_data, timeout=30)
                            if http_response.status_code != 201:
                                logger.error(f"存储回复失败: {http_response.status_code}")
                            else:
                                logger.info(f"成功存储回复到数据库: session_id={frontend_session_id}")
                    except Exception as e:
                        logger.error(f"存储回复异常: {e}")
        except Exception as e:
            logger.error(f"读取最终报告失败: {e}")

        # 构建完整的结果对象
        result_data = {
            'success': response.success,
            'query': query_text,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': session_id,
            'task_id': task_id,
            'planner_result': response.result if response.success else None,
            'planner_error': response.error if not response.success else None,
            'planner_reasoning_trace': getattr(response, 'reasoning_trace', []),
            'planner_iterations': response.iterations,
            'planner_execution_time': execution_time,
            'planner_agent_name': 'PlannerAgent',
            'section_writer_responses': [],
            'final_report': final_report_content,
            'report_path': report_relative_path
        }
        
        # 【关键修复】更新任务状态为完成，并保存完整的结果对象（包含final_report）
        # 这样前端轮询时可以获取到报告内容
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=result_data)
        
        # 发送任务完成的SSE进度消息，让前端移除进度容器
        send_progress_update(task_id, {
            'type': 'completed',
            'message': '任务完成',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': session_id,
            'report_path': report_relative_path,
            'final_report': final_report_content
        })
        
        return result_data
    except Exception as e:
        task_manager.update_task_status(task_id, TaskStatus.FAILED, error=str(e))
        
        # 发送任务失败的SSE进度消息
        send_progress_update(task_id, {
            'type': 'error',
            'message': str(e),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # 【关键修复】检查是否是用户主动取消（499错误），如果是则不存储错误消息
        is_user_cancelled = isinstance(e, HTTPException) and e.status_code == 499
        
        # 【关键修复】将错误消息写入数据库，让前端轮询能感知任务失败
        # 但如果是用户主动取消（499），则跳过存储，避免显示多余的错误提示
        try:
            if frontend_session_id and not is_user_cancelled:
                # 检测用户查询语言，决定错误提示的语言
                import re
                query_zh_count = len(re.findall(r'[\u4e00-\u9fff]', query_text))
                is_chinese = query_zh_count > len(query_text) * 0.3
                
                error_message = (
                    f"抱歉，处理您的请求时遇到了问题：{str(e)}\n\n请尝试重新提问或简化您的问题。" 
                    if is_chinese 
                    else f"Sorry, an error occurred while processing your request: {str(e)}\n\nPlease try rephrasing or simplifying your question."
                )
                
                store_url = "http://localhost:5000/api/chat/messages"
                store_data = {
                    "session_id": frontend_session_id,
                    "from_who": "ai",
                    "content": error_message,
                    "round": 1,
                    "uuid": session_id,
                    "has_report": 0,
                    "report_title": ""
                }
                
                http_response = requests.post(store_url, json=store_data, timeout=30)
                if http_response.status_code == 201:
                    logger.info(f"成功存储错误消息到数据库: session_id={frontend_session_id}")
                else:
                    logger.error(f"存储错误消息失败: {http_response.status_code}, {http_response.text}")
            elif is_user_cancelled:
                logger.info(f"用户主动取消任务 (499)，跳过错误消息存储: task_id={task_id}")
        except Exception as store_error:
            logger.error(f"存储错误消息到数据库失败: {store_error}")
        
        # 返回符合 QueryResponse 模型的字典结构
        return {
            'success': False,
            'query': query_text,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': session_id,
            'task_id': task_id,
            'planner_result': None,
            'planner_error': str(e),
            'planner_reasoning_trace': [],
            'planner_iterations': 0,
            'planner_execution_time': 0,
            'planner_agent_name': 'PlannerAgent',
            'section_writer_responses': [],
            'final_report': None,
            'report_path': None
        }


# 批量处理任务（用于后台执行）
def process_batch_task(
        queries: List[str],
        max_workers: Optional[int],
        batch_id: str,
        results_store: Dict[str, BatchResponse]
):
    """批量处理查询并存储结果"""
    query_data = [(q, idx) for idx, q in enumerate(queries)]
    max_workers = max_workers or min(mp.cpu_count(), len(queries), 4)
    results = []

    # 【HITL修复】使用 ThreadPoolExecutor 而不是 ProcessPoolExecutor
    # 因为 pending_outline_confirmations 需要在所有任务间共享状态
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_query = {executor.submit(process_single_query, qd): qd for qd in query_data}
        for future in as_completed(future_to_query):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                qd = future_to_query[future]
                results.append({
                    'success': False,
                    'query': qd[0],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'session_id': '',
                    'task_id': None,
                    'planner_result': None,
                    'planner_error': str(e),
                    'planner_reasoning_trace': [],
                    'planner_iterations': 0,
                    'planner_execution_time': 0,
                    'planner_agent_name': 'PlannerAgent',
                    'section_writer_responses': [],
                    'final_report': None,
                    'report_path': None
                })

    # 排序并更新结果存储
    results.sort(key=lambda x: x['query_index'])
    results_store[batch_id] = BatchResponse(
        batch_id=batch_id,
        status="completed",
        total_queries=len(queries),
        completed_count=len(results),
        results=[QueryResponse(**r) for r in results]
    )


# 队列处理函数
def process_task_queue():
    """后台线程持续处理队列中的任务"""
    logger.info("Queue processor started")
    loop_count = 0
    while True:
        try:
            loop_count += 1
            # 每30次循环输出一次状态日志（约30秒）
            if loop_count % 30 == 0:
                queued_count = task_manager.get_queued_tasks_count()
                running_count = task_manager.get_running_tasks_count()
                logger.info(f"Queue processor heartbeat: running={running_count}, queued={queued_count}, max={MAX_CONCURRENT_TASKS}")
            
            # 检查是否有空闲槽位
            running_count = task_manager.get_running_tasks_count()
            if running_count < MAX_CONCURRENT_TASKS:
                # 【关键修复】先收集需要处理的任务信息，然后释放锁再进行状态更新
                # 避免在持有锁的情况下调用update_task_status导致死锁
                cancelled_task_ids = []
                next_task_id = None
                next_task_params = None
                should_start_task = False
                
                with task_manager._tasks_lock:
                    # 收集已取消的队列任务ID
                    for task in task_manager._tasks.values():
                        if task.status == TaskStatus.QUEUED and task.is_cancelled():
                            cancelled_task_ids.append(task.task_id)
                    
                    # 获取未取消的队列任务
                    queued_tasks = sorted(
                        [task for task in task_manager._tasks.values() 
                         if task.status == TaskStatus.QUEUED and not task.is_cancelled()],
                        key=lambda t: t.created_at
                    )
                    
                    if queued_tasks:
                        next_task = queued_tasks[0]
                        if not next_task.is_cancelled():
                            next_task_id = next_task.task_id
                            next_task_params = next_task.progress.get('params')
                            
                            # 【关键修复】只有当params存在且包含query_data时才启动任务
                            if next_task_params is not None and 'query_data' in next_task_params:
                                should_start_task = True
                                # 直接在锁内更新状态（不调用update_task_status避免死锁）
                                next_task.status = TaskStatus.RUNNING
                                next_task.updated_at = time.time()
                                logger.info(f"Task {next_task_id} status changed to RUNNING")
                            else:
                                # params无效，记录错误并标记任务失败
                                logger.error(f"Task {next_task_id} has invalid params: {next_task_params}")
                                next_task.status = TaskStatus.FAILED
                                next_task.error = "任务参数无效"
                                next_task.updated_at = time.time()
                
                # 锁已释放，现在安全地更新已取消任务的状态
                for task_id in cancelled_task_ids:
                    logger.info(f"Removing cancelled queued task {task_id}")
                    task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
                
                # 启动下一个任务
                if should_start_task and next_task_id and next_task_params:
                    logger.info(f"Starting queued task {next_task_id} with params keys: {list(next_task_params.keys())}")
                    executor.submit(
                        execute_queued_task,
                        next_task_id,
                        next_task_params
                    )
            
            # 更新所有队列任务的位置
            task_manager.update_queue_positions()
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(1)
        except Exception as e:
            logger.error(f"Queue processor error: {e}", exc_info=True)
            time.sleep(5)


def execute_queued_task(task_id: str, params: Dict[str, Any]):
    """执行队列中的任务"""
    try:
        # 在开始执行前检查任务是否已被取消
        if task_manager.is_task_cancelled(task_id):
            logger.info(f"Queued task {task_id} was cancelled before execution started")
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
            return
        
        logger.info(f"Executing queued task {task_id}")
        result = process_single_query(
            params['query_data'],
            task_id=task_id,
            username=params.get('username', '用户'),
            skip_task_creation=True,
            frontend_session_id=params.get('frontend_session_id'),
            human_in_loop=params.get('human_in_loop', False)
        )
        
        # 【关键修复】任务完成后保存结果到TaskManager，以便前端轮询获取
        if result:
            if result.get('success', False):
                logger.info(f"Queued task {task_id} completed successfully")
                task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
            else:
                error_msg = result.get('planner_error', '任务执行失败')
                logger.warning(f"Queued task {task_id} failed: {error_msg}")
                task_manager.update_task_status(task_id, TaskStatus.FAILED, result=result, error=error_msg)
        else:
            logger.warning(f"Queued task {task_id} returned no result")
            task_manager.update_task_status(task_id, TaskStatus.FAILED, error="任务未返回结果")
            
    except Exception as e:
        # 检查是否是因为任务取消导致的异常
        if task_manager.is_task_cancelled(task_id):
            logger.info(f"Queued task {task_id} was cancelled during execution")
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
        else:
            logger.error(f"Error executing queued task {task_id}: {e}", exc_info=True)
            task_manager.update_task_status(task_id, TaskStatus.FAILED, error=str(e))


# API端点实现
@app.post("/api/query", response_model=QueryResponse, summary="处理单个查询")
async def handle_single_query(request: SingleQueryRequest):
    """异步处理单个查询，支持高并发和用户文件上传"""
    import time
    request_received_time = time.time()

    # 生成唯一任务ID
    task_id = request.taskId
    
    logger.info(f"[API] 收到任务创建请求: task_id={task_id}, timestamp={request_received_time}")
    
    # 【并发控制】检查当前运行中的query数量
    running_count = task_manager.get_running_tasks_count()
    
    # 创建任务
    task_created_time = time.time()
    task_manager.create_task(task_id, request.query)
    logger.info(f"[API] 任务已创建: task_id={task_id}, 创建耗时={task_created_time - request_received_time:.3f}秒")

    # 准备任务参数
    user_files_data = []
    if request.user_files and len(request.user_files) > 0:
        for file in request.user_files:
            user_files_data.append({
                'file_id': file.file_id,
                'filename': file.filename,
                'type': 'mandatory'  # 标记为强制使用
            })

    # 准备可选参考的文件数据（从文档库选择）
    reference_files_data = []
    if request.reference_files and len(request.reference_files) > 0:
        for file in request.reference_files:
            reference_files_data.append({
                'file_id': file.file_id,
                'filename': file.filename,
                'type': 'optional'  # 标记为可选参考
            })

    # 合并所有文件数据，传递给处理函数
    all_files_data = user_files_data + reference_files_data
    
    # 准备搜索源数据
    search_sources_dict = None
    if request.search_sources:
        search_sources_dict = {
            'websearch': request.search_sources.websearch,
            'pubmed': request.search_sources.pubmed,
            'arxiv': request.search_sources.arxiv,
            'google_scholar': request.search_sources.google_scholar,
            'scihub': request.search_sources.scihub
        }

    # 判断是立即执行还是加入队列
    if running_count >= MAX_CONCURRENT_TASKS:
        # 超过并发限制，加入队列
        task_manager.update_task_status(task_id, TaskStatus.QUEUED)
        
        # 保存任务参数到progress中，供队列处理器使用
        task_manager.update_task_progress(task_id, {
            'params': {
                'query_data': (request.query, 0, all_files_data, search_sources_dict),
                'username': request.username,
                'frontend_session_id': request.frontend_session_id,
                'human_in_loop': request.human_in_loop
            }
        })
        
        # 更新队列位置
        task_manager.update_queue_positions()
        queue_position = task_manager.get_queue_position(task_id)
        
        logger.info(f"Task {task_id} queued at position {queue_position}")
        
        # 返回队列状态响应
        return {
            'success': False,
            'query': request.query,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': '',
            'task_id': task_id,
            'planner_result': None,
            'planner_error': f'任务已加入队列，当前排队位置: {queue_position}',
            'planner_reasoning_trace': [],
            'planner_iterations': 0,
            'planner_execution_time': 0,
            'planner_agent_name': 'QueueManager',
            'section_writer_responses': [],
            'final_report': None,
            'report_path': None
        }
    
    # 有空闲槽位，提交到后台执行（不等待完成）
    task_manager.update_task_status(task_id, TaskStatus.RUNNING)
    # 【关键修复】保存frontend_session_id到progress，供app.py交叉验证has_pending_task使用
    if request.frontend_session_id:
        task_manager.update_task_progress(task_id, {
            'params': {
                'frontend_session_id': request.frontend_session_id
            }
        })
    loop = asyncio.get_event_loop()
    
    # 使用线程池执行，避免阻塞事件循环
    if executor is None:
        raise HTTPException(status_code=500, detail="Server executor not initialized")

    # 提交任务到后台执行，不等待结果（去掉 await）
    loop.run_in_executor(
        executor,
        lambda: process_single_query((request.query, 0, all_files_data, search_sources_dict), task_id=task_id, username=request.username,
                                     skip_task_creation=True, frontend_session_id=request.frontend_session_id,
                                     human_in_loop=request.human_in_loop)
    )
    
    # 立即返回任务已接受的响应，不等待任务完成
    # 前端将通过 SSE 接收进度更新和最终结果
    logger.info(f"[API] 任务已提交到后台执行: task_id={task_id}")
    return {
        'success': True,
        'query': request.query,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'session_id': '',
        'task_id': task_id,
        'planner_result': None,
        'planner_error': None,
        'planner_reasoning_trace': [],
        'planner_iterations': 0,
        'planner_execution_time': 0,
        'planner_agent_name': 'TaskAccepted',
        'section_writer_responses': [],
        'final_report': None,
        'report_path': None
    }


@app.post("/api/query/sync", response_model=QueryResponse, summary="处理单个查询（同步模式）")
async def handle_single_query_sync(request: SingleQueryRequest):
    """同步处理单个查询，等待任务完成后返回完整结果（适用于 Postman/脚本调用）"""
    import time
    request_received_time = time.time()

    # 生成唯一任务ID
    task_id = request.taskId
    
    logger.info(f"[API-SYNC] 收到同步任务请求: task_id={task_id}, timestamp={request_received_time}")
    
    # 【并发控制】检查当前运行中的query数量
    running_count = task_manager.get_running_tasks_count()
    
    # 创建任务
    task_created_time = time.time()
    task_manager.create_task(task_id, request.query)
    logger.info(f"[API-SYNC] 任务已创建: task_id={task_id}, 创建耗时={task_created_time - request_received_time:.3f}秒")

    # 准备任务参数
    user_files_data = []
    if request.user_files and len(request.user_files) > 0:
        for file in request.user_files:
            user_files_data.append({
                'file_id': file.file_id,
                'filename': file.filename,
                'type': 'mandatory'
            })

    reference_files_data = []
    if request.reference_files and len(request.reference_files) > 0:
        for file in request.reference_files:
            reference_files_data.append({
                'file_id': file.file_id,
                'filename': file.filename,
                'type': 'optional'
            })

    all_files_data = user_files_data + reference_files_data
    
    search_sources_dict = None
    if request.search_sources:
        search_sources_dict = {
            'websearch': request.search_sources.websearch,
            'pubmed': request.search_sources.pubmed,
            'arxiv': request.search_sources.arxiv,
            'google_scholar': request.search_sources.google_scholar,
            'scihub': request.search_sources.scihub
        }

    # 判断是立即执行还是加入队列
    if running_count >= MAX_CONCURRENT_TASKS:
        # 超过并发限制，加入队列
        task_manager.update_task_status(task_id, TaskStatus.QUEUED)
        
        task_manager.update_task_progress(task_id, {
            'params': {
                'query_data': (request.query, 0, all_files_data, search_sources_dict),
                'username': request.username,
                'frontend_session_id': request.frontend_session_id
            }
        })
        
        task_manager.update_queue_positions()
        queue_position = task_manager.get_queue_position(task_id)
        
        logger.info(f"[API-SYNC] Task {task_id} queued at position {queue_position}")
        
        return {
            'success': False,
            'query': request.query,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': '',
            'task_id': task_id,
            'planner_result': None,
            'planner_error': f'任务已加入队列，当前排队位置: {queue_position}',
            'planner_reasoning_trace': [],
            'planner_iterations': 0,
            'planner_execution_time': 0,
            'planner_agent_name': 'QueueManager',
            'section_writer_responses': [],
            'final_report': None,
            'report_path': None
        }
    
    # 有空闲槽位，同步执行并等待完成
    task_manager.update_task_status(task_id, TaskStatus.RUNNING)
    # 【关键修复】保存frontend_session_id到progress，供app.py交叉验证has_pending_task使用
    if request.frontend_session_id:
        task_manager.update_task_progress(task_id, {
            'params': {
                'frontend_session_id': request.frontend_session_id
            }
        })
    loop = asyncio.get_event_loop()
    
    if executor is None:
        raise HTTPException(status_code=500, detail="Server executor not initialized")

    # 同步模式：等待任务完成
    result = await loop.run_in_executor(
        executor,
        lambda: process_single_query((request.query, 0, all_files_data, search_sources_dict), task_id=task_id, username=request.username,
                                     skip_task_creation=True, frontend_session_id=request.frontend_session_id)
    )
    
    logger.info(f"[API-SYNC] 任务执行完成: task_id={task_id}")
    query_history.append({
        "task_id": task_id,
        "request_id": result['session_id'],
        "query": request.query,
        "timestamp": result['timestamp'],
        "success": result['success'],
        "user_files_count": len(user_files_data),
        "reference_files_count": len(reference_files_data)
    })
    return result


@app.post("/api/batch", response_model=BatchResponse, summary="处理批量查询")
def handle_batch_query(request: BatchQueryRequest, background_tasks: BackgroundTasks):
    """处理批量查询，后台异步执行"""
    if not request.queries:
        raise HTTPException(status_code=400, detail="批量查询列表不能为空")

    # 生成唯一批次ID
    batch_id = f"batch_{int(time.time())}"
    # 初始化批次状态
    batch_results[batch_id] = BatchResponse(
        batch_id=batch_id,
        status="processing",
        total_queries=len(request.queries),
        completed_count=0
    )

    # 将批量处理任务添加到后台
    background_tasks.add_task(
        process_batch_task,
        queries=request.queries,
        max_workers=request.max_workers,
        batch_id=batch_id,
        results_store=batch_results
    )

    return batch_results[batch_id]


@app.get("/api/batch/{batch_id}", response_model=BatchResponse, summary="查询批量任务结果")
def get_batch_result(batch_id: str):
    """通过批次ID查询结果"""
    if batch_id not in batch_results:
        raise HTTPException(status_code=404, detail="批次ID不存在")
    return batch_results[batch_id]


@app.get("/api/concurrency", summary="获取当前并发状态")
async def get_concurrency_status():
    """
    返回当前运行中的query数量和队列状态，用于前端预检查
    
    Returns:
        running_queries: 当前运行中的query数量
        queued_queries: 当前排队中的query数量
        max_concurrent: 最大并发数阈值
        status: 状态标识 (available/queuing/busy)
    """
    running_count = task_manager.get_running_tasks_count()
    queued_count = task_manager.get_queued_tasks_count()

    if running_count >= MAX_CONCURRENT_TASKS:
        status = "busy"
    elif running_count >= 3:
        status = "queuing"
    else:
        status = "available"

    return {
        "running_queries": running_count,
        "queued_queries": queued_count,
        "max_concurrent": MAX_CONCURRENT_TASKS,
        "status": status
    }


@app.get("/api/queue/status", summary="获取队列详细状态")
async def get_queue_status():
    """
    获取队列的详细状态信息，包括所有排队任务
    
    Returns:
        running_count: 运行中任务数
        queued_count: 排队中任务数
        queued_tasks: 排队任务列表（包含task_id, query, position, created_at）
    """
    running_count = task_manager.get_running_tasks_count()
    queued_count = task_manager.get_queued_tasks_count()
    
    # 获取所有排队任务的详细信息
    with task_manager._tasks_lock:
        queued_tasks = sorted(
            [task for task in task_manager._tasks.values() if task.status == TaskStatus.QUEUED],
            key=lambda t: t.created_at
        )
        queued_tasks_info = [
            {
                "task_id": task.task_id,
                "query": task.query[:100] + "..." if len(task.query) > 100 else task.query,
                "position": i + 1,
                "created_at": task.created_at,
                "waiting_time": time.time() - task.created_at
            }
            for i, task in enumerate(queued_tasks)
        ]
    
    return {
        "running_count": running_count,
        "queued_count": queued_count,
        "max_concurrent": MAX_CONCURRENT_TASKS,
        "queued_tasks": queued_tasks_info
    }


@app.get("/api/status", summary="获取服务器状态")
def get_server_status():
    """返回服务器状态和统计信息"""
    running_count = task_manager.get_running_tasks_count()
    return {
        "status": "运行中",
        "concurrent_workers": executor._max_workers if executor and hasattr(executor, '_max_workers') else 8,
        "query_history_count": len(query_history),
        "active_batch_tasks": sum(1 for res in batch_results.values() if res.status == "processing"),
        "running_queries": running_count,
        "concurrency_status": "busy" if running_count >= 4 else ("queuing" if running_count >= 3 else "available")
    }


@app.get("/api/history", summary="获取查询历史")
def get_query_history(limit: int = 10):
    """返回最近的查询历史"""
    return {
        "total": len(query_history),
        "history": query_history[-limit:]
    }


# ==================== SSE进度推送相关 ====================

def send_progress_update(task_id: str, progress_data: dict):
    """
    发送进度更新到SSE流
    
    Args:
        task_id: 任务ID
        progress_data: 进度数据字典
    """
    logger.info(f"[SSE] send_progress_update called: task_id={task_id}, in_queues={task_id in progress_queues}")
    
    # 存储历史进度消息（除了心跳消息）
    if progress_data.get('type') != 'heartbeat':
        if task_id not in progress_history:
            progress_history[task_id] = []
        progress_history[task_id].append(progress_data)
        logger.info(f"[SSE] 存储历史进度: task_id={task_id}, 当前历史数量={len(progress_history[task_id])}")
    
    if task_id in progress_queues:
        try:
            progress_queues[task_id].put(progress_data)
            logger.info(f"[SSE] 发送进度更新: task_id={task_id}, type={progress_data.get('type')}, message={progress_data.get('message')}")
        except Exception as e:
            logger.error(f"[SSE] 发送进度更新失败: {e}")
    else:
        logger.warning(f"[SSE] task_id={task_id} 不在 progress_queues 中，无法发送进度")
        if progress_data.get('type') in ('completed', 'error', 'cancelled'):
            logger.error(
                f"[SSE] 终端事件可能未实时送达(无活跃SSE队列): task_id={task_id}, "
                f"type={progress_data.get('type')}。历史已写入 progress_history，客户端可 GET /api/task 或刷新后重连。"
            )


@app.get("/api/query/stream/{task_id}", summary="SSE流式推送任务进度")
async def stream_task_progress(task_id: str):
    """
    SSE端点：实时推送任务进度
    
    Args:
        task_id: 任务ID
        
    Returns:
        StreamingResponse: SSE事件流
    """
    
    async def event_generator():
        import time
        sse_start_time = time.time()
        
        # 为该任务创建进度队列
        progress_queue = queue.Queue()
        progress_queues[task_id] = progress_queue
        
        logger.info(f"[SSE] 客户端连接: task_id={task_id}, timestamp={sse_start_time}")
        
        # 先发送历史进度消息（如果存在）
        if task_id in progress_history:
            history_messages = progress_history[task_id]
            logger.info(f"[SSE] 重发历史进度: task_id={task_id}, 历史消息数量={len(history_messages)}")
            for msg in history_messages:
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                # 移除延迟，立即发送所有历史消息以加快显示速度
        
        # 等待任务创建（增加到15秒），允许前端先建立SSE连接再创建任务
        task_wait_count = 0
        max_wait_iterations = 30  # 15秒 (30次 * 0.5秒) - 增加等待时间
        while task_wait_count < max_wait_iterations:
            task_info = task_manager.get_task(task_id)
            if task_info:
                elapsed = time.time() - sse_start_time
                logger.info(f"[SSE] 任务已找到: task_id={task_id}, 等待时长={elapsed:.2f}秒")
                break
            elapsed = time.time() - sse_start_time
            logger.info(f"[SSE] 等待任务创建: task_id={task_id}, 尝试 {task_wait_count + 1}/{max_wait_iterations}, 已等待={elapsed:.2f}秒")
            await asyncio.sleep(0.5)
            task_wait_count += 1
        
        # 如果超时仍未找到任务，记录详细信息
        if task_wait_count >= max_wait_iterations:
            elapsed = time.time() - sse_start_time
            all_tasks = task_manager.get_all_tasks()
            logger.error(f"[SSE] 等待任务超时: task_id={task_id}, 总等待时长={elapsed:.2f}秒, 当前任务数量={len(all_tasks)}")
        
        try:
            while True:
                # 检查任务状态
                task_info = task_manager.get_task(task_id)
                if not task_info:
                    yield f"data: {json.dumps({'type': 'error', 'message': '任务不存在'}, ensure_ascii=False)}\n\n"
                    break
                
                # 从队列获取进度更新
                try:
                    progress_data = progress_queue.get(timeout=1)
                    yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                    
                    # 如果任务完成或失败，发送完成信号后退出
                    if progress_data.get('type') in ['completed', 'error', 'cancelled']:
                        logger.info(f"[SSE] 任务结束: task_id={task_id}, type={progress_data.get('type')}")
                        break
                        
                except queue.Empty:
                    # 发送心跳保持连接
                    yield f": heartbeat\n\n"
                
                await asyncio.sleep(0.5)
                
        except asyncio.CancelledError:
            logger.info(f"[SSE] 客户端断开连接: task_id={task_id}")
        except Exception as e:
            logger.error(f"[SSE] 流式推送异常: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # 仅移除本连接注册的队列。新 SSE 会覆盖 progress_queues[task_id]，
            # 若旧连接在 finally 里无条件 del，会误删新连接的队列，导致 Phase2 进度无法实时入队（仅写入 progress_history）。
            if progress_queues.get(task_id) is progress_queue:
                del progress_queues[task_id]
                logger.info(f"[SSE] 清理进度队列: task_id={task_id}")
            else:
                logger.info(
                    f"[SSE] 跳过清理进度队列（已被较新的SSE连接替换）: task_id={task_id}"
                )
            
            # 如果任务已完成，清理历史进度消息
            task_info = task_manager.get_task(task_id)
            if task_info and task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task_id in progress_history:
                    del progress_history[task_id]
                    logger.info(f"[SSE] 清理历史进度消息: task_id={task_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用nginx缓冲
        }
    )


@app.post("/api/task/{task_id}/cancel", summary="取消正在运行的任务")
async def cancel_task(task_id: str):
    """
    取消正在运行的任务
    
    Args:
        task_id: 任务ID
        
    Returns:
        取消结果
    """
    # 【关键修复】直接同步执行取消操作
    # cancel_task操作本身非常快（只是设置一个threading.Event），不需要放到线程池
    # 这样可以避免线程池阻塞导致的请求pending问题
    try:
        logger.info(f"Received cancel request for task {task_id}")
        success = task_manager.cancel_task(task_id)
        logger.info(f"Cancel task {task_id} result: {success}")
        
        if not success:
            task_info = task_manager.get_task(task_id)
            if task_info is None:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            else:
                return {
                    "success": False,
                    "message": "任务已经中断！",
                    "task_id": task_id,
                    "status": task_info.status.value
                }

        return {
            "success": True,
            "message": "任务中断成功！",
            "task_id": task_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "message": f"取消任务时发生错误: {str(e)}",
            "task_id": task_id
        }


@app.get("/api/task/{task_id}", summary="获取任务状态")
async def get_task_status(task_id: str):
    """
    获取任务状态和进度信息
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务状态信息
    """
    task_info = task_manager.get_task(task_id)

    if task_info is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return {
        "task_id": task_info.task_id,
        "query": task_info.query,
        "status": task_info.status.value,
        "created_at": task_info.created_at,
        "updated_at": task_info.updated_at,
        "progress": task_info.progress,
        "error": task_info.error,
        "result": task_info.result  # 返回完整结果，以便前端轮询获取
    }


@app.get("/api/tasks", summary="获取所有任务列表")
async def get_all_tasks():
    """
    获取所有任务的列表
    
    Returns:
        所有任务的信息
    """
    tasks = task_manager.get_all_tasks()
    running_count = task_manager.get_running_tasks_count()

    return {
        "total_tasks": len(tasks),
        "running_tasks": running_count,
        "tasks": list(tasks.values())
    }


@app.delete("/api/tasks/cleanup", summary="清理已完成的旧任务")
async def cleanup_old_tasks(max_age_seconds: int = 3600):
    """
    清理已完成、已取消或失败的旧任务
    
    Args:
        max_age_seconds: 任务最大保留时间（秒），默认1小时
        
    Returns:
        清理结果
    """
    task_manager.cleanup_completed_tasks(max_age_seconds)

    return {
        "success": True,
        "message": f"Cleaned up tasks older than {max_age_seconds} seconds"
    }


# ========== 大纲状态查询相关 ==========
@app.get("/api/task/{task_id}/outline", summary="获取任务的大纲生成状态")
async def get_task_outline_status(task_id: str):
    """
    获取任务的大纲生成状态（用于Auto模式下前端轮询展示大纲）
    
    Args:
        task_id: 任务ID
        
    Returns:
        大纲生成状态和内容
    """
    task_info = task_manager.get_task(task_id)
    
    if task_info is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # 通过映射获取session_id
    session_id = task_session_mapping.get(task_id)
    
    outline_content = None
    outline_generated = False
    
    # 进度信息
    progress_info = None
    
    if session_id:
        # 查找workspace路径
        current_file = Path(__file__).resolve()
        project_root = None
        for parent in [current_file.parent] + list(current_file.parents):
            if (parent / "app.py").exists():
                project_root = parent
                break
        if project_root is None:
            project_root = Path.cwd()
        
        workspace_path = project_root / "workspaces" / session_id
        outline_file = workspace_path / ".outline_generated"
        progress_file = workspace_path / ".progress"
        
        if outline_file.exists():
            try:
                with open(outline_file, 'r', encoding='utf-8') as f:
                    outline_content = f.read()
                outline_generated = True
            except Exception as e:
                logger.warning(f"读取大纲文件失败: {e}")
        
        # 读取进度信息
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_info = json.load(f)
            except Exception as e:
                logger.warning(f"读取进度文件失败: {e}")
    
    return {
        "task_id": task_id,
        "status": task_info.status.value,
        "outline_generated": outline_generated,
        "outline_content": outline_content,
        "task_completed": task_info.status.value in ["completed", "failed", "cancelled"],
        "progress": progress_info
    }


# ========== Human in the loop 大纲确认相关 ==========

@app.post("/api/outline/confirm", summary="确认或修改大纲")
async def confirm_outline(request: OutlineConfirmRequest):
    """
    Human in the loop 模式：用户确认或修改大纲后继续执行
    
    Args:
        request: 大纲确认请求，包含 task_id, session_id, action, outline, modified
        
    Returns:
        确认结果
    """
    task_id = request.task_id
    
    # 【调试日志】检查 pending_outline_confirmations 状态
    logger.info(f"[HITL DEBUG] confirm_outline 被调用: task_id={task_id}")
    with pending_outline_lock:
        logger.info(f"[HITL DEBUG] pending_outline_confirmations 键: {list(pending_outline_confirmations.keys())}")
        if task_id not in pending_outline_confirmations:
            logger.error(f"[HITL DEBUG] task_id={task_id} 不在 pending_outline_confirmations 中!")
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or not waiting for outline confirmation")
        
        pending_task = pending_outline_confirmations[task_id]
        logger.info(f"[HITL DEBUG] 找到任务: task_id={task_id}, status={pending_task.get('status')}")
        
        if request.action == "cancel":
            # 用户取消，清理状态
            del pending_outline_confirmations[task_id]
            task_manager.cancel_task(task_id)
            return {
                "success": True,
                "message": "Task cancelled by user",
                "task_id": task_id
            }
        
        elif request.action == "confirm":
            # 用户确认大纲，设置确认标志
            pending_task["confirmed"] = True
            pending_task["user_outline"] = request.outline if request.modified else pending_task["outline_content"]
            pending_task["modified"] = request.modified
            
            # 通知等待的协程继续执行
            if "event" in pending_task and pending_task["event"]:
                pending_task["event"].set()
            
            # 从 pending 中移除，但保留在内存中供 process_single_query 检查
            if task_id in pending_outline_confirmations:
                confirmed_task = pending_outline_confirmations.pop(task_id)
                # 重新添加标记为已确认，让 process_single_query 可以获取
                confirmed_task['status'] = 'confirmed'
                pending_outline_confirmations[task_id] = confirmed_task
            
            logger.info(f"[HITL DEBUG] 大纲已确认: task_id={task_id}, modified={request.modified}")
            
            return {
                "success": True,
                "message": "Outline confirmed, continuing report generation",
                "task_id": task_id,
                "modified": request.modified
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")


@app.get("/api/outline/{task_id}", summary="获取等待确认的大纲")
async def get_pending_outline(task_id: str):
    """
    获取等待用户确认的大纲内容
    
    Args:
        task_id: 任务ID
        
    Returns:
        大纲内容和状态
    """
    with pending_outline_lock:
        if task_id not in pending_outline_confirmations:
            return {
                "success": False,
                "waiting": False,
                "message": f"Task {task_id} not found or not waiting for outline confirmation"
            }
        
        pending_task = pending_outline_confirmations[task_id]
        return {
            "success": True,
            "waiting": True,
            "task_id": task_id,
            "session_id": pending_task.get("session_id"),
            "outline": pending_task.get("outline_content"),
            "reasoning_content": pending_task.get("reasoning_content"),
            "confirmed": pending_task.get("confirmed", False)
        }


@app.post("/api/outline/continue", summary="继续执行报告生成")
async def continue_with_outline(request: OutlineConfirmRequest):
    """
    Human in the loop 模式：用户确认大纲后继续执行报告生成（异步提交到线程池）
    """
    task_id = request.task_id
    
    # 【调试】检查 task_manager
    logger.info(f"[HITL DEBUG] continue_with_outline 被调用, task_id={task_id}")
    logger.info(f"[HITL DEBUG] task_manager 类型: {type(task_manager)}, 值: {task_manager}")
    
    with pending_outline_lock:
        if task_id not in pending_outline_confirmations:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or not waiting for outline confirmation")
        
        pending_task = pending_outline_confirmations[task_id]
        
        if request.action == "cancel":
            # 用户取消，清理状态
            del pending_outline_confirmations[task_id]
            task_manager.cancel_task(task_id)
            
            # 发送取消 SSE 通知
            send_progress_update(task_id, {
                'type': 'cancelled',
                'message': '用户取消了大纲确认'
            })
            
            return {
                'success': False,
                'query': pending_task.get('query_data', ('',))[0] if pending_task.get('query_data') else '',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'session_id': pending_task.get('session_id', ''),
                'task_id': task_id,
                'planner_result': None,
                'planner_error': 'Task cancelled by user',
                'planner_reasoning_trace': [],
                'planner_iterations': 0,
                'planner_execution_time': 0,
                'planner_agent_name': 'QueueManager',
                'section_writer_responses': [],
                'final_report': None,
                'report_path': None,
                'waiting_for_outline_confirm': False,
                'outline_content': None,
                'reasoning_content': None
            }
        
        # 用户确认大纲，继续执行
        # 优先使用请求中传入的非空 outline，避免 modified 标志异常时回退到旧大纲
        request_outline = (request.outline or "").strip() if request.outline is not None else ""
        user_outline = request_outline if request_outline else (pending_task.get("outline_content") or "")
        query_data = pending_task["query_data"]
        username = pending_task["username"]
        workspace_path = pending_task["workspace_path"]
        session_id = pending_task["session_id"]
        frontend_session_id = pending_task.get("frontend_session_id")
        phase1_execution_time = float(pending_task.get("phase1_execution_time", 0.0) or 0.0)
        outline_ready_at = float(pending_task.get("outline_ready_at", 0.0) or 0.0)
        outline_confirm_wait_time = max(0.0, time.time() - outline_ready_at) if outline_ready_at > 0 else 0.0
        
        # 清理 pending 状态
        del pending_outline_confirmations[task_id]
    
    # 更新任务状态
    task_manager.update_task_status(task_id, TaskStatus.RUNNING)
    
    # 发送 SSE 通知：大纲已确认，开始写作
    send_progress_update(task_id, {
        'type': 'outline_confirmed',
        'message': '大纲已确认，开始生成报告...'
    })
    
    loop = asyncio.get_event_loop()
    
    # 使用线程池异步执行 Phase 2
    if executor is None:
        raise HTTPException(status_code=500, detail="Server executor not initialized")
    
    loop.run_in_executor(
        executor,
        lambda: process_single_query_phase2(
            query_data=query_data,
            task_id=task_id,
            username=username,
            workspace_path=workspace_path,
            session_id=session_id,
            user_outline=user_outline,
            frontend_session_id=frontend_session_id,
            phase1_execution_time=phase1_execution_time,
            outline_confirm_wait_time=outline_confirm_wait_time
        )
    )
    
    logger.info(f"[HITL] Phase 2 已提交到后台执行: task_id={task_id}")
    return {
        'success': True,
        'query': query_data[0] if query_data else '',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'session_id': session_id,
        'task_id': task_id,
        'planner_result': None,
        'planner_error': None,
        'planner_reasoning_trace': [],
        'planner_iterations': 0,
        'planner_execution_time': 0,
        'planner_agent_name': 'TaskAccepted',
        'section_writer_responses': [],
        'final_report': None,
        'report_path': None,
        'waiting_for_outline_confirm': False,
        'outline_content': None,
        'reasoning_content': None
    }


def process_single_query_phase2(query_data, task_id: str, username: str, 
                                 workspace_path: str, session_id: str, user_outline: str,
                                 frontend_session_id: Optional[str] = None,
                                 phase1_execution_time: float = 0.0,
                                 outline_confirm_wait_time: float = 0.0):
    """Human in the loop 阶段2：使用用户确认的大纲继续生成报告"""
    query_text, query_index, user_files_data, search_sources_dict = query_data
    workspace_path = Path(workspace_path)
    
    try:
        app_config = get_config()
        sub_agent_configs = {
            "information_seeker": {
                "model": app_config.model_name,
                **({"max_iterations": app_config.information_seeker_max_iterations} if app_config.information_seeker_max_iterations else {})
            },
            "writer": {
                "model": app_config.model_name,
                **({"max_iterations": app_config.writer_max_iterations} if app_config.writer_max_iterations else {})
            }
        }
        
        # 设置环境变量
        os.environ['AGENT_SESSION_ID'] = session_id
        os.environ['AGENT_WORKSPACE_PATH'] = str(workspace_path)
        os.environ['HUMAN_IN_LOOP'] = 'true'
        os.environ['HUMAN_IN_LOOP_PHASE2'] = 'true'  # 标记为阶段2，跳过搜索直接写作
        
        # 设置搜索源偏好
        if search_sources_dict:
            os.environ['SEARCH_SOURCE_WEBSEARCH'] = str(search_sources_dict.get('websearch', False))
            os.environ['SEARCH_SOURCE_PUBMED'] = str(search_sources_dict.get('pubmed', False))
            os.environ['SEARCH_SOURCE_ARXIV'] = str(search_sources_dict.get('arxiv', False))
            os.environ['SEARCH_SOURCE_GOOGLE_SCHOLAR'] = str(search_sources_dict.get('google_scholar', False))
            os.environ['SEARCH_SOURCE_SPRINGER'] = str(search_sources_dict.get('springer', False))
        
        # 确保 .human_in_loop 标记文件存在（供 MCP 工具读取）
        human_in_loop_file = workspace_path / '.human_in_loop'
        with open(human_in_loop_file, 'w', encoding='utf-8') as f:
            f.write('true')
        
        # 写入用户确认的大纲
        outline_file = workspace_path / '.user_outline'
        with open(outline_file, 'w', encoding='utf-8') as f:
            f.write(user_outline)
        os.environ['USER_OUTLINE_PATH'] = str(outline_file)
        logger.info(f"Human in the loop Phase 2: 用户大纲已写入 {outline_file}")
        
        # 删除 .outline_pending 文件，表示大纲已确认
        outline_pending_path = workspace_path / ".outline_pending"
        if outline_pending_path.exists():
            outline_pending_path.unlink()
        
        agent = create_planner_agent(
            agent_name=f"PlannerAgent",
            model=app_config.model_name,
            max_iterations=app_config.planner_max_iterations or 40,
            sub_agent_configs=sub_agent_configs,
            task_id=task_id
        )
        # 直接注入确认后的大纲，避免在并发场景下依赖全局环境变量读取错误工作区文件
        try:
            setattr(agent, "_hitl_user_outline", user_outline or "")
        except Exception:
            pass
        
        # 设置取消令牌
        cancellation_token = task_manager.get_cancellation_token(task_id)
        if cancellation_token:
            agent.set_cancellation_token(cancellation_token)
        
        if hasattr(agent, 'set_progress_callback'):
            agent.set_progress_callback(send_progress_update)
        
        # 构建增强的查询文本
        enhanced_query = _build_enhanced_query(query_text, user_files_data)
        
        start_time = time.time()
        response = agent.execute_task(enhanced_query + " /no_think")
        execution_time = time.time() - start_time
        phase2_execution_time = execution_time
        total_generation_time = max(0.0, phase1_execution_time) + phase2_execution_time
        
        # 检查是否被取消
        if (task_manager.is_task_cancelled(task_id) or
            (hasattr(response, 'error') and response.error and "cancelled" in str(response.error).lower())):
            task_manager.update_task_status(task_id, TaskStatus.CANCELLED, error=getattr(response, 'error', 'Task cancelled'))
            send_progress_update(task_id, {
                'type': 'cancelled',
                'message': '任务已取消'
            })
            return
        
        # 读取最终报告内容
        final_report_content = None
        report_relative_path = None
        try:
            final_report_path = workspace_path / "report" / "final_report.md"
            if final_report_path.exists():
                with open(final_report_path, 'r', encoding='utf-8') as f:
                    final_report_content = f.read()
                report_relative_path = "report/final_report.md"
                logger.info(f"[HITL Phase2] 成功读取最终报告: {final_report_path} (大小: {len(final_report_content)} 字符)")
                
                # 计算报告字数
                word_count = 0
                try:
                    import re
                    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', final_report_content))
                    english_words = len(re.findall(r'\b[a-zA-Z]+\b', final_report_content))
                    word_count = chinese_chars + english_words
                except Exception:
                    pass
                
                # 统计检索次数
                total_search_count = 0
                try:
                    tool_call_logs_dir = workspace_path / "tool_call_logs"
                    if tool_call_logs_dir.is_dir():
                        for log_file in tool_call_logs_dir.glob("tool_calls_*.jsonl"):
                            try:
                                with open(log_file, 'r', encoding='utf-8') as lf:
                                    for line in lf:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        try:
                                            rec = json.loads(line)
                                        except Exception:
                                            continue
                                        tool_name = rec.get('tool_name')
                                        if tool_name == 'batch_web_search':
                                            input_args = rec.get('input_args') or {}
                                            queries = input_args.get('queries') or []
                                            if isinstance(queries, list):
                                                total_search_count += len(queries)
                                        elif tool_name == 'url_crawler':
                                            input_args = rec.get('input_args') or {}
                                            documents = input_args.get('documents') or []
                                            if isinstance(documents, list):
                                                total_search_count += len(documents)
                                        elif tool_name in ['arxiv_search', 'search_pubmed_key_words', 'search_pubmed_advanced',
                                                          'medrxiv_search', 'springer_search', 'get_pubmed_article',
                                                          'arxiv_read_paper', 'medrxiv_read_paper', 'springer_get_article']:
                                            total_search_count += 1
                            except Exception:
                                pass
                except Exception:
                    pass
                
                # 添加统计信息
                try:
                    import re
                    query_zh_count = len(re.findall(r'[\u4e00-\u9fff]', query_text))
                    query_total_chars = len(query_text.strip())
                    is_chinese_query = (query_zh_count / max(query_total_chars, 1)) > 0.3
                    
                    if is_chinese_query:
                        stats_section = f"\n\n<div style=\"page-break-before: always;\"></div>\n\n## 报告统计信息\n\n"
                        stats_section += f"- 报告字数: {word_count:,} 字\n"
                        stats_section += f"- 生成耗时(总计): {total_generation_time:.2f} 秒 ({total_generation_time/60:.1f} 分钟)\n"
                        stats_section += f"- 生成耗时(Phase1-大纲): {phase1_execution_time:.2f} 秒 ({phase1_execution_time/60:.1f} 分钟)\n"
                        stats_section += f"- 生成耗时(Phase2-写作): {phase2_execution_time:.2f} 秒 ({phase2_execution_time/60:.1f} 分钟)\n"
                        if outline_confirm_wait_time > 0:
                            stats_section += f"- 用户确认等待: {outline_confirm_wait_time:.2f} 秒 ({outline_confirm_wait_time/60:.1f} 分钟)\n"
                        stats_section += f"- 网站检索: {total_search_count:,} 次\n"
                    else:
                        stats_section = f"\n\n<div style=\"page-break-before: always;\"></div>\n\n## Report Statistics\n\n"
                        stats_section += f"- Word Count: {word_count:,} words\n"
                        stats_section += f"- Generation Time (Total): {total_generation_time:.2f} seconds ({total_generation_time/60:.1f} minutes)\n"
                        stats_section += f"- Generation Time (Phase 1 - Outline): {phase1_execution_time:.2f} seconds ({phase1_execution_time/60:.1f} minutes)\n"
                        stats_section += f"- Generation Time (Phase 2 - Writing): {phase2_execution_time:.2f} seconds ({phase2_execution_time/60:.1f} minutes)\n"
                        if outline_confirm_wait_time > 0:
                            stats_section += f"- User Confirmation Wait: {outline_confirm_wait_time:.2f} seconds ({outline_confirm_wait_time/60:.1f} minutes)\n"
                        stats_section += f"- Web Searches: {total_search_count:,} times\n"
                    
                    final_report_content_with_stats = final_report_content + stats_section
                    
                    with open(final_report_path, 'w', encoding='utf-8') as f:
                        f.write(final_report_content_with_stats)
                    
                    final_report_content = final_report_content_with_stats
                    logger.info(f"[HITL Phase2] 已在报告末尾添加统计信息")
                    
                    # 异步生成PDF
                    def generate_pdf_async():
                        try:
                            from src.tools.mcp_tools import generate_pdf_with_reportlab
                            pdf_path = Path(final_report_path).parent.parent / "final_report.pdf"
                            success = generate_pdf_with_reportlab(final_report_content_with_stats, pdf_path)
                            if success:
                                logger.info(f"[HITL Phase2][异步] 成功生成PDF文件: {pdf_path}")
                        except Exception as pdf_error:
                            logger.warning(f"[HITL Phase2][异步] 生成PDF失败: {pdf_error}")
                    
                    import threading
                    pdf_thread = threading.Thread(target=generate_pdf_async, daemon=True)
                    pdf_thread.start()
                except Exception:
                    pass
                
                # 存储报告到数据库
                try:
                    if frontend_session_id:
                        import re
                        report_title = "研究报告"
                        title_match = re.search(r'^#\s+(.+?)$', final_report_content, re.MULTILINE)
                        if title_match:
                            report_title = title_match.group(1).strip()
                        if len(report_title) > 50:
                            report_title = report_title[:50] + '...'
                        
                        store_url = "http://localhost:5000/api/chat/messages"
                        store_data = {
                            "session_id": frontend_session_id,
                            "from_who": "ai",
                            "content": final_report_content,
                            "round": 1,
                            "uuid": session_id,
                            "has_report": 1,
                            "report_title": report_title
                        }
                        http_response = requests.post(store_url, json=store_data, timeout=30)
                        if http_response.status_code == 201:
                            logger.info(f"[HITL Phase2] 成功存储报告到数据库")
                        else:
                            logger.error(f"[HITL Phase2] 存储报告失败: {http_response.status_code}")
                except Exception as store_error:
                    logger.error(f"[HITL Phase2] 自动存储报告到数据库失败: {store_error}")
            else:
                logger.warning(f"[HITL Phase2] 最终报告文件不存在: {final_report_path}")
        except Exception as e:
            logger.error(f"[HITL Phase2] 读取最终报告失败: {e}")
        
        planner_success = bool(getattr(response, 'success', False))
        planner_error = (getattr(response, 'error', None) or '').strip()
        has_final_report = bool(final_report_content and str(final_report_content).strip())

        def _phase2_store_error_message_to_db(content: str) -> None:
            if not frontend_session_id:
                return
            try:
                store_url = "http://localhost:5000/api/chat/messages"
                store_data = {
                    "session_id": frontend_session_id,
                    "from_who": "ai",
                    "content": content,
                    "round": 1,
                    "uuid": session_id,
                    "has_report": 0,
                    "report_title": ""
                }
                http_response = requests.post(store_url, json=store_data, timeout=30)
                if http_response.status_code != 201:
                    logger.error(f"[HITL Phase2] 存储错误提示到数据库失败: {http_response.status_code}")
                else:
                    logger.info("[HITL Phase2] 已存储错误提示到数据库")
            except Exception as db_err:
                logger.error(f"[HITL Phase2] 存储错误提示异常: {db_err}")

        if not has_final_report:
            detail = planner_error if not planner_success else '工作区未生成可读最终报告（report/final_report.md）'
            user_msg = (
                "⚠️ HITL 阶段2 未能生成最终报告。\n\n"
                + (f"原因：{detail}\n\n" if detail else "")
                + "请稍后重试，或适当缩短/简化大纲与要点后再次提交。"
            )
            fail_result = {
                'session_id': session_id,
                'final_report': None,
                'report_path': None,
                'execution_time': execution_time,
                'phase1_execution_time': phase1_execution_time,
                'phase2_execution_time': phase2_execution_time,
                'total_generation_time': total_generation_time,
                'outline_confirm_wait_time': outline_confirm_wait_time,
                'planner_success': planner_success,
                'planner_error': planner_error or None,
            }
            task_manager.update_task_status(
                task_id, TaskStatus.FAILED,
                result=fail_result,
                error=(user_msg[:4000] if user_msg else 'Phase2: no final report'),
            )
            send_progress_update(task_id, {
                'type': 'error',
                'message': user_msg,
                'session_id': session_id,
            })
            _phase2_store_error_message_to_db(user_msg)
            logger.warning(
                f"[HITL Phase2] 无有效最终报告，已标记 FAILED: task_id={task_id}, "
                f"planner_success={planner_success}, planner_error={planner_error!r}"
            )
            return

        if not planner_success:
            logger.warning(
                f"[HITL Phase2] Planner 标记失败但已存在 final_report（可能为兜底合并），仍按完成交付: {planner_error!r}"
            )

        # 更新任务状态为完成
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result={
            'session_id': session_id,
            'final_report': final_report_content,
            'report_path': report_relative_path,
            'execution_time': execution_time,
            'phase1_execution_time': phase1_execution_time,
            'phase2_execution_time': phase2_execution_time,
            'total_generation_time': total_generation_time,
            'outline_confirm_wait_time': outline_confirm_wait_time,
            'planner_success': planner_success,
            'planner_error': planner_error or None,
        })
        
        # 发送完成 SSE 通知
        send_progress_update(task_id, {
            'type': 'completed',
            'message': '报告生成完成',
            'session_id': session_id,
            'report_path': report_relative_path,
            'final_report': final_report_content
        })
        
        logger.info(
            f"[HITL Phase2] 任务 {task_id} 完成, "
            f"phase1={phase1_execution_time:.2f}s, phase2={phase2_execution_time:.2f}s, "
            f"总计={total_generation_time:.2f}s, 用户确认等待={outline_confirm_wait_time:.2f}s"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[HITL Phase2] 任务 {task_id} 执行失败: {e}", exc_info=True)
        task_manager.update_task_status(task_id, TaskStatus.FAILED, error=str(e))
        
        # 发送失败 SSE 通知
        send_progress_update(task_id, {
            'type': 'error',
            'message': f'报告生成失败: {str(e)}'
        })
        
        # 存储错误信息到数据库
        if frontend_session_id:
            try:
                store_url = "http://localhost:5000/api/chat/messages"
                store_data = {
                    "session_id": frontend_session_id,
                    "from_who": "ai",
                    "content": f"⚠️ 报告生成失败：{str(e)}\n\n请尝试重新提问。",
                    "round": 1,
                    "uuid": session_id,
                    "has_report": 0,
                    "report_title": ""
                }
                requests.post(store_url, json=store_data, timeout=30)
            except Exception:
                pass
    finally:
        # 清理阶段2 环境变量
        os.environ.pop('HUMAN_IN_LOOP_PHASE2', None)


if __name__ == "__main__":
    # 【修复 SIGHUP 导致的进程重启问题】
    # 忽略 SIGHUP 信号，防止 SSH 断开或终端关闭时触发 uvicorn 重启
    # SIGHUP 仅在 Unix 系统上可用
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # 启动UVicorn服务器（使用当前文件名作为模块）
    uvicorn.run(
        app="a:app",  # 因为文件名为a.py，所以模块名为a
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # 建议启动1个worker进程，避免多进程下出现数据不一致问题，已使用线程池技术实现多并发
    )
