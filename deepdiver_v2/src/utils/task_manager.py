# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""
Task Manager for handling concurrent agent tasks with cancellation support
支持取消功能的并发agent任务管理器
"""
import os
import threading
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """Information about a running task"""
    task_id: str
    query: str
    status: TaskStatus
    created_at: float
    updated_at: float
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    cancellation_token: threading.Event = field(default_factory=threading.Event)
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    queue_position: Optional[int] = None
    
    def is_cancelled(self) -> bool:
        """Check if task has been cancelled"""
        return self.cancellation_token.is_set()
    
    def cancel(self):
        """Request task cancellation"""
        self.cancellation_token.set()
        self.status = TaskStatus.CANCELLED
        self.updated_at = time.time()


class TaskManager:
    """
    Global task manager for tracking and managing all running agent tasks
    用于跟踪和管理所有运行中agent任务的全局任务管理器
    
    注意：在多进程环境下（如 uvicorn workers > 1），每个进程会有独立的 TaskManager 实例
    """
    
    def __init__(self):
        """Initialize task manager"""
        # 每次创建新实例时都初始化（多进程安全）
        self._tasks: Dict[str, TaskInfo] = {}
        self._tasks_lock = threading.Lock()
        logger.info(f"TaskManager initialized in process {os.getpid()}")
    
    def create_task(self, task_id: str, query: str) -> TaskInfo:
        """
        Create a new task and register it
        
        Args:
            task_id: Unique task identifier
            query: User query for this task
            
        Returns:
            TaskInfo object for the new task
        """
        with self._tasks_lock:
            if task_id in self._tasks:
                logger.warning(f"Task {task_id} already exists, returning existing task")
                return self._tasks[task_id]
            
            task_info = TaskInfo(
                task_id=task_id,
                query=query,
                status=TaskStatus.PENDING,
                created_at=time.time(),
                updated_at=time.time()
            )
            self._tasks[task_id] = task_info
            logger.info(f"Created task {task_id}: {query[:100]}...")
            return task_info
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get task information by task ID
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskInfo object or None if not found
        """
        with self._tasks_lock:
            return self._tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          result: Optional[Any] = None, 
                          error: Optional[str] = None):
        """
        Update task status
        
        Args:
            task_id: Task identifier
            status: New status
            result: Task result (if completed)
            error: Error message (if failed)
        """
        with self._tasks_lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found for status update")
                return
            
            task = self._tasks[task_id]
            task.status = status
            task.updated_at = time.time()
            
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            
            logger.info(f"Task {task_id} status updated to {status.value}")
    
    def update_task_progress(self, task_id: str, progress_info: Dict[str, Any]):
        """
        Update task progress information
        
        Args:
            task_id: Task identifier
            progress_info: Progress information dict
        """
        with self._tasks_lock:
            if task_id not in self._tasks:
                return
            
            task = self._tasks[task_id]
            task.progress.update(progress_info)
            task.updated_at = time.time()
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Request task cancellation
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was found and cancellation requested, False otherwise
        """
        # 【关键修复】使用单次锁获取，避免死锁和长时间阻塞
        # 整个操作非常快（只是设置Event和更新状态），不需要复杂的双重检查
        try:
            with self._tasks_lock:
                if task_id not in self._tasks:
                    logger.warning(f"Task {task_id} not found for cancellation")
                    return False
                
                task = self._tasks[task_id]
                
                # 检查任务是否已经在终态
                if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED]:
                    logger.info(f"Task {task_id} already in terminal state: {task.status.value}")
                    return False
                
                # 执行取消操作
                task.cancel()
                logger.info(f"Task {task_id} cancellation requested, status: {task.status.value}")
                return True
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False
    
    def get_cancellation_token(self, task_id: str) -> Optional[threading.Event]:
        """
        Get cancellation token for a task
        
        Args:
            task_id: Task identifier
            
        Returns:
            threading.Event object that will be set when task should be cancelled
        """
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            return task.cancellation_token if task else None
    
    def is_task_cancelled(self, task_id: str) -> bool:
        """
        Check if task has been cancelled
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task is cancelled, False otherwise
        """
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            return task.is_cancelled() if task else False
    
    def cleanup_completed_tasks(self, max_age_seconds: int = 3600):
        """
        Remove completed/cancelled/failed tasks older than max_age_seconds
        
        Args:
            max_age_seconds: Maximum age for completed tasks in seconds
        """
        current_time = time.time()
        with self._tasks_lock:
            tasks_to_remove = []
            for task_id, task in self._tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED]:
                    age = current_time - task.updated_at
                    if age > max_age_seconds:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                logger.info(f"Cleaned up old task {task_id}")
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all tasks
        
        Returns:
            Dictionary mapping task_id to task info
        """
        with self._tasks_lock:
            return {
                task_id: {
                    "task_id": task.task_id,
                    "query": task.query,
                    "status": task.status.value,
                    "created_at": task.created_at,
                    "updated_at": task.updated_at,
                    "thread_id": task.thread_id,
                    "progress": task.progress,
                    "has_error": task.error is not None
                }
                for task_id, task in self._tasks.items()
            }
    
    def get_running_tasks_count(self) -> int:
        """Get count of currently running tasks"""
        with self._tasks_lock:
            return sum(1 for task in self._tasks.values() 
                      if task.status == TaskStatus.RUNNING)
    
    def get_queued_tasks_count(self) -> int:
        """Get count of queued tasks"""
        with self._tasks_lock:
            return sum(1 for task in self._tasks.values() 
                      if task.status == TaskStatus.QUEUED)
    
    def get_queue_position(self, task_id: str) -> Optional[int]:
        """Get the position of a task in the queue (1-indexed)"""
        with self._tasks_lock:
            queued_tasks = sorted(
                [task for task in self._tasks.values() if task.status == TaskStatus.QUEUED],
                key=lambda t: t.created_at
            )
            for i, task in enumerate(queued_tasks, 1):
                if task.task_id == task_id:
                    return i
            return None
    
    def update_queue_positions(self):
        """Update queue positions for all queued tasks"""
        with self._tasks_lock:
            queued_tasks = sorted(
                [task for task in self._tasks.values() if task.status == TaskStatus.QUEUED],
                key=lambda t: t.created_at
            )
            for i, task in enumerate(queued_tasks, 1):
                task.queue_position = i
    
    def remove_task(self, task_id: str):
        """
        Remove a task from the manager
        
        Args:
            task_id: Task identifier
        """
        with self._tasks_lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"Removed task {task_id}")


# Global singleton instance - 使用延迟初始化避免多进程问题
_task_manager_instance = None

def get_task_manager() -> TaskManager:
    """
    获取 TaskManager 单例（延迟初始化）
    
    在多进程环境下（如 uvicorn workers > 1），每个进程会在首次调用时
    创建自己的 TaskManager 实例，避免 fork 时继承父进程的锁状态
    """
    global _task_manager_instance
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager()
    return _task_manager_instance

# 向后兼容：保留旧的全局变量名，但改为属性访问
class _TaskManagerProxy:
    """代理对象，延迟初始化真实的 TaskManager"""
    def __getattr__(self, name):
        return getattr(get_task_manager(), name)
    
    def __setattr__(self, name, value):
        setattr(get_task_manager(), name, value)

task_manager = _TaskManagerProxy()

