# Copyright (c) 2026 South China Sea Institute of Oceanology, Chinese Academy of Sciences (SCSIO, CAS). All rights reserved.
"""
Local Workspace Manager for Multi-Agent System

This module provides session-based workspace management using local directories.
Each chat session gets its own isolated workspace directory that persists
throughout the conversation and can be cleaned up when the session ends.

Features:
- Session-based workspace lifecycle management
- Local directory isolation per session
- File operations within session workspaces
- Integration with existing MCP tools
- Comprehensive error handling and logging
"""

import shutil
import logging
from typing import Dict, Optional, Any, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import json

# Configure logging
logger = logging.getLogger(__name__)


class WorkspaceStatus(Enum):
    """Workspace lifecycle status"""
    CREATING = "creating"
    ACTIVE = "active"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"
    ERROR = "error"


@dataclass
class WorkspaceInfo:
    """Information about a workspace instance"""
    workspace_id: str
    session_id: str
    workspace_path: Path
    created_at: datetime
    last_activity: datetime
    status: WorkspaceStatus
    workspace_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "workspace_path": str(self.workspace_path),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "status": self.status.value,
            "workspace_files": self.workspace_files,
            "metadata": self.metadata,
            "error_message": self.error_message
        }


class LocalWorkspaceManager:
    """
    Manages local workspaces for multi-agent chat sessions.
    
    Each chat session gets its own isolated workspace directory that persists
    throughout the conversation. Workspaces are automatically managed
    with cleanup capabilities.
    """
    
    def __init__(
        self,
        base_workspace_dir: str = "workspaces",
        default_timeout: int = 86400,  # 24 hours default
        cleanup_on_exit: bool = False  # Don't auto-cleanup by default
    ):
        """
        Initialize the workspace manager.
        
        Args:
            base_workspace_dir: Base directory for all workspaces
            default_timeout: Default workspace timeout in seconds
            cleanup_on_exit: Whether to cleanup workspaces on manager shutdown
        """
        self.base_workspace_dir = Path(base_workspace_dir)
        self.base_workspace_dir.mkdir(exist_ok=True)
        self.default_timeout = default_timeout
        self.cleanup_on_exit = cleanup_on_exit
        
        # Active workspaces by session ID
        self.workspaces: Dict[str, WorkspaceInfo] = {}
        
        # Load existing workspaces from metadata
        self._load_existing_workspaces()
        
        logger.info(f"LocalWorkspaceManager initialized with base_dir={base_workspace_dir}")

    def _load_existing_workspaces(self):
        """Load existing workspaces from metadata files"""
        try:
            for workspace_dir in self.base_workspace_dir.iterdir():
                if workspace_dir.is_dir():
                    metadata_file = workspace_dir / ".workspace_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                data = json.load(f)
                            
                            workspace_info = WorkspaceInfo(
                                workspace_id=data["workspace_id"],
                                session_id=data["session_id"],
                                workspace_path=Path(data["workspace_path"]),
                                created_at=datetime.fromisoformat(data["created_at"]),
                                last_activity=datetime.fromisoformat(data["last_activity"]),
                                status=WorkspaceStatus(data["status"]),
                                workspace_files=data.get("workspace_files", []),
                                metadata=data.get("metadata", {}),
                                error_message=data.get("error_message")
                            )
                            
                            self.workspaces[workspace_info.session_id] = workspace_info
                            logger.info(f"Loaded existing workspace for session {workspace_info.session_id}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to load workspace metadata from {metadata_file}: {e}")
                            
        except Exception as e:
            logger.warning(f"Failed to load existing workspaces: {e}")

    @staticmethod
    def _save_workspace_metadata(workspace_info: WorkspaceInfo):
        """Save workspace metadata to disk"""
        try:
            metadata_file = workspace_info.workspace_path / ".workspace_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(workspace_info.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save workspace metadata: {e}")

    def create_workspace(
        self,
        session_id: str,
        workspace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkspaceInfo:
        """
        Create a new workspace for a chat session.
        
        Args:
            session_id: Unique session identifier
            workspace_id: Optional custom workspace ID (defaults to session_id)
            metadata: Additional metadata to store with the workspace
            
        Returns:
            WorkspaceInfo: Information about the created workspace
            
        Raises:
            ValueError: If session already has an active workspace
            Exception: If workspace creation fails
        """
        if session_id in self.workspaces:
            raise ValueError(f"Session {session_id} already has an active workspace")
        
        workspace_id = workspace_id or session_id
        workspace_path = self.base_workspace_dir / workspace_id
        
        logger.info(f"Creating workspace for session {session_id} at {workspace_path}")
        
        # Create workspace info with creating status
        workspace_info = WorkspaceInfo(
            workspace_id=workspace_id,
            session_id=session_id,
            workspace_path=workspace_path,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            status=WorkspaceStatus.CREATING,
            metadata=metadata or {}
        )
        
        try:
            # Create workspace directory
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (workspace_path / "downloads").mkdir(exist_ok=True)
            (workspace_path / "outputs").mkdir(exist_ok=True)
            (workspace_path / "temp").mkdir(exist_ok=True)
            
            # Update status
            workspace_info.status = WorkspaceStatus.ACTIVE
            self.workspaces[session_id] = workspace_info
            
            # Save metadata
            self._save_workspace_metadata(workspace_info)
            
            # Update workspace files list
            self._update_workspace_files(session_id)
            
            logger.info(f"Workspace created successfully: {workspace_path} for session {session_id}")
            return workspace_info
            
        except Exception as e:
            workspace_info.status = WorkspaceStatus.ERROR
            workspace_info.error_message = str(e)
            logger.error(f"Failed to create workspace for session {session_id}: {e}")
            raise

    def get_workspace(self, session_id: str) -> Optional[WorkspaceInfo]:
        """Get workspace info for a session"""
        workspace_info = self.workspaces.get(session_id)
        if workspace_info:
            # Update last activity
            workspace_info.last_activity = datetime.now()
            self._save_workspace_metadata(workspace_info)
        return workspace_info

    def get_workspace_path(self, session_id: str) -> Optional[Path]:
        """Get workspace path for a session"""
        workspace_info = self.get_workspace(session_id)
        return workspace_info.workspace_path if workspace_info else None

    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.workspaces.keys())

    def destroy_workspace(self, session_id: str, force: bool = False) -> bool:
        """
        Destroy a workspace for a session.
        
        Args:
            session_id: Session identifier
            force: Force removal even if files exist
            
        Returns:
            bool: True if destroyed successfully
        """
        if session_id not in self.workspaces:
            logger.warning(f"No workspace found for session {session_id}")
            return False
        
        workspace_info = self.workspaces[session_id]
        
        try:
            logger.info(f"Destroying workspace for session {session_id}")
            workspace_info.status = WorkspaceStatus.DESTROYING
            
            # Remove workspace directory
            if workspace_info.workspace_path.exists():
                if force or not any(workspace_info.workspace_path.iterdir()):
                    shutil.rmtree(workspace_info.workspace_path)
                    logger.info(f"Workspace directory removed: {workspace_info.workspace_path}")
                else:
                    logger.warning(f"Workspace contains files, use force=True to remove: {workspace_info.workspace_path}")
                    return False
            
            # Update status and remove from active workspaces
            workspace_info.status = WorkspaceStatus.DESTROYED
            del self.workspaces[session_id]
            
            logger.info(f"Workspace destroyed for session {session_id}")
            return True
            
        except Exception as e:
            workspace_info.status = WorkspaceStatus.ERROR
            workspace_info.error_message = str(e)
            logger.error(f"Failed to destroy workspace for session {session_id}: {e}")
            return False

    def write_file(self, session_id: str, file_path: str, content: Union[str, bytes]) -> bool:
        """Write content to a file in the workspace"""
        workspace_info = self.get_workspace(session_id)
        if not workspace_info:
            logger.error(f"No workspace found for session {session_id}")
            return False
        
        try:
            full_path = workspace_info.workspace_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(content, str):
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                with open(full_path, 'wb') as f:
                    f.write(content)
            
            # Update workspace files list
            self._update_workspace_files(session_id)
            
            logger.info(f"File written to workspace: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write file {file_path} in workspace {session_id}: {e}")
            return False

    def read_file(self, session_id: str, file_path: str) -> Optional[Union[str, bytes]]:
        """Read content from a file in the workspace"""
        workspace_info = self.get_workspace(session_id)
        if not workspace_info:
            logger.error(f"No workspace found for session {session_id}")
            return None
        
        try:
            full_path = workspace_info.workspace_path / file_path
            
            if not full_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Try to read as text first
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # If text reading fails, read as bytes
                with open(full_path, 'rb') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Failed to read file {file_path} from workspace {session_id}: {e}")
            return None

    def list_files(self, session_id: str, directory: str = "") -> List[str]:
        """List files in the workspace directory"""
        workspace_info = self.get_workspace(session_id)
        if not workspace_info:
            logger.error(f"No workspace found for session {session_id}")
            return []
        
        try:
            target_path = workspace_info.workspace_path / directory if directory else workspace_info.workspace_path
            
            if not target_path.exists():
                return []
            
            files = []
            for item in target_path.rglob('*'):
                if item.is_file() and not item.name.startswith('.'):
                    rel_path = item.relative_to(workspace_info.workspace_path)
                    files.append(str(rel_path))
            
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Failed to list files in workspace {session_id}: {e}")
            return []

    def _update_workspace_files(self, session_id: str):
        """Update the list of workspace files for a session."""
        try:
            workspace_info = self.workspaces.get(session_id)
            if workspace_info:
                files = self.list_files(session_id)
                workspace_info.workspace_files = files
                self._save_workspace_metadata(workspace_info)
        except Exception as e:
            logger.debug("Failed to update workspace files for session {%s}: {%s}", session_id, e)

    def cleanup_expired_workspaces(self, max_age_hours: int = 24):
        """Clean up workspaces older than max_age_hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_sessions = []
        
        for session_id, workspace_info in self.workspaces.items():
            if workspace_info.last_activity < cutoff_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired workspace for session {session_id}")
            self.destroy_workspace(session_id, force=True)

    def shutdown(self):
        """Shutdown the workspace manager"""
        logger.info("Shutting down LocalWorkspaceManager...")
        
        if self.cleanup_on_exit:
            # Clean up all workspaces
            session_ids = list(self.workspaces.keys())
            for session_id in session_ids:
                self.destroy_workspace(session_id, force=True)
        else:
            # Just save metadata for all workspaces
            for workspace_info in self.workspaces.values():
                self._save_workspace_metadata(workspace_info)
        
        logger.info("LocalWorkspaceManager shutdown complete")


# Global instance
_workspace_manager: Optional[LocalWorkspaceManager] = None


def get_workspace_manager(base_workspace_dir: str = "workspaces") -> LocalWorkspaceManager:
    """Get or create the global workspace manager instance"""
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = LocalWorkspaceManager(base_workspace_dir)
    return _workspace_manager


def initialize_workspace_manager(base_workspace_dir: str = "workspaces", **kwargs) -> LocalWorkspaceManager:
    """Initialize the workspace manager with custom settings"""
    global _workspace_manager
    _workspace_manager = LocalWorkspaceManager(base_workspace_dir, **kwargs)
    logger.info(f"Workspace manager initialized with base directory: {base_workspace_dir}")
    return _workspace_manager


def shutdown_workspace_manager():
    """Shutdown the global workspace manager"""
    global _workspace_manager
    if _workspace_manager:
        _workspace_manager.shutdown()
        _workspace_manager = None 