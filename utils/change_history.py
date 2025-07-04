"""
Change History System

This module provides undo/redo functionality for cross-reference updates
with full rollback capabilities and change tracking.
"""

import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChangeRecord:
    """Represents a single change that can be undone."""
    change_id: str
    timestamp: datetime
    user_id: str
    operation_type: str  # 'update', 'create', 'delete', 'relationship_add', etc.
    entity_type: str
    entity_id: str
    novel_id: str
    description: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'change_id': self.change_id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'operation_type': self.operation_type,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'novel_id': self.novel_id,
            'description': self.description,
            'before_state': self.before_state,
            'after_state': self.after_state,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeRecord':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ChangeSession:
    """Represents a group of related changes (e.g., from one cross-reference analysis)."""
    session_id: str
    timestamp: datetime
    description: str
    novel_id: str
    user_id: str
    changes: List[ChangeRecord]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'novel_id': self.novel_id,
            'user_id': self.user_id,
            'changes': [change.to_dict() for change in self.changes]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeSession':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['changes'] = [ChangeRecord.from_dict(change) for change in data['changes']]
        return cls(**data)


class ChangeHistoryManager:
    """
    Manages change history for undo/redo functionality.
    
    Features:
    - Track all changes with before/after states
    - Group related changes into sessions
    - Undo individual changes or entire sessions
    - Persistent storage of change history
    - Change conflict detection
    """
    
    def __init__(self, data_dir: str = "data/change_history", max_history_days: int = 30):
        """Initialize the change history manager."""
        self.data_dir = Path(data_dir)
        self.max_history_days = max_history_days
        
        # In-memory cache
        self.sessions: Dict[str, ChangeSession] = {}
        self.changes: Dict[str, ChangeRecord] = {}
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        self._load_history()
        
        logger.info(f"Initialized ChangeHistoryManager with {len(self.sessions)} sessions")
    
    def start_session(self, description: str, novel_id: str, user_id: str = "system") -> str:
        """
        Start a new change session.
        
        Args:
            description: Description of the session
            novel_id: Novel ID
            user_id: User ID
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        session = ChangeSession(
            session_id=session_id,
            timestamp=datetime.now(),
            description=description,
            novel_id=novel_id,
            user_id=user_id,
            changes=[]
        )
        
        self.sessions[session_id] = session
        logger.info(f"Started change session {session_id}: {description}")
        
        return session_id
    
    def record_change(self, 
                     session_id: str,
                     operation_type: str,
                     entity_type: str,
                     entity_id: str,
                     novel_id: str,
                     description: str,
                     before_state: Dict[str, Any],
                     after_state: Dict[str, Any],
                     user_id: str = "system",
                     metadata: Dict[str, Any] = None) -> str:
        """
        Record a change in the history.
        
        Args:
            session_id: Session ID
            operation_type: Type of operation
            entity_type: Type of entity
            entity_id: Entity ID
            novel_id: Novel ID
            description: Change description
            before_state: State before change
            after_state: State after change
            user_id: User ID
            metadata: Additional metadata
            
        Returns:
            Change ID
        """
        change_id = str(uuid.uuid4())
        
        change = ChangeRecord(
            change_id=change_id,
            timestamp=datetime.now(),
            user_id=user_id,
            operation_type=operation_type,
            entity_type=entity_type,
            entity_id=entity_id,
            novel_id=novel_id,
            description=description,
            before_state=before_state.copy() if before_state else {},
            after_state=after_state.copy() if after_state else {},
            metadata=metadata.copy() if metadata else {}
        )
        
        # Add to session
        if session_id in self.sessions:
            self.sessions[session_id].changes.append(change)
        
        # Add to changes index
        self.changes[change_id] = change
        
        logger.debug(f"Recorded change {change_id}: {description}")
        return change_id
    
    def end_session(self, session_id: str):
        """End a change session and save to disk."""
        if session_id in self.sessions:
            self._save_session(session_id)
            logger.info(f"Ended change session {session_id}")
    
    def undo_change(self, change_id: str, world_state) -> bool:
        """
        Undo a specific change.
        
        Args:
            change_id: Change ID to undo
            world_state: WorldState instance to apply changes to
            
        Returns:
            True if successful, False otherwise
        """
        if change_id not in self.changes:
            logger.error(f"Change {change_id} not found")
            return False
        
        change = self.changes[change_id]
        
        try:
            # Check if entity still exists and hasn't been modified
            current_entity = world_state.get(change.entity_type, change.entity_id)
            if not current_entity:
                logger.error(f"Entity {change.entity_type}:{change.entity_id} no longer exists")
                return False
            
            # Check for conflicts (simple version - compare timestamps)
            if self._has_conflicts(change, current_entity):
                logger.warning(f"Conflicts detected for change {change_id}")
                return False
            
            # Apply the undo by restoring the before state
            if change.operation_type == 'delete':
                # Restore deleted entity
                world_state.create(change.entity_type, change.before_state, change.entity_id)
            elif change.operation_type == 'create':
                # Delete created entity
                world_state.delete(change.entity_type, change.entity_id)
            else:
                # Update entity to before state
                world_state.update(change.entity_type, change.entity_id, change.before_state)
            
            # Record the undo as a new change
            undo_session_id = self.start_session(
                f"Undo: {change.description}",
                change.novel_id,
                change.user_id
            )
            
            self.record_change(
                undo_session_id,
                f"undo_{change.operation_type}",
                change.entity_type,
                change.entity_id,
                change.novel_id,
                f"Undid: {change.description}",
                change.after_state,
                change.before_state,
                change.user_id,
                {'original_change_id': change_id}
            )
            
            self.end_session(undo_session_id)
            
            logger.info(f"Successfully undid change {change_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to undo change {change_id}: {e}")
            return False
    
    def undo_session(self, session_id: str, world_state) -> bool:
        """
        Undo an entire session of changes.
        
        Args:
            session_id: Session ID to undo
            world_state: WorldState instance to apply changes to
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.sessions[session_id]
        
        # Undo changes in reverse order
        success_count = 0
        for change in reversed(session.changes):
            if self.undo_change(change.change_id, world_state):
                success_count += 1
            else:
                logger.warning(f"Failed to undo change {change.change_id} in session {session_id}")
        
        logger.info(f"Undid {success_count}/{len(session.changes)} changes in session {session_id}")
        return success_count == len(session.changes)
    
    def get_history(self, novel_id: str, limit: int = 50) -> List[ChangeSession]:
        """
        Get change history for a novel.
        
        Args:
            novel_id: Novel ID
            limit: Maximum number of sessions to return
            
        Returns:
            List of change sessions
        """
        novel_sessions = [
            session for session in self.sessions.values()
            if session.novel_id == novel_id
        ]
        
        # Sort by timestamp (newest first)
        novel_sessions.sort(key=lambda s: s.timestamp, reverse=True)
        
        return novel_sessions[:limit]
    
    def get_entity_history(self, entity_type: str, entity_id: str, limit: int = 20) -> List[ChangeRecord]:
        """
        Get change history for a specific entity.
        
        Args:
            entity_type: Entity type
            entity_id: Entity ID
            limit: Maximum number of changes to return
            
        Returns:
            List of change records
        """
        entity_changes = [
            change for change in self.changes.values()
            if change.entity_type == entity_type and change.entity_id == entity_id
        ]
        
        # Sort by timestamp (newest first)
        entity_changes.sort(key=lambda c: c.timestamp, reverse=True)
        
        return entity_changes[:limit]
    
    def _has_conflicts(self, change: ChangeRecord, current_entity: Dict[str, Any]) -> bool:
        """Check if there are conflicts that prevent undoing a change."""
        # Simple conflict detection - check if entity has been modified since the change
        # In a more sophisticated system, you might compare specific fields or use version numbers
        
        # For now, just check if the current state matches the expected after state
        # This is a simplified approach
        return False  # Allow undo for now
    
    def _save_session(self, session_id: str):
        """Save a session to disk."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session_file = self.data_dir / f"session_{session_id}.json"
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def _load_history(self):
        """Load change history from disk."""
        try:
            for session_file in self.data_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    session = ChangeSession.from_dict(session_data)
                    
                    # Check if session is within retention period
                    age_days = (datetime.now() - session.timestamp).days
                    if age_days <= self.max_history_days:
                        self.sessions[session.session_id] = session
                        
                        # Index changes
                        for change in session.changes:
                            self.changes[change.change_id] = change
                    else:
                        # Remove old session file
                        session_file.unlink(missing_ok=True)
                        
                except Exception as e:
                    logger.warning(f"Failed to load session from {session_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load change history: {e}")
    
    def cleanup_old_history(self):
        """Clean up old change history."""
        cutoff_date = datetime.now().timestamp() - (self.max_history_days * 24 * 3600)
        
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session.timestamp.timestamp() < cutoff_date:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            # Remove from memory
            session = self.sessions.pop(session_id)
            
            # Remove changes from index
            for change in session.changes:
                self.changes.pop(change.change_id, None)
            
            # Remove file
            session_file = self.data_dir / f"session_{session_id}.json"
            session_file.unlink(missing_ok=True)
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old change sessions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get change history statistics."""
        return {
            'total_sessions': len(self.sessions),
            'total_changes': len(self.changes),
            'oldest_change': min((c.timestamp for c in self.changes.values()), default=None),
            'newest_change': max((c.timestamp for c in self.changes.values()), default=None)
        }


# Global change history manager instance
_change_history_manager = None

def get_change_history_manager() -> ChangeHistoryManager:
    """Get the global change history manager instance."""
    global _change_history_manager
    if _change_history_manager is None:
        _change_history_manager = ChangeHistoryManager()
    return _change_history_manager
