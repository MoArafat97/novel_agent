"""
Streaming Analysis System

This module provides real-time streaming of cross-reference analysis results
using Server-Sent Events (SSE) for better user experience.
"""

import json
import time
import uuid
import logging
import threading
from typing import Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, asdict
from enum import Enum
from queue import Queue, Empty
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalysisStage(Enum):
    """Analysis stages for progress tracking."""
    INITIALIZING = "initializing"
    EXTRACTING_CONTENT = "extracting_content"
    DETECTING_ENTITIES = "detecting_entities"
    CLASSIFYING_ENTITIES = "classifying_entities"
    FINDING_RELATIONSHIPS = "finding_relationships"
    GENERATING_UPDATES = "generating_updates"
    VERIFYING_RESULTS = "verifying_results"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    """Progress update for streaming."""
    job_id: str
    stage: AnalysisStage
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'job_id': self.job_id,
            'stage': self.stage.value,
            'progress': self.progress,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data or {}
        }


@dataclass
class AnalysisJob:
    """Analysis job tracking."""
    job_id: str
    entity_type: str
    entity_id: str
    novel_id: str
    status: AnalysisStage
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancelled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'novel_id': self.novel_id,
            'status': self.status.value,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error': self.error,
            'cancelled': self.cancelled
        }


class StreamingAnalysisManager:
    """
    Manages streaming analysis jobs with real-time progress updates.
    """
    
    def __init__(self):
        """Initialize the streaming analysis manager."""
        self.jobs: Dict[str, AnalysisJob] = {}
        self.progress_queues: Dict[str, Queue] = {}
        self.job_lock = threading.Lock()
        self.cleanup_interval = 3600  # 1 hour
        self.max_job_age = 7200  # 2 hours
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info("Initialized StreamingAnalysisManager")
    
    def create_job(self, entity_type: str, entity_id: str, novel_id: str) -> str:
        """
        Create a new analysis job.
        
        Args:
            entity_type: Type of entity to analyze
            entity_id: ID of entity to analyze
            novel_id: Novel ID
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        with self.job_lock:
            job = AnalysisJob(
                job_id=job_id,
                entity_type=entity_type,
                entity_id=entity_id,
                novel_id=novel_id,
                status=AnalysisStage.INITIALIZING,
                progress=0.0,
                created_at=datetime.now()
            )
            
            self.jobs[job_id] = job
            self.progress_queues[job_id] = Queue()
        
        logger.info(f"Created analysis job {job_id} for {entity_type}:{entity_id}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        """Get job by ID."""
        with self.job_lock:
            return self.jobs.get(job_id)
    
    def update_progress(self, job_id: str, stage: AnalysisStage, progress: float, 
                       message: str, data: Optional[Dict[str, Any]] = None):
        """
        Update job progress and notify listeners.
        
        Args:
            job_id: Job ID
            stage: Current analysis stage
            progress: Progress percentage (0.0 to 1.0)
            message: Progress message
            data: Optional additional data
        """
        with self.job_lock:
            job = self.jobs.get(job_id)
            if not job or job.cancelled:
                return
            
            job.status = stage
            job.progress = progress
            
            if stage == AnalysisStage.INITIALIZING and not job.started_at:
                job.started_at = datetime.now()
            elif stage in [AnalysisStage.COMPLETED, AnalysisStage.ERROR, AnalysisStage.CANCELLED]:
                job.completed_at = datetime.now()
            
            # Create progress update
            update = ProgressUpdate(
                job_id=job_id,
                stage=stage,
                progress=progress,
                message=message,
                timestamp=datetime.now(),
                data=data
            )
            
            # Add to progress queue
            if job_id in self.progress_queues:
                try:
                    self.progress_queues[job_id].put_nowait(update)
                except:
                    # Queue full, skip this update
                    pass
        
        logger.debug(f"Job {job_id}: {stage.value} - {progress:.1%} - {message}")
    
    def set_job_result(self, job_id: str, result: Dict[str, Any]):
        """Set job result and mark as completed."""
        with self.job_lock:
            job = self.jobs.get(job_id)
            if job and not job.cancelled:
                job.result = result
                job.status = AnalysisStage.COMPLETED
                job.progress = 1.0
                job.completed_at = datetime.now()
    
    def set_job_error(self, job_id: str, error: str):
        """Set job error and mark as failed."""
        with self.job_lock:
            job = self.jobs.get(job_id)
            if job and not job.cancelled:
                job.error = error
                job.status = AnalysisStage.ERROR
                job.completed_at = datetime.now()
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled, False if not found or already completed
        """
        with self.job_lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            
            if job.status in [AnalysisStage.COMPLETED, AnalysisStage.ERROR, AnalysisStage.CANCELLED]:
                return False
            
            job.cancelled = True
            job.status = AnalysisStage.CANCELLED
            job.completed_at = datetime.now()
            
            # Notify listeners
            self.update_progress(job_id, AnalysisStage.CANCELLED, job.progress, "Analysis cancelled by user")
        
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def get_progress_stream(self, job_id: str) -> Generator[str, None, None]:
        """
        Get Server-Sent Events stream for job progress.
        
        Args:
            job_id: Job ID to stream
            
        Yields:
            SSE formatted progress updates
        """
        if job_id not in self.progress_queues:
            yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
            return
        
        queue = self.progress_queues[job_id]
        timeout = 30  # 30 second timeout
        
        try:
            while True:
                try:
                    # Get progress update with timeout
                    update = queue.get(timeout=timeout)
                    
                    # Format as SSE
                    data = json.dumps(update.to_dict())
                    yield f"data: {data}\n\n"
                    
                    # Check if job is complete
                    if update.stage in [AnalysisStage.COMPLETED, AnalysisStage.ERROR, AnalysisStage.CANCELLED]:
                        break
                        
                except Empty:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.now().isoformat()})}\n\n"
                    
                    # Check if job still exists
                    job = self.get_job(job_id)
                    if not job:
                        break
                    
                    # Check if job is complete (in case we missed the final update)
                    if job.status in [AnalysisStage.COMPLETED, AnalysisStage.ERROR, AnalysisStage.CANCELLED]:
                        final_update = ProgressUpdate(
                            job_id=job_id,
                            stage=job.status,
                            progress=job.progress,
                            message=f"Analysis {job.status.value}",
                            timestamp=datetime.now(),
                            data={'result': job.result, 'error': job.error}
                        )
                        data = json.dumps(final_update.to_dict())
                        yield f"data: {data}\n\n"
                        break
                        
        except Exception as e:
            logger.error(f"Error in progress stream for job {job_id}: {e}")
            error_data = json.dumps({
                'job_id': job_id,
                'stage': 'error',
                'progress': 0.0,
                'message': f'Stream error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
            yield f"data: {error_data}\n\n"
        
        finally:
            # Cleanup
            self._cleanup_job_queue(job_id)
    
    def _cleanup_job_queue(self, job_id: str):
        """Clean up job queue."""
        with self.job_lock:
            if job_id in self.progress_queues:
                del self.progress_queues[job_id]
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self._cleanup_old_jobs()
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_jobs(self):
        """Clean up old completed jobs."""
        cutoff_time = datetime.now().timestamp() - self.max_job_age
        
        with self.job_lock:
            jobs_to_remove = []
            
            for job_id, job in self.jobs.items():
                if (job.completed_at and 
                    job.completed_at.timestamp() < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                if job_id in self.progress_queues:
                    del self.progress_queues[job_id]
            
            if jobs_to_remove:
                logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    def get_job_stats(self) -> Dict[str, Any]:
        """Get job statistics."""
        with self.job_lock:
            total_jobs = len(self.jobs)
            active_jobs = sum(1 for job in self.jobs.values() 
                            if job.status not in [AnalysisStage.COMPLETED, AnalysisStage.ERROR, AnalysisStage.CANCELLED])
            completed_jobs = sum(1 for job in self.jobs.values() if job.status == AnalysisStage.COMPLETED)
            failed_jobs = sum(1 for job in self.jobs.values() if job.status == AnalysisStage.ERROR)
            cancelled_jobs = sum(1 for job in self.jobs.values() if job.status == AnalysisStage.CANCELLED)
            
            return {
                'total_jobs': total_jobs,
                'active_jobs': active_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'cancelled_jobs': cancelled_jobs,
                'active_streams': len(self.progress_queues)
            }


# Global streaming manager instance
_streaming_manager = None

def get_streaming_manager() -> StreamingAnalysisManager:
    """Get the global streaming analysis manager instance."""
    global _streaming_manager
    if _streaming_manager is None:
        _streaming_manager = StreamingAnalysisManager()
    return _streaming_manager
