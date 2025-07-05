"""
Database Monitoring: Real-time performance monitoring and alerting for database operations.

This module provides comprehensive monitoring capabilities for TinyDB and ChromaDB performance,
including real-time metrics, alerting, and performance trend analysis.
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import deque, defaultdict
import json
import psutil

logger = logging.getLogger(__name__)

class DatabaseMonitor:
    """
    Real-time database performance monitoring system.
    """
    
    def __init__(self, world_state, alert_callback: Optional[Callable] = None):
        """
        Initialize database monitor.
        
        Args:
            world_state: WorldState instance to monitor
            alert_callback: Optional callback function for alerts
        """
        self.world_state = world_state
        self.alert_callback = alert_callback
        
        # Monitoring configuration
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # seconds
        
        # Performance metrics storage
        self.metrics_history = {
            'query_times': deque(maxlen=1000),
            'embedding_times': deque(maxlen=1000),
            'cache_hit_rates': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'error_counts': deque(maxlen=1000),
            'timestamps': deque(maxlen=1000)
        }
        
        # Alert thresholds
        self.thresholds = {
            'query_time_warning': 1.0,
            'query_time_critical': 3.0,
            'cache_hit_rate_warning': 0.7,
            'cache_hit_rate_critical': 0.5,
            'memory_usage_warning': 500,  # MB
            'memory_usage_critical': 1000,  # MB
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.1   # 10%
        }
        
        # Alert state tracking
        self.alert_states = defaultdict(bool)
        self.last_alert_times = defaultdict(float)
        self.alert_cooldown = 300  # 5 minutes
        
        logger.info("Database Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Database monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("Database monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Sleep until next collection
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        metrics = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        # Query performance metrics
        if hasattr(self.world_state, 'performance_metrics'):
            perf_metrics = self.world_state.performance_metrics
            
            query_times = perf_metrics.get('query_times', [])
            if query_times:
                metrics['avg_query_time'] = sum(query_times[-10:]) / len(query_times[-10:])
                metrics['max_query_time'] = max(query_times[-10:])
            else:
                metrics['avg_query_time'] = 0
                metrics['max_query_time'] = 0
            
            embedding_times = perf_metrics.get('embedding_times', [])
            if embedding_times:
                metrics['avg_embedding_time'] = sum(embedding_times[-10:]) / len(embedding_times[-10:])
            else:
                metrics['avg_embedding_time'] = 0
        
        # Cache performance metrics
        if hasattr(self.world_state, 'cache_stats'):
            cache_stats = self.world_state.cache_stats
            total_requests = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
            metrics['cache_hit_rate'] = cache_stats.get('hits', 0) / total_requests if total_requests > 0 else 0
            metrics['cache_size'] = len(self.world_state.query_cache) if hasattr(self.world_state, 'query_cache') and self.world_state.query_cache else 0
        else:
            metrics['cache_hit_rate'] = 0
            metrics['cache_size'] = 0
        
        # Memory usage
        try:
            process = psutil.Process()
            metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            metrics['cpu_percent'] = process.cpu_percent()
        except:
            metrics['memory_usage_mb'] = 0
            metrics['cpu_percent'] = 0
        
        # Database sizes
        metrics['database_sizes'] = {}
        if hasattr(self.world_state, 'tinydb_connections'):
            for entity_type in self.world_state.entity_types:
                try:
                    db = self.world_state.tinydb_connections[entity_type]
                    metrics['database_sizes'][entity_type] = len(db.all())
                except:
                    metrics['database_sizes'][entity_type] = 0
        
        # ChromaDB collection sizes
        metrics['embedding_counts'] = {}
        if hasattr(self.world_state, 'chroma_collections'):
            for entity_type in self.world_state.entity_types:
                try:
                    collection = self.world_state.chroma_collections[entity_type]
                    metrics['embedding_counts'][entity_type] = collection.count()
                except:
                    metrics['embedding_counts'][entity_type] = 0
        
        return metrics
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in history."""
        timestamp = metrics['timestamp']
        
        self.metrics_history['timestamps'].append(timestamp)
        self.metrics_history['query_times'].append(metrics.get('avg_query_time', 0))
        self.metrics_history['embedding_times'].append(metrics.get('avg_embedding_time', 0))
        self.metrics_history['cache_hit_rates'].append(metrics.get('cache_hit_rate', 0))
        self.metrics_history['memory_usage'].append(metrics.get('memory_usage_mb', 0))
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and trigger alerts."""
        current_time = time.time()
        
        # Query time alerts
        avg_query_time = metrics.get('avg_query_time', 0)
        if avg_query_time > self.thresholds['query_time_critical']:
            self._trigger_alert('query_time_critical', f"Critical query performance: {avg_query_time:.3f}s average", metrics)
        elif avg_query_time > self.thresholds['query_time_warning']:
            self._trigger_alert('query_time_warning', f"Slow query performance: {avg_query_time:.3f}s average", metrics)
        
        # Cache hit rate alerts
        cache_hit_rate = metrics.get('cache_hit_rate', 0)
        if cache_hit_rate < self.thresholds['cache_hit_rate_critical']:
            self._trigger_alert('cache_hit_rate_critical', f"Critical cache performance: {cache_hit_rate:.1%} hit rate", metrics)
        elif cache_hit_rate < self.thresholds['cache_hit_rate_warning']:
            self._trigger_alert('cache_hit_rate_warning', f"Low cache performance: {cache_hit_rate:.1%} hit rate", metrics)
        
        # Memory usage alerts
        memory_usage = metrics.get('memory_usage_mb', 0)
        if memory_usage > self.thresholds['memory_usage_critical']:
            self._trigger_alert('memory_usage_critical', f"Critical memory usage: {memory_usage:.1f}MB", metrics)
        elif memory_usage > self.thresholds['memory_usage_warning']:
            self._trigger_alert('memory_usage_warning', f"High memory usage: {memory_usage:.1f}MB", metrics)
    
    def _trigger_alert(self, alert_type: str, message: str, metrics: Dict[str, Any]):
        """Trigger an alert if not in cooldown period."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_times[alert_type] < self.alert_cooldown:
            return
        
        # Update alert state
        self.alert_states[alert_type] = True
        self.last_alert_times[alert_type] = current_time
        
        # Log alert
        logger.warning(f"DATABASE ALERT [{alert_type}]: {message}")
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert_type, message, metrics)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self._collect_metrics()
    
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance metrics summary for specified time window.
        
        Args:
            time_window_minutes: Time window in minutes
            
        Returns:
            Metrics summary
        """
        if not self.metrics_history['timestamps']:
            return {}
        
        # Calculate time window
        current_time = time.time()
        window_start = current_time - (time_window_minutes * 60)
        
        # Filter metrics within time window
        timestamps = list(self.metrics_history['timestamps'])
        indices = [i for i, ts in enumerate(timestamps) if ts >= window_start]
        
        if not indices:
            return {}
        
        # Calculate summary statistics
        query_times = [self.metrics_history['query_times'][i] for i in indices]
        embedding_times = [self.metrics_history['embedding_times'][i] for i in indices]
        cache_hit_rates = [self.metrics_history['cache_hit_rates'][i] for i in indices]
        memory_usage = [self.metrics_history['memory_usage'][i] for i in indices]
        
        summary = {
            'time_window_minutes': time_window_minutes,
            'sample_count': len(indices),
            'query_performance': {
                'avg_time': sum(query_times) / len(query_times) if query_times else 0,
                'max_time': max(query_times) if query_times else 0,
                'min_time': min(query_times) if query_times else 0
            },
            'embedding_performance': {
                'avg_time': sum(embedding_times) / len(embedding_times) if embedding_times else 0,
                'max_time': max(embedding_times) if embedding_times else 0
            },
            'cache_performance': {
                'avg_hit_rate': sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0,
                'min_hit_rate': min(cache_hit_rates) if cache_hit_rates else 0
            },
            'memory_usage': {
                'avg_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'max_mb': max(memory_usage) if memory_usage else 0,
                'min_mb': min(memory_usage) if memory_usage else 0
            }
        }
        
        return summary
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.metrics_history['timestamps']) < 10:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Get recent metrics (last 50% of data)
        half_point = len(self.metrics_history['timestamps']) // 2
        
        recent_query_times = list(self.metrics_history['query_times'])[half_point:]
        older_query_times = list(self.metrics_history['query_times'])[:half_point]
        
        recent_cache_rates = list(self.metrics_history['cache_hit_rates'])[half_point:]
        older_cache_rates = list(self.metrics_history['cache_hit_rates'])[:half_point]
        
        recent_memory = list(self.metrics_history['memory_usage'])[half_point:]
        older_memory = list(self.metrics_history['memory_usage'])[:half_point]
        
        # Calculate trends
        trends = {}
        
        if recent_query_times and older_query_times:
            recent_avg = sum(recent_query_times) / len(recent_query_times)
            older_avg = sum(older_query_times) / len(older_query_times)
            trends['query_time_trend'] = 'improving' if recent_avg < older_avg else 'degrading'
            trends['query_time_change'] = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        
        if recent_cache_rates and older_cache_rates:
            recent_avg = sum(recent_cache_rates) / len(recent_cache_rates)
            older_avg = sum(older_cache_rates) / len(older_cache_rates)
            trends['cache_trend'] = 'improving' if recent_avg > older_avg else 'degrading'
            trends['cache_change'] = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        
        if recent_memory and older_memory:
            recent_avg = sum(recent_memory) / len(recent_memory)
            older_avg = sum(older_memory) / len(older_memory)
            trends['memory_trend'] = 'improving' if recent_avg < older_avg else 'degrading'
            trends['memory_change'] = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        
        return trends
    
    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics_history': {
                    'timestamps': list(self.metrics_history['timestamps']),
                    'query_times': list(self.metrics_history['query_times']),
                    'embedding_times': list(self.metrics_history['embedding_times']),
                    'cache_hit_rates': list(self.metrics_history['cache_hit_rates']),
                    'memory_usage': list(self.metrics_history['memory_usage'])
                },
                'thresholds': self.thresholds,
                'alert_states': dict(self.alert_states)
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall database health status."""
        current_metrics = self._collect_metrics()
        
        # Determine health status based on current metrics
        health_score = 100
        issues = []
        
        # Check query performance
        avg_query_time = current_metrics.get('avg_query_time', 0)
        if avg_query_time > self.thresholds['query_time_critical']:
            health_score -= 30
            issues.append('Critical query performance')
        elif avg_query_time > self.thresholds['query_time_warning']:
            health_score -= 15
            issues.append('Slow query performance')
        
        # Check cache performance
        cache_hit_rate = current_metrics.get('cache_hit_rate', 0)
        if cache_hit_rate < self.thresholds['cache_hit_rate_critical']:
            health_score -= 25
            issues.append('Critical cache performance')
        elif cache_hit_rate < self.thresholds['cache_hit_rate_warning']:
            health_score -= 10
            issues.append('Low cache performance')
        
        # Check memory usage
        memory_usage = current_metrics.get('memory_usage_mb', 0)
        if memory_usage > self.thresholds['memory_usage_critical']:
            health_score -= 20
            issues.append('Critical memory usage')
        elif memory_usage > self.thresholds['memory_usage_warning']:
            health_score -= 10
            issues.append('High memory usage')
        
        # Determine status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 50:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'status': status,
            'health_score': max(0, health_score),
            'issues': issues,
            'metrics': current_metrics,
            'monitoring_active': self.monitoring_active
        }
