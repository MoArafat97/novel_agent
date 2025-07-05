"""
Database Performance Optimizer: Comprehensive optimization utilities for TinyDB and ChromaDB.

This module provides tools for monitoring, optimizing, and maintaining database performance
across the entire worldbuilding system.
"""

import os
import time
import logging
import threading
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import psutil

from .world_state import WorldState
from .tinydb_manager import TinyDBManager
from .chromadb_manager import ChromaDBManager

logger = logging.getLogger(__name__)

class DatabasePerformanceOptimizer:
    """
    Comprehensive database performance optimization and monitoring system.
    """
    
    def __init__(self, world_state: WorldState):
        """
        Initialize performance optimizer.
        
        Args:
            world_state: WorldState instance to optimize
        """
        self.world_state = world_state
        self.optimization_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Performance thresholds
        self.thresholds = {
            'query_time_warning': 1.0,  # seconds
            'query_time_critical': 3.0,
            'cache_hit_rate_warning': 0.7,  # 70%
            'cache_hit_rate_critical': 0.5,  # 50%
            'memory_usage_warning': 500,  # MB
            'memory_usage_critical': 1000,  # MB
        }
        
        logger.info("Database Performance Optimizer initialized")
    
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """
        Run comprehensive database optimization across all systems.
        
        Returns:
            Optimization results and recommendations
        """
        start_time = time.time()
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimizations': [],
            'recommendations': [],
            'performance_gains': {},
            'warnings': []
        }
        
        logger.info("Starting comprehensive database optimization...")
        
        # 1. Memory optimization
        memory_results = self._optimize_memory_usage()
        results['optimizations'].append(memory_results)
        
        # 2. Cache optimization
        cache_results = self._optimize_caches()
        results['optimizations'].append(cache_results)
        
        # 3. Index optimization
        index_results = self._optimize_indexes()
        results['optimizations'].append(index_results)
        
        # 4. Query optimization
        query_results = self._optimize_queries()
        results['optimizations'].append(query_results)
        
        # 5. Embedding optimization
        embedding_results = self._optimize_embeddings()
        results['optimizations'].append(embedding_results)
        
        # 6. Data consistency check
        consistency_results = self._check_data_consistency()
        results['optimizations'].append(consistency_results)
        
        # 7. Generate recommendations
        recommendations = self._generate_recommendations()
        results['recommendations'] = recommendations
        
        # 8. Calculate performance gains
        performance_gains = self._calculate_performance_gains()
        results['performance_gains'] = performance_gains
        
        total_time = time.time() - start_time
        results['optimization_time'] = total_time
        
        # Store optimization history
        self.optimization_history.append(results)
        
        logger.info(f"Comprehensive optimization completed in {total_time:.3f}s")
        return results
    
    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across all database components."""
        start_time = time.time()
        
        # Get current memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        optimizations = []
        
        # Clear expired cache entries
        if hasattr(self.world_state, 'query_cache') and self.world_state.query_cache:
            cache_size_before = len(self.world_state.query_cache)
            # Clear oldest 25% of cache entries
            entries_to_remove = cache_size_before // 4
            for _ in range(entries_to_remove):
                if self.world_state.query_cache:
                    self.world_state.query_cache.popitem(last=False)
            
            optimizations.append(f"Cleared {entries_to_remove} old cache entries")
        
        # Clear embedding cache if too large
        if hasattr(self.world_state, 'embedding_cache'):
            embedding_cache_size = len(self.world_state.embedding_cache)
            if embedding_cache_size > 1000:
                # Keep only most recent 500 entries
                items = list(self.world_state.embedding_cache.items())
                self.world_state.embedding_cache.clear()
                self.world_state.embedding_cache.update(items[-500:])
                optimizations.append(f"Trimmed embedding cache from {embedding_cache_size} to 500 entries")
        
        # Force garbage collection
        collected = gc.collect()
        optimizations.append(f"Garbage collected {collected} objects")
        
        # Get memory usage after optimization
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_saved = memory_before - memory_after
        
        return {
            'type': 'memory_optimization',
            'time': time.time() - start_time,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_saved_mb': memory_saved,
            'optimizations': optimizations
        }
    
    def _optimize_caches(self) -> Dict[str, Any]:
        """Optimize cache configurations and performance."""
        start_time = time.time()
        optimizations = []
        
        # Get cache statistics
        cache_stats = {}
        if hasattr(self.world_state, 'cache_stats'):
            cache_stats = self.world_state.cache_stats.copy()
        
        # Calculate hit rates
        total_requests = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
        hit_rate = cache_stats.get('hits', 0) / total_requests if total_requests > 0 else 0
        
        # Optimize cache size based on hit rate
        if hit_rate < self.thresholds['cache_hit_rate_warning']:
            # Increase cache size if hit rate is low
            if hasattr(self.world_state, 'cache_size'):
                old_size = self.world_state.cache_size
                self.world_state.cache_size = min(old_size * 2, 2000)
                optimizations.append(f"Increased cache size from {old_size} to {self.world_state.cache_size}")
        
        # Warm up cache with frequently accessed data
        if hasattr(self.world_state, 'tinydb_connections'):
            for entity_type in self.world_state.entity_types:
                # Pre-load recent entities into cache
                db = self.world_state.tinydb_connections[entity_type]
                recent_entities = db.all()[-50:]  # Last 50 entities
                for entity in recent_entities:
                    cache_key = self.world_state._get_cache_key("get_entity", 
                                                              entity_type=entity_type, 
                                                              entity_id=entity.get('id'))
                    self.world_state._cache_set(cache_key, entity)
            
            optimizations.append("Warmed up cache with recent entities")
        
        return {
            'type': 'cache_optimization',
            'time': time.time() - start_time,
            'hit_rate_before': hit_rate,
            'cache_stats': cache_stats,
            'optimizations': optimizations
        }
    
    def _optimize_indexes(self) -> Dict[str, Any]:
        """Optimize database indexes for better query performance."""
        start_time = time.time()
        optimizations = []
        
        # Rebuild indexes if enabled
        if hasattr(self.world_state, 'enable_indexing') and self.world_state.enable_indexing:
            # Clear existing indexes
            for index_type in self.world_state.indexes:
                self.world_state.indexes[index_type].clear()
            
            # Rebuild indexes
            self.world_state._build_indexes()
            optimizations.append("Rebuilt all database indexes")
        
        # Optimize TinyDB manager indexes if available
        if hasattr(self.world_state, 'tinydb_connections'):
            for entity_type in self.world_state.entity_types:
                db = self.world_state.tinydb_connections[entity_type]
                entity_count = len(db.all())
                optimizations.append(f"Indexed {entity_count} {entity_type} entities")
        
        return {
            'type': 'index_optimization',
            'time': time.time() - start_time,
            'optimizations': optimizations
        }
    
    def _optimize_queries(self) -> Dict[str, Any]:
        """Optimize query patterns and performance."""
        start_time = time.time()
        optimizations = []
        
        # Analyze query performance
        query_times = []
        if hasattr(self.world_state, 'performance_metrics'):
            query_times = self.world_state.performance_metrics.get('query_times', [])
        
        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            max_query_time = max(query_times)
            
            optimizations.append(f"Analyzed {len(query_times)} queries")
            optimizations.append(f"Average query time: {avg_query_time:.3f}s")
            optimizations.append(f"Maximum query time: {max_query_time:.3f}s")
            
            # Warn about slow queries
            if avg_query_time > self.thresholds['query_time_warning']:
                optimizations.append(f"WARNING: Average query time exceeds {self.thresholds['query_time_warning']}s")
        
        return {
            'type': 'query_optimization',
            'time': time.time() - start_time,
            'query_stats': {
                'count': len(query_times),
                'avg_time': sum(query_times) / len(query_times) if query_times else 0,
                'max_time': max(query_times) if query_times else 0
            },
            'optimizations': optimizations
        }
    
    def _optimize_embeddings(self) -> Dict[str, Any]:
        """Optimize embedding generation and caching."""
        start_time = time.time()
        optimizations = []
        
        # Analyze embedding performance
        embedding_times = []
        if hasattr(self.world_state, 'performance_metrics'):
            embedding_times = self.world_state.performance_metrics.get('embedding_times', [])
        
        if embedding_times:
            avg_embedding_time = sum(embedding_times) / len(embedding_times)
            optimizations.append(f"Analyzed {len(embedding_times)} embedding generations")
            optimizations.append(f"Average embedding time: {avg_embedding_time:.3f}s")
        
        # Check embedding cache efficiency
        if hasattr(self.world_state, 'embedding_cache_stats'):
            stats = self.world_state.embedding_cache_stats
            total_requests = stats.get('hits', 0) + stats.get('misses', 0)
            hit_rate = stats.get('hits', 0) / total_requests if total_requests > 0 else 0
            optimizations.append(f"Embedding cache hit rate: {hit_rate:.2%}")
        
        return {
            'type': 'embedding_optimization',
            'time': time.time() - start_time,
            'embedding_stats': {
                'count': len(embedding_times),
                'avg_time': sum(embedding_times) / len(embedding_times) if embedding_times else 0
            },
            'optimizations': optimizations
        }
    
    def _check_data_consistency(self) -> Dict[str, Any]:
        """Check and repair data consistency between TinyDB and ChromaDB."""
        start_time = time.time()
        issues = []
        repairs = []
        
        try:
            for entity_type in self.world_state.entity_types:
                # Get entities from TinyDB
                tinydb_entities = self.world_state.tinydb_connections[entity_type].all()
                tinydb_ids = {entity.get('id') for entity in tinydb_entities if entity.get('id')}
                
                # Get entities from ChromaDB
                chroma_collection = self.world_state.chroma_collections[entity_type]
                try:
                    chroma_data = chroma_collection.get()
                    chroma_ids = set(chroma_data['ids']) if chroma_data['ids'] else set()
                except:
                    chroma_ids = set()
                
                # Find inconsistencies
                missing_in_chroma = tinydb_ids - chroma_ids
                missing_in_tinydb = chroma_ids - tinydb_ids
                
                if missing_in_chroma:
                    issues.append(f"{len(missing_in_chroma)} {entity_type} entities missing from ChromaDB")
                
                if missing_in_tinydb:
                    issues.append(f"{len(missing_in_tinydb)} {entity_type} entities missing from TinyDB")
                    # Remove orphaned ChromaDB entries
                    try:
                        chroma_collection.delete(ids=list(missing_in_tinydb))
                        repairs.append(f"Removed {len(missing_in_tinydb)} orphaned {entity_type} embeddings")
                    except Exception as e:
                        logger.warning(f"Failed to remove orphaned embeddings: {e}")
        
        except Exception as e:
            issues.append(f"Consistency check failed: {e}")
        
        return {
            'type': 'data_consistency',
            'time': time.time() - start_time,
            'issues': issues,
            'repairs': repairs
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check cache performance
        if hasattr(self.world_state, 'cache_stats'):
            stats = self.world_state.cache_stats
            total_requests = stats.get('hits', 0) + stats.get('misses', 0)
            hit_rate = stats.get('hits', 0) / total_requests if total_requests > 0 else 0
            
            if hit_rate < self.thresholds['cache_hit_rate_critical']:
                recommendations.append("CRITICAL: Cache hit rate is very low. Consider increasing cache size or reviewing query patterns.")
            elif hit_rate < self.thresholds['cache_hit_rate_warning']:
                recommendations.append("WARNING: Cache hit rate is below optimal. Consider tuning cache configuration.")
        
        # Check memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.thresholds['memory_usage_critical']:
                recommendations.append("CRITICAL: High memory usage detected. Consider reducing cache sizes or optimizing data structures.")
            elif memory_mb > self.thresholds['memory_usage_warning']:
                recommendations.append("WARNING: Elevated memory usage. Monitor for memory leaks.")
        except:
            pass
        
        # Check query performance
        if hasattr(self.world_state, 'performance_metrics'):
            query_times = self.world_state.performance_metrics.get('query_times', [])
            if query_times:
                avg_time = sum(query_times) / len(query_times)
                if avg_time > self.thresholds['query_time_critical']:
                    recommendations.append("CRITICAL: Query performance is very slow. Review indexing and query optimization.")
                elif avg_time > self.thresholds['query_time_warning']:
                    recommendations.append("WARNING: Query performance could be improved. Consider index optimization.")
        
        return recommendations
    
    def _calculate_performance_gains(self) -> Dict[str, float]:
        """Calculate performance improvements from optimization."""
        gains = {}
        
        if len(self.optimization_history) >= 2:
            current = self.optimization_history[-1]
            previous = self.optimization_history[-2]
            
            # Calculate memory improvement
            current_memory = current.get('optimizations', [{}])[0].get('memory_after_mb', 0)
            previous_memory = previous.get('optimizations', [{}])[0].get('memory_after_mb', 0)
            
            if previous_memory > 0:
                memory_improvement = (previous_memory - current_memory) / previous_memory
                gains['memory_improvement'] = memory_improvement
        
        return gains
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'current_performance': self.world_state.get_performance_stats() if hasattr(self.world_state, 'get_performance_stats') else {},
            'thresholds': self.thresholds,
            'recommendations': self._generate_recommendations()
        }
