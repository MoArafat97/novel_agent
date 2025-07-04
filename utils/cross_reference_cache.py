"""
Smart Caching System for Cross-Reference Analysis

This module provides intelligent caching for entity classifications, relationship analyses,
and API responses with TTL (Time To Live) and cache invalidation capabilities.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    key: str
    data: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl
    
    def is_stale(self, max_age: float = 3600.0) -> bool:
        """Check if the cache entry is stale (older than max_age)."""
        return time.time() - self.timestamp > max_age
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class CrossReferenceCacheManager:
    """
    High-performance caching system for cross-reference analysis operations.
    
    Features:
    - TTL-based expiration
    - LRU eviction policy
    - Persistent storage
    - Cache statistics
    - Intelligent invalidation
    """
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 max_memory_entries: int = 1000,
                 default_ttl: float = 3600.0,  # 1 hour
                 enable_persistence: bool = True):
        """Initialize the cache manager."""
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        
        # In-memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0
        }
        
        # Create cache directory
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized CrossReferenceCacheManager with {max_memory_entries} max entries")
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        # Create a deterministic string from all arguments
        key_data = {
            'prefix': prefix,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return f"{prefix}:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            if entry.is_expired():
                # Remove expired entry
                del self.memory_cache[key]
                self.stats['misses'] += 1
                return None
            
            # Update access statistics
            entry.touch()
            self.stats['hits'] += 1
            return entry.data
        
        # Check persistent cache if enabled
        if self.enable_persistence:
            persistent_data = self._load_from_disk(key)
            if persistent_data:
                # Add to memory cache for faster future access
                self._add_to_memory_cache(key, persistent_data, self.default_ttl)
                self.stats['hits'] += 1
                return persistent_data
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        # Add to memory cache
        self._add_to_memory_cache(key, value, ttl)
        
        # Save to persistent cache if enabled
        if self.enable_persistence:
            self._save_to_disk(key, value, ttl)
    
    def _add_to_memory_cache(self, key: str, value: Any, ttl: float) -> None:
        """Add entry to memory cache with LRU eviction."""
        # Check if we need to evict entries
        if len(self.memory_cache) >= self.max_memory_entries:
            self._evict_lru_entries()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=value,
            timestamp=time.time(),
            ttl=ttl
        )
        
        self.memory_cache[key] = entry
    
    def _evict_lru_entries(self, target_size: Optional[int] = None) -> None:
        """Evict least recently used entries."""
        if target_size is None:
            target_size = int(self.max_memory_entries * 0.8)  # Evict 20%
        
        # Sort by last access time (oldest first)
        entries_by_access = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_access or x[1].timestamp
        )
        
        # Remove oldest entries
        entries_to_remove = len(self.memory_cache) - target_size
        for i in range(min(entries_to_remove, len(entries_by_access))):
            key, _ = entries_by_access[i]
            del self.memory_cache[key]
            self.stats['evictions'] += 1
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load cache entry from disk."""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if expired
            if time.time() - cache_data['timestamp'] > cache_data['ttl']:
                # Remove expired file
                cache_file.unlink(missing_ok=True)
                return None
            
            return cache_data['data']
            
        except Exception as e:
            logger.warning(f"Failed to load cache entry {key} from disk: {e}")
            return None
    
    def _save_to_disk(self, key: str, value: Any, ttl: float) -> None:
        """Save cache entry to disk."""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            cache_data = {
                'key': key,
                'data': value,
                'timestamp': time.time(),
                'ttl': ttl
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, default=str, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache entry {key} to disk: {e}")
    
    def invalidate(self, pattern: str = None, prefix: str = None) -> int:
        """Invalidate cache entries matching pattern or prefix."""
        invalidated_count = 0
        
        # Invalidate memory cache
        keys_to_remove = []
        for key in self.memory_cache:
            if self._should_invalidate(key, pattern, prefix):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.memory_cache[key]
            invalidated_count += 1
        
        # Invalidate persistent cache if enabled
        if self.enable_persistence:
            try:
                for cache_file in self.cache_dir.glob("*.json"):
                    key = cache_file.stem
                    if self._should_invalidate(key, pattern, prefix):
                        cache_file.unlink(missing_ok=True)
                        invalidated_count += 1
            except Exception as e:
                logger.warning(f"Failed to invalidate persistent cache: {e}")
        
        self.stats['invalidations'] += invalidated_count
        logger.info(f"Invalidated {invalidated_count} cache entries")
        return invalidated_count
    
    def _should_invalidate(self, key: str, pattern: str = None, prefix: str = None) -> bool:
        """Check if a key should be invalidated."""
        if pattern and pattern in key:
            return True
        if prefix and key.startswith(prefix):
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_entries': len(self.memory_cache),
            'max_memory_entries': self.max_memory_entries
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear persistent cache
        if self.enable_persistence:
            try:
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clear persistent cache: {e}")
        
        # Reset statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0
        }
        
        logger.info("Cleared all cache entries")


# Global cache instance
_cache_manager = None

def get_cache_manager() -> CrossReferenceCacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CrossReferenceCacheManager()
    return _cache_manager
