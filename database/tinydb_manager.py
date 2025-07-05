"""
TinyDB Manager: Handles structured data operations for worldbuilding entities.

This module provides specialized operations for TinyDB, including validation,
schema enforcement, entity-specific business logic, and performance optimizations.
"""

import os
import uuid
import logging
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from collections import defaultdict, OrderedDict
from functools import lru_cache

from tinydb import TinyDB, Query
from tinydb.table import Table

logger = logging.getLogger(__name__)

class TinyDBManager:
    """
    Manages TinyDB operations with schema validation, business logic, and performance optimizations.
    """

    def __init__(self, db_path: str = './data/tinydb', enable_caching: bool = True,
                 cache_size: int = 500, enable_indexing: bool = True):
        """
        Initialize TinyDB manager with performance optimizations.

        Args:
            db_path: Path to TinyDB storage directory
            enable_caching: Enable query result caching
            cache_size: Maximum number of cached items
            enable_indexing: Enable in-memory indexing
        """
        self.db_path = db_path
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.enable_indexing = enable_indexing

        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        # Initialize database connections
        self.databases = {}
        self.entity_types = ['novels', 'characters', 'locations', 'lore']

        # Performance optimization structures
        self.query_cache = OrderedDict() if enable_caching else None
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}

        # In-memory indexes for fast lookups
        self.indexes = {
            'by_id': defaultdict(dict),
            'by_novel_id': defaultdict(lambda: defaultdict(list)),
            'by_name': defaultdict(dict),
            'by_tags': defaultdict(lambda: defaultdict(list))
        }

        # Performance metrics
        self.performance_metrics = {
            'query_times': [],
            'insert_times': [],
            'update_times': []
        }

        for entity_type in self.entity_types:
            db_file = os.path.join(self.db_path, f'{entity_type}.json')
            self.databases[entity_type] = TinyDB(db_file)

        # Build indexes if enabled
        if self.enable_indexing:
            self._build_indexes()

        logger.info(f"TinyDB Manager initialized at {self.db_path} with optimizations")

    def _build_indexes(self):
        """Build in-memory indexes for faster queries."""
        start_time = time.time()

        for entity_type in self.entity_types:
            db = self.databases[entity_type]
            all_entities = db.all()

            for entity in all_entities:
                self._update_indexes(entity_type, entity, remove=False)

        build_time = time.time() - start_time
        logger.info(f"TinyDB indexes built in {build_time:.3f}s")

    def _update_indexes(self, entity_type: str, entity: Dict[str, Any], remove: bool = False):
        """Update indexes when entity is added/updated/removed."""
        if not self.enable_indexing:
            return

        entity_id = entity.get('id')
        if not entity_id:
            return

        novel_id = entity.get('novel_id')
        name = entity.get('name') or entity.get('title', '')
        tags = entity.get('tags', [])

        if remove:
            # Remove from indexes
            self.indexes['by_id'][entity_type].pop(entity_id, None)

            if novel_id and entity_id in self.indexes['by_novel_id'][novel_id][entity_type]:
                self.indexes['by_novel_id'][novel_id][entity_type].remove(entity_id)

            if name and name.lower() in self.indexes['by_name'][entity_type]:
                if self.indexes['by_name'][entity_type][name.lower()] == entity_id:
                    del self.indexes['by_name'][entity_type][name.lower()]

            for tag in tags:
                if tag and entity_id in self.indexes['by_tags'][tag.lower()][entity_type]:
                    self.indexes['by_tags'][tag.lower()][entity_type].remove(entity_id)
        else:
            # Add to indexes
            self.indexes['by_id'][entity_type][entity_id] = entity

            if novel_id:
                if entity_id not in self.indexes['by_novel_id'][novel_id][entity_type]:
                    self.indexes['by_novel_id'][novel_id][entity_type].append(entity_id)

            if name:
                self.indexes['by_name'][entity_type][name.lower()] = entity_id

            for tag in tags:
                if tag and entity_id not in self.indexes['by_tags'][tag.lower()][entity_type]:
                    self.indexes['by_tags'][tag.lower()][entity_type].append(entity_id)

    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for operation."""
        key_parts = [operation]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            key_parts.append(f"{k}:{v}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def _cache_get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enable_caching or not self.query_cache:
            return None

        if cache_key in self.query_cache:
            # Move to end (LRU)
            value = self.query_cache.pop(cache_key)
            self.query_cache[cache_key] = value
            self.cache_stats['hits'] += 1
            return value

        self.cache_stats['misses'] += 1
        return None

    def _cache_set(self, cache_key: str, value: Any):
        """Set value in cache with LRU eviction."""
        if not self.enable_caching or not self.query_cache:
            return

        # Evict oldest if at capacity
        if len(self.query_cache) >= self.cache_size:
            self.query_cache.popitem(last=False)
            self.cache_stats['evictions'] += 1

        self.query_cache[cache_key] = value

    def _validate_entity_data(self, entity_type: str, data: Dict[str, Any]) -> bool:
        """
        Validate entity data based on type-specific schemas.
        
        Args:
            entity_type: Type of entity
            data: Entity data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Common required fields
            if 'id' not in data:
                logger.error(f"Missing required field 'id' for {entity_type}")
                return False
            
            # Type-specific validation
            if entity_type == 'novels':
                required_fields = ['title', 'description', 'genre']
                for field in required_fields:
                    if field not in data or not data[field]:
                        logger.error(f"Missing required field '{field}' for novel")
                        return False
            
            elif entity_type == 'characters':
                required_fields = ['name', 'description', 'novel_id']
                for field in required_fields:
                    if field not in data or not data[field]:
                        logger.error(f"Missing required field '{field}' for character")
                        return False
            
            elif entity_type == 'locations':
                required_fields = ['name', 'description', 'novel_id']
                for field in required_fields:
                    if field not in data or not data[field]:
                        logger.error(f"Missing required field '{field}' for location")
                        return False
            
            elif entity_type == 'lore':
                required_fields = ['title', 'description', 'novel_id']
                for field in required_fields:
                    if field not in data or not data[field]:
                        logger.error(f"Missing required field '{field}' for lore")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error for {entity_type}: {e}")
            return False
    
    def _add_metadata(self, data: Dict[str, Any], is_update: bool = False) -> Dict[str, Any]:
        """
        Add metadata fields to entity data.
        
        Args:
            data: Entity data
            is_update: Whether this is an update operation
            
        Returns:
            Data with metadata added
        """
        current_time = datetime.now().isoformat()
        
        if not is_update:
            data['created_at'] = current_time
        
        data['updated_at'] = current_time
        
        # Add origin metadata
        if 'origin' not in data:
            data['origin'] = 'manual'  # Can be 'manual', 'ai_generated', 'imported', etc.
        
        # Add tags if not present
        if 'tags' not in data:
            data['tags'] = []
        
        return data
    
    def create(self, entity_type: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new entity with performance optimizations.

        Args:
            entity_type: Type of entity
            data: Entity data

        Returns:
            Entity ID if successful, None otherwise
        """
        start_time = time.time()

        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")

            # Generate ID if not provided
            if 'id' not in data:
                data['id'] = str(uuid.uuid4())

            # Validate data
            if not self._validate_entity_data(entity_type, data):
                return None

            # Add metadata
            data = self._add_metadata(data, is_update=False)

            # Insert into database
            db = self.databases[entity_type]
            db.insert(data)

            # Update indexes
            self._update_indexes(entity_type, data, remove=False)

            # Invalidate relevant cache entries
            if self.enable_caching:
                self._invalidate_cache_for_entity(entity_type, data)

            # Track performance
            insert_time = time.time() - start_time
            self.performance_metrics['insert_times'].append(insert_time)

            logger.info(f"Created {entity_type} entity: {data['id']} in {insert_time:.3f}s")
            return data['id']

        except Exception as e:
            logger.error(f"Failed to create {entity_type} entity: {e}")
            return None
    
    def read(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Read an entity by ID with caching optimization.

        Args:
            entity_type: Type of entity
            entity_id: Entity ID

        Returns:
            Entity data or None if not found
        """
        # Check cache first
        cache_key = self._get_cache_key("read", entity_type=entity_type, entity_id=entity_id)
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result

        start_time = time.time()

        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")

            # Use index if available
            if self.enable_indexing and entity_id in self.indexes['by_id'][entity_type]:
                result = self.indexes['by_id'][entity_type][entity_id]
            else:
                # Fallback to database query
                db = self.databases[entity_type]
                EntityQuery = Query()
                results = db.search(EntityQuery.id == entity_id)
                result = results[0] if results else None

            # Cache result
            self._cache_set(cache_key, result)

            # Track performance
            query_time = time.time() - start_time
            self.performance_metrics['query_times'].append(query_time)

            return result

        except Exception as e:
            logger.error(f"Failed to read {entity_type} entity {entity_id}: {e}")
            return None
    
    def read_all(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Read all entities of a specific type.
        
        Args:
            entity_type: Type of entity
            
        Returns:
            List of entity data
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")
            
            db = self.databases[entity_type]
            return db.all()
            
        except Exception as e:
            logger.error(f"Failed to read all {entity_type} entities: {e}")
            return []
    
    def update(self, entity_type: str, entity_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            data: Updated entity data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")
            
            # Ensure ID is set
            data['id'] = entity_id
            
            # Validate data
            if not self._validate_entity_data(entity_type, data):
                return False
            
            # Get existing data to preserve created_at
            existing = self.read(entity_type, entity_id)
            if existing and 'created_at' in existing:
                data['created_at'] = existing['created_at']
            
            # Add metadata
            data = self._add_metadata(data, is_update=True)
            
            # Update in database
            db = self.databases[entity_type]
            EntityQuery = Query()
            updated_count = len(db.upsert(data, EntityQuery.id == entity_id))
            
            if updated_count > 0:
                logger.info(f"Updated {entity_type} entity: {entity_id}")
                return True
            else:
                logger.warning(f"No entity found to update: {entity_type} {entity_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update {entity_type} entity {entity_id}: {e}")
            return False
    
    def delete(self, entity_type: str, entity_id: str) -> bool:
        """
        Delete an entity.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")
            
            db = self.databases[entity_type]
            EntityQuery = Query()
            removed_count = len(db.remove(EntityQuery.id == entity_id))
            
            if removed_count > 0:
                logger.info(f"Deleted {entity_type} entity: {entity_id}")
                return True
            else:
                logger.warning(f"No entity found to delete: {entity_type} {entity_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete {entity_type} entity {entity_id}: {e}")
            return False
    
    def search(self, entity_type: str, query_func) -> List[Dict[str, Any]]:
        """
        Search entities using a custom query function.
        
        Args:
            entity_type: Type of entity
            query_func: TinyDB query function
            
        Returns:
            List of matching entities
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")
            
            db = self.databases[entity_type]
            return db.search(query_func)
            
        except Exception as e:
            logger.error(f"Failed to search {entity_type} entities: {e}")
            return []
    
    def get_entities_by_novel(self, novel_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all entities associated with a specific novel.
        
        Args:
            novel_id: Novel ID
            
        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        try:
            result = {}
            EntityQuery = Query()

            for entity_type in ['characters', 'locations', 'lore']:
                entities = self.search(entity_type, EntityQuery.novel_id == novel_id)
                result[entity_type] = entities
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get entities for novel {novel_id}: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about stored entities.
        
        Returns:
            Dictionary with entity counts
        """
        try:
            stats = {}
            
            for entity_type in self.entity_types:
                count = len(self.read_all(entity_type))
                stats[f'{entity_type}_count'] = count
            
            stats['total_entities'] = sum(stats.values())
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    # ===== OPTIMIZED BATCH OPERATIONS =====

    def batch_create(self, entity_type: str, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Create multiple entities in a batch operation.

        Args:
            entity_type: Type of entities
            entities: List of entity data

        Returns:
            List of created entity IDs
        """
        start_time = time.time()
        created_ids = []

        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")

            db = self.databases[entity_type]

            # Prepare all entities
            prepared_entities = []
            for data in entities:
                if 'id' not in data:
                    data['id'] = str(uuid.uuid4())

                if self._validate_entity_data(entity_type, data):
                    data = self._add_metadata(data, is_update=False)
                    prepared_entities.append(data)
                    created_ids.append(data['id'])

            # Batch insert
            if prepared_entities:
                db.insert_multiple(prepared_entities)

                # Update indexes for all entities
                for entity in prepared_entities:
                    self._update_indexes(entity_type, entity, remove=False)

                # Invalidate cache
                if self.enable_caching:
                    self._invalidate_cache_pattern(entity_type)

            # Track performance
            batch_time = time.time() - start_time
            logger.info(f"Batch created {len(created_ids)} {entity_type} entities in {batch_time:.3f}s")

            return created_ids

        except Exception as e:
            logger.error(f"Failed to batch create {entity_type} entities: {e}")
            return []

    def get_entities_by_novel_optimized(self, novel_id: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """
        Get entities by novel ID using optimized indexing.

        Args:
            novel_id: Novel ID to filter by
            entity_type: Optional entity type filter

        Returns:
            List of entities
        """
        cache_key = self._get_cache_key("get_by_novel", novel_id=novel_id, entity_type=entity_type)

        # Check cache first
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result

        start_time = time.time()
        results = []

        # Use index if available
        if self.enable_indexing and novel_id in self.indexes['by_novel_id']:
            entity_types = [entity_type] if entity_type else self.entity_types

            for etype in entity_types:
                entity_ids = self.indexes['by_novel_id'][novel_id].get(etype, [])
                for entity_id in entity_ids:
                    entity = self.indexes['by_id'][etype].get(entity_id)
                    if entity:
                        results.append(entity)
        else:
            # Fallback to direct query
            entity_types = [entity_type] if entity_type else self.entity_types
            EntityQuery = Query()

            for etype in entity_types:
                db = self.databases[etype]
                entities = db.search(EntityQuery.novel_id == novel_id)
                results.extend(entities)

        # Track performance
        query_time = time.time() - start_time
        self.performance_metrics['query_times'].append(query_time)

        # Cache result
        self._cache_set(cache_key, results)

        return results

    def _invalidate_cache_for_entity(self, entity_type: str, entity_data: Dict[str, Any]):
        """Invalidate cache entries related to an entity."""
        if not self.enable_caching or not self.query_cache:
            return

        novel_id = entity_data.get('novel_id')
        patterns_to_invalidate = [
            entity_type,
            f"novel_{novel_id}" if novel_id else None,
            "get_by_novel",
            "search"
        ]

        for pattern in patterns_to_invalidate:
            if pattern:
                self._invalidate_cache_pattern(pattern)

    def _invalidate_cache_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        if not self.enable_caching or not self.query_cache:
            return

        keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.query_cache[key]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = 0.0
        if self.cache_stats['hits'] + self.cache_stats['misses'] > 0:
            cache_hit_rate = self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses'])

        return {
            'cache_stats': self.cache_stats,
            'cache_hit_rate': cache_hit_rate,
            'query_times': self.performance_metrics['query_times'][-100:],  # Last 100
            'insert_times': self.performance_metrics['insert_times'][-100:],
            'update_times': self.performance_metrics['update_times'][-100:],
            'index_sizes': {
                'by_id': sum(len(entities) for entities in self.indexes['by_id'].values()),
                'by_novel_id': len(self.indexes['by_novel_id']),
                'by_name': sum(len(names) for names in self.indexes['by_name'].values()),
                'by_tags': len(self.indexes['by_tags'])
            }
        }

    def close(self):
        """Close all database connections and cleanup resources."""
        try:
            for db in self.databases.values():
                db.close()

            # Clear caches and indexes to free memory
            if self.query_cache:
                self.query_cache.clear()

            for index_type in self.indexes:
                self.indexes[index_type].clear()

            logger.info("TinyDB connections closed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing TinyDB connections: {e}")
