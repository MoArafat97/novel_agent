"""
TinyDB Manager: Handles structured data operations for worldbuilding entities.

This module provides specialized operations for TinyDB, including validation,
schema enforcement, and entity-specific business logic.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from tinydb import TinyDB, Query
from tinydb.table import Table

logger = logging.getLogger(__name__)

class TinyDBManager:
    """
    Manages TinyDB operations with schema validation and business logic.
    """
    
    def __init__(self, db_path: str = './data/tinydb'):
        """
        Initialize TinyDB manager.
        
        Args:
            db_path: Path to TinyDB storage directory
        """
        self.db_path = db_path
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize database connections
        self.databases = {}
        self.entity_types = ['novels', 'characters', 'locations', 'lore']
        
        for entity_type in self.entity_types:
            db_file = os.path.join(self.db_path, f'{entity_type}.json')
            self.databases[entity_type] = TinyDB(db_file)
        
        logger.info(f"TinyDB Manager initialized at {self.db_path}")
    
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
        Create a new entity.
        
        Args:
            entity_type: Type of entity
            data: Entity data
            
        Returns:
            Entity ID if successful, None otherwise
        """
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
            
            logger.info(f"Created {entity_type} entity: {data['id']}")
            return data['id']
            
        except Exception as e:
            logger.error(f"Failed to create {entity_type} entity: {e}")
            return None
    
    def read(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Read an entity by ID.
        
        Args:
            entity_type: Type of entity
            entity_id: Entity ID
            
        Returns:
            Entity data or None if not found
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")
            
            db = self.databases[entity_type]
            EntityQuery = Query()
            results = db.search(EntityQuery.id == entity_id)
            
            return results[0] if results else None
            
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
    
    def close(self):
        """Close all database connections."""
        try:
            for db in self.databases.values():
                db.close()
            logger.info("TinyDB connections closed")
        except Exception as e:
            logger.error(f"Error closing TinyDB connections: {e}")
