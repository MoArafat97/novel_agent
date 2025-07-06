"""
Simplified Cross-Reference Service

Focuses solely on detecting new entities that should be added to the database.
No relationship analysis or complex workflows.
"""

import logging
from typing import Dict, Any, List, Optional
from utils.simple_entity_detector import SimpleEntityDetector
from database.world_state import WorldState

logger = logging.getLogger(__name__)

class SimpleCrossReferenceService:
    """Simplified service for detecting new entities in content."""
    
    def __init__(self, world_state: WorldState = None):
        self.world_state = world_state
        self.entity_detector = SimpleEntityDetector(world_state)
        
    def analyze_content_for_new_entities(self, entity_type: str, entity_id: str, 
                                       entity_data: Dict[str, Any], novel_id: str) -> Dict[str, Any]:
        """
        Analyze entity content to find new entities that should be added to the database.
        
        Args:
            entity_type: Type of entity being analyzed (characters, locations, lore)
            entity_id: ID of the entity being analyzed
            entity_data: Data of the entity being analyzed
            novel_id: ID of the novel
            
        Returns:
            Dictionary with detected new entities and metadata
        """
        try:
            entity_name = entity_data.get('name') or entity_data.get('title', 'Unknown')
            logger.info(f"Analyzing {entity_type} '{entity_name}' (ID: {entity_id}) for new entities in novel {novel_id}")

            # Extract all text content from the entity
            content = self._extract_entity_content(entity_data, entity_type)

            if not content or len(content.strip()) < 10:
                logger.warning(f"No content to analyze for {entity_type} '{entity_name}'")
                return self._create_empty_result(entity_type, entity_id, novel_id, "No content to analyze")

            logger.info(f"Extracted {len(content)} characters of content for analysis")

            # Detect new entities (excluding the current entity)
            detected_entities = self.entity_detector.detect_new_entities(
                content, novel_id, entity_type, entity_id
            )

            logger.info(f"Detected {len(detected_entities)} new entities for {entity_type} '{entity_name}'")
            
            # Create result
            entity_name = entity_data.get('name') or entity_data.get('title', 'Unknown')
            
            return {
                'success': True,
                'source_entity': {
                    'type': entity_type,
                    'id': entity_id,
                    'name': entity_name
                },
                'novel_id': novel_id,
                'detected_entities': detected_entities,
                'total_detected': len(detected_entities),
                'analysis_type': 'new_entity_detection',
                'message': f"Found {len(detected_entities)} potential new entities" if detected_entities else "No new entities detected"
            }
            
        except Exception as e:
            logger.error(f"Cross-reference analysis failed: {e}")
            return self._create_error_result(entity_type, entity_id, novel_id, str(e))
    
    def create_selected_entities(self, selected_entities: List[Dict[str, Any]], novel_id: str) -> Dict[str, Any]:
        """
        Create the entities that the user selected from the detection results.
        
        Args:
            selected_entities: List of entities the user chose to create
            novel_id: ID of the novel
            
        Returns:
            Dictionary with creation results
        """
        if not self.world_state:
            return {'success': False, 'error': 'Database not available'}
            
        created_entities = []
        failed_entities = []
        
        for entity in selected_entities:
            try:
                # Generate a unique ID for the new entity
                entity_id = self._generate_entity_id(entity['name'], entity['type'])
                
                # Prepare entity data
                entity_data = entity['suggested_fields'].copy()
                entity_data.update({
                    'id': entity_id,
                    'novel_id': novel_id,
                    'created_by': 'cross_reference_detection',
                    'detection_confidence': entity.get('confidence', 0.5)
                })

                logger.info(f"Creating {entity['type']} '{entity['name']}' with data: {list(entity_data.keys())}")

                # Create the entity in the database
                success = self.world_state.add_or_update(entity['type'], entity_id, entity_data)

                if success:
                    # Verify the entity was actually created
                    verification = self.world_state.get(entity['type'], entity_id)
                    if verification:
                        logger.info(f"Successfully created and verified {entity['type']} '{entity['name']}' with ID {entity_id}")
                        created_entities.append({
                            'name': entity['name'],
                            'type': entity['type'],
                            'id': entity_id
                        })

                        # Invalidate cache for this novel to ensure new entities appear
                        self._invalidate_novel_cache(novel_id)
                    else:
                        logger.error(f"Entity {entity['type']} '{entity['name']}' was reported as created but verification failed")
                        failed_entities.append({
                            'name': entity['name'],
                            'type': entity['type'],
                            'error': 'Entity creation verification failed'
                        })
                else:
                    logger.error(f"Failed to create {entity['type']} '{entity['name']}' in database")
                    failed_entities.append({
                        'name': entity['name'],
                        'type': entity['type'],
                        'error': 'Database creation failed'
                    })
                    
            except Exception as e:
                logger.error(f"Failed to create entity {entity['name']}: {e}")
                failed_entities.append({
                    'name': entity['name'],
                    'type': entity['type'],
                    'error': str(e)
                })
        
        result = {
            'success': len(failed_entities) == 0,
            'created_entities': created_entities,
            'failed_entities': failed_entities,
            'total_created': len(created_entities),
            'total_failed': len(failed_entities)
        }

        logger.info(f"Entity creation completed: {len(created_entities)} created, {len(failed_entities)} failed")
        return result
    
    def _extract_entity_content(self, entity_data: Dict[str, Any], entity_type: str) -> str:
        """Extract all relevant text content from an entity."""
        text_parts = []
        
        def add_content(content):
            if isinstance(content, str) and content.strip():
                text_parts.append(content.strip())
            elif isinstance(content, list):
                text_parts.extend([str(item) for item in content if str(item).strip()])
        
        # Add name/title
        if entity_data.get('name'):
            add_content(entity_data['name'])
        if entity_data.get('title'):
            add_content(entity_data['title'])
        
        # Add description
        if entity_data.get('description'):
            add_content(entity_data['description'])
        
        # Add type-specific fields
        if entity_type == 'characters':
            for field in ['personality', 'backstory', 'background', 'occupation']:
                if entity_data.get(field):
                    add_content(entity_data[field])
        elif entity_type == 'locations':
            for field in ['geography', 'climate', 'culture', 'history', 'notable_features']:
                if entity_data.get(field):
                    add_content(entity_data[field])
        elif entity_type == 'lore':
            for field in ['details', 'significance', 'related_events']:
                if entity_data.get(field):
                    add_content(entity_data[field])
        
        # Add tags
        if entity_data.get('tags'):
            add_content(entity_data['tags'])
        
        return ' '.join(text_parts)
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a unique ID for a new entity."""
        import uuid
        import re
        
        # Create a base ID from the name
        base_id = re.sub(r'[^a-zA-Z0-9]', '-', name.lower())
        base_id = re.sub(r'-+', '-', base_id).strip('-')
        
        # Add a short UUID to ensure uniqueness
        short_uuid = str(uuid.uuid4())[:8]
        
        return f"{base_id}-{short_uuid}"
    
    def _create_empty_result(self, entity_type: str, entity_id: str, novel_id: str, message: str) -> Dict[str, Any]:
        """Create an empty result when no entities are detected."""
        return {
            'success': True,
            'source_entity': {
                'type': entity_type,
                'id': entity_id
            },
            'novel_id': novel_id,
            'detected_entities': [],
            'total_detected': 0,
            'analysis_type': 'new_entity_detection',
            'message': message
        }
    
    def _create_error_result(self, entity_type: str, entity_id: str, novel_id: str, error: str) -> Dict[str, Any]:
        """Create an error result when analysis fails."""
        return {
            'success': False,
            'source_entity': {
                'type': entity_type,
                'id': entity_id
            },
            'novel_id': novel_id,
            'detected_entities': [],
            'total_detected': 0,
            'analysis_type': 'new_entity_detection',
            'error': error,
            'message': f"Analysis failed: {error}"
        }

    def _invalidate_novel_cache(self, novel_id: str) -> None:
        """Invalidate cache entries related to a specific novel."""
        try:
            # Import cache manager here to avoid circular imports
            from utils.cross_reference_cache import get_cache_manager

            cache_manager = get_cache_manager()

            # Invalidate cache entries for this novel
            invalidated = cache_manager.invalidate(pattern=novel_id)
            logger.info(f"Invalidated {invalidated} cache entries for novel {novel_id}")

            # Also invalidate any world state cache if it exists
            if hasattr(self.world_state, '_cache_invalidate'):
                self.world_state._cache_invalidate(f"novel_{novel_id}")

        except Exception as e:
            logger.warning(f"Failed to invalidate cache for novel {novel_id}: {e}")
