"""
Simple Entity Detector for Cross-Reference System

Focuses solely on detecting new entities (Characters, Locations, Lore) 
that should be added to the database. No relationship analysis.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from utils.entity_type_classifier import EntityTypeClassifier
from database.world_state import WorldState

logger = logging.getLogger(__name__)

class SimpleEntityDetector:
    """Simplified entity detector focused only on new entity detection."""
    
    def __init__(self, world_state: WorldState = None):
        self.world_state = world_state
        self.entity_classifier = EntityTypeClassifier()
        
    def detect_new_entities(self, content: str, novel_id: str, source_entity_type: str = None, source_entity_id: str = None) -> List[Dict[str, Any]]:
        """
        Detect new entities mentioned in content that don't exist in the database.
        
        Args:
            content: Text content to analyze
            novel_id: ID of the novel for context
            source_entity_type: Type of entity being updated (optional)
            source_entity_id: ID of entity being updated (optional)
            
        Returns:
            List of detected new entities with metadata
        """
        if not content or len(content.strip()) < 10:
            return []
            
        try:
            # Step 1: Extract potential entity names from content
            potential_entities = self._extract_potential_entities(content)
            
            if not potential_entities:
                return []
            
            # Step 2: Get existing entities for this novel
            existing_entities = self._get_existing_entities(novel_id)
            
            # Step 3: Filter out entities that already exist and exclude current entity
            new_entities = self._filter_new_entities(potential_entities, existing_entities, source_entity_type, source_entity_id)
            
            if not new_entities:
                return []
            
            # Step 4: Classify entity types using AI
            classified_entities = self._classify_entities(new_entities)
            
            # Step 5: Format results for user interface
            return self._format_detection_results(classified_entities, novel_id, source_entity_type, source_entity_id)
            
        except Exception as e:
            logger.error(f"Entity detection failed: {e}")
            return []
    
    def _extract_potential_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract potential entity names using pattern matching."""
        potential_entities = []
        
        # Use regex to find proper nouns (capitalized words/phrases)
        # Limit to max 3 words to avoid matching full sentences
        pattern = r'\b[A-Z][a-zA-Z]{1,}(?:\s+[A-Z][a-zA-Z]{1,}){0,2}\b'
        
        # Common words to exclude
        exclude_words = {
            'The', 'And', 'Or', 'But', 'With', 'From', 'To', 'In', 'On', 'At', 'By', 'For',
            'Of', 'As', 'Is', 'Was', 'Are', 'Were', 'Be', 'Been', 'Being', 'Have', 'Has', 'Had',
            'Do', 'Does', 'Did', 'Will', 'Would', 'Could', 'Should', 'May', 'Might', 'Must',
            'This', 'That', 'These', 'Those', 'Here', 'There', 'Where', 'When', 'Why', 'How',
            'What', 'Who', 'Which', 'Whose', 'Whom', 'All', 'Any', 'Some', 'Many', 'Much',
            'Character', 'Protagonist', 'Story', 'Main', 'During', 'Their', 'Adventures',
            'Together', 'Later', 'Both', 'Under', 'Wise', 'Ancient', 'Legendary',
            # Pronouns and common false positives
            'He', 'She', 'It', 'They', 'We', 'You', 'I', 'Me', 'Him', 'Her', 'Us', 'Them',
            'His', 'Hers', 'Its', 'Our', 'Your', 'My', 'Mine', 'Yours', 'Theirs'
        }
        
        for match in re.finditer(pattern, content):
            name = match.group().strip()
            
            # Skip if too short or in exclude list
            if len(name) < 2 or name in exclude_words:
                continue
                
            # Skip if contains excluded words
            if any(word in exclude_words for word in name.split()):
                continue
            
            # Get context around the entity (50 chars before and after)
            start_pos = match.start()
            end_pos = match.end()
            context_start = max(0, start_pos - 50)
            context_end = min(len(content), end_pos + 50)
            context = content[context_start:context_end].strip()
            
            potential_entities.append({
                'name': name,
                'context': context,
                'start_pos': start_pos,
                'end_pos': end_pos
            })
        
        return potential_entities
    
    def _get_existing_entities(self, novel_id: str) -> Dict[str, List[str]]:
        """Get all existing entity names for this novel."""
        existing = {'characters': [], 'locations': [], 'lore': []}
        
        if not self.world_state:
            return existing
            
        try:
            novel_entities = self.world_state.get_entities_by_novel(novel_id)
            
            for entity_type, entities in novel_entities.items():
                for entity in entities:
                    name = entity.get('name') or entity.get('title', '')
                    if name:
                        existing[entity_type].append(name.lower())
                        
        except Exception as e:
            logger.error(f"Failed to get existing entities: {e}")
            
        return existing
    
    def _filter_new_entities(self, potential_entities: List[Dict[str, Any]], existing_entities: Dict[str, List[str]],
                           source_entity_type: str = None, source_entity_id: str = None) -> List[Dict[str, Any]]:
        """Filter out entities that already exist in the database and exclude the current entity being analyzed."""
        new_entities = []

        # Create a set of all existing entity names (lowercase for comparison)
        all_existing = set()
        for entity_list in existing_entities.values():
            all_existing.update(entity_list)

        # Get the current entity name to exclude it from suggestions
        current_entity_name = None
        if source_entity_type and source_entity_id and self.world_state:
            try:
                current_entity_data = self.world_state.get(source_entity_type, source_entity_id)
                if current_entity_data:
                    current_entity_name = (current_entity_data.get('name') or
                                         current_entity_data.get('title', '')).lower()
                    logger.info(f"Excluding current entity '{current_entity_name}' from cross-reference suggestions")
            except Exception as e:
                logger.warning(f"Could not get current entity name for exclusion: {e}")

        for entity in potential_entities:
            name_lower = entity['name'].lower()

            # Skip if entity already exists in database
            if name_lower in all_existing:
                logger.debug(f"Skipping existing entity: {entity['name']}")
                continue

            # Skip if this is the current entity being analyzed
            if current_entity_name and name_lower == current_entity_name:
                logger.info(f"Skipping current entity from suggestions: {entity['name']}")
                continue

            # Skip duplicates within this detection
            if any(e['name'].lower() == name_lower for e in new_entities):
                logger.debug(f"Skipping duplicate entity: {entity['name']}")
                continue

            new_entities.append(entity)

        logger.info(f"Filtered {len(potential_entities)} potential entities down to {len(new_entities)} new entities")
        return new_entities
    
    def _classify_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify entity types using AI classifier."""
        classified = []
        
        for entity in entities:
            try:
                # Use the entity classifier to determine type
                classification = self.entity_classifier.classify_entity(
                    entity['name'], 
                    entity['context']
                )
                
                # Only include entities with valid classifications and reasonable confidence
                if (classification.get('entity_type') in ['characters', 'locations', 'lore'] and 
                    classification.get('confidence', 0) >= 0.3):
                    
                    entity['entity_type'] = classification['entity_type']
                    entity['confidence'] = classification['confidence']
                    entity['reasoning'] = classification.get('reasoning', [])
                    classified.append(entity)
                    
            except Exception as e:
                logger.error(f"Failed to classify entity {entity['name']}: {e}")
                continue
        
        return classified
    
    def _format_detection_results(self, entities: List[Dict[str, Any]], novel_id: str,
                                source_entity_type: str = None, source_entity_id: str = None) -> List[Dict[str, Any]]:
        """Format detection results for the user interface."""
        results = []

        for entity in entities:
            # Create suggested fields based on entity type
            suggested_fields = self._generate_suggested_fields(entity['name'], entity['entity_type'])

            # Create a clean, user-friendly result
            result = {
                'name': entity['name'],
                'type': entity['entity_type'],
                'confidence': entity['confidence'],
                'confidence_label': self._get_confidence_label(entity['confidence']),
                'context': entity['context'],
                'evidence': f"Mentioned in content: \"{entity['context']}\"",
                'novel_id': novel_id,
                'suggested_fields': suggested_fields,  # Add this for entity creation
                'source_entity_type': source_entity_type,
                'source_entity_id': source_entity_id
            }
            results.append(result)
        
        return results
    
    def _get_confidence_label(self, confidence: float) -> str:
        """Convert confidence score to user-friendly label."""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        else:
            return "Low"
    
    def _generate_suggested_fields(self, name: str, entity_type: str) -> Dict[str, Any]:
        """Generate suggested database fields for the new entity."""
        base_fields = {
            'name': name,
            'description': f"Auto-detected {entity_type[:-1]} from content analysis",
            'tags': [],
            'created_by': 'cross_reference_detection'
        }

        if entity_type == 'characters':
            base_fields.update({
                'personality': '',
                'backstory': '',
                'background': '',
                'occupation': '',
                'appearance': '',
                'relationships': []
            })
        elif entity_type == 'locations':
            base_fields.update({
                'geography': '',
                'climate': '',
                'culture': '',
                'history': '',
                'notable_features': '',
                'inhabitants': []
            })
        elif entity_type == 'lore':
            base_fields.update({
                'title': name,
                'details': '',
                'significance': '',
                'related_events': '',
                'category': 'general'
            })

        return base_fields
