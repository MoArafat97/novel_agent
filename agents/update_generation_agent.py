"""
Update Generation Agent

This specialized agent focuses on generating actionable database updates
and suggestions based on verified cross-reference analysis results.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import openai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class UpdateGenerationAgent:
    """
    Specialized agent for generating database updates and suggestions.
    """
    
    def __init__(self, world_state=None):
        """Initialize the Update Generation Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        self.world_state = world_state
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Update generation will use basic templates only.")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client."""
        try:
            self.openrouter_client = openai.OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized for update generation agent")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None
    
    def is_available(self) -> bool:
        """Check if the agent is available."""
        return self.world_state is not None
    
    def generate_updates(self, 
                        verified_relationships: List[Dict[str, Any]],
                        verified_entities: List[Dict[str, Any]],
                        source_entity: Dict[str, Any],
                        novel_id: str) -> List[Dict[str, Any]]:
        """
        Generate actionable database updates based on verified relationships.
        
        Args:
            verified_relationships: List of verified relationships
            verified_entities: List of verified entities
            source_entity: The source entity being analyzed
            novel_id: Novel ID for context
            
        Returns:
            List of suggested database updates
        """
        updates = []
        
        try:
            logger.info(f"Generating updates for {len(verified_relationships)} relationships")
            
            # Generate relationship-based updates
            relationship_updates = self._generate_relationship_updates(
                verified_relationships, source_entity, novel_id
            )
            updates.extend(relationship_updates)
            
            # Generate tag updates
            tag_updates = self._generate_tag_updates(
                verified_relationships, verified_entities, source_entity
            )
            updates.extend(tag_updates)
            
            # Generate cross-reference updates
            cross_ref_updates = self._generate_cross_reference_updates(
                verified_relationships, source_entity, novel_id
            )
            updates.extend(cross_ref_updates)
            
            # Remove duplicates and prioritize
            final_updates = self._prioritize_updates(updates)
            
            logger.info(f"Generated {len(final_updates)} update suggestions")
            return final_updates
            
        except Exception as e:
            logger.error(f"Update generation failed: {e}")
            return []
    
    def _generate_relationship_updates(self, 
                                     relationships: List[Dict[str, Any]],
                                     source_entity: Dict[str, Any],
                                     novel_id: str) -> List[Dict[str, Any]]:
        """Generate updates based on discovered relationships."""
        updates = []
        
        for relationship in relationships:
            target_name = relationship.get('target_entity', '')
            relationship_type = relationship.get('relationship_type', '')
            confidence = relationship.get('confidence', 0.0)
            
            # Find the target entity in the database
            target_entity = self._find_entity_by_name(target_name, novel_id)
            
            if not target_entity:
                continue
            
            # Generate specific updates based on relationship type
            if relationship_type == 'spatial':
                update = self._generate_spatial_update(
                    source_entity, target_entity, relationship
                )
            elif relationship_type == 'social':
                update = self._generate_social_update(
                    source_entity, target_entity, relationship
                )
            elif relationship_type == 'hierarchical':
                update = self._generate_hierarchical_update(
                    source_entity, target_entity, relationship
                )
            else:
                update = self._generate_generic_update(
                    source_entity, target_entity, relationship
                )
            
            if update:
                updates.append(update)
        
        return updates
    
    def _generate_spatial_update(self, 
                               source_entity: Dict[str, Any],
                               target_entity: Dict[str, Any],
                               relationship: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate spatial relationship update."""
        source_type = source_entity.get('entity_type', '')
        target_type = target_entity.get('entity_type', '')
        
        # Character lives in/visits location
        if source_type == 'characters' and target_type == 'locations':
            return {
                'target_entity_id': target_entity['id'],
                'target_entity_name': target_entity.get('name', ''),
                'target_entity_type': 'locations',
                'update_type': 'add_reference',
                'changes': {
                    'notable_residents': [source_entity.get('name', '')]
                },
                'confidence': relationship.get('confidence', 0.0),
                'evidence': relationship.get('evidence', ''),
                'relationship': relationship
            }
        
        # Location contains character
        elif source_type == 'locations' and target_type == 'characters':
            return {
                'target_entity_id': target_entity['id'],
                'target_entity_name': target_entity.get('name', ''),
                'target_entity_type': 'characters',
                'update_type': 'add_reference',
                'changes': {
                    'current_location': source_entity.get('name', '')
                },
                'confidence': relationship.get('confidence', 0.0),
                'evidence': relationship.get('evidence', ''),
                'relationship': relationship
            }
        
        return None
    
    def _generate_social_update(self, 
                              source_entity: Dict[str, Any],
                              target_entity: Dict[str, Any],
                              relationship: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate social relationship update."""
        if source_entity.get('entity_type') == 'characters' and target_entity.get('entity_type') == 'characters':
            return {
                'target_entity_id': target_entity['id'],
                'target_entity_name': target_entity.get('name', ''),
                'target_entity_type': 'characters',
                'update_type': 'add_reference',
                'changes': {
                    'relationships': [f"Connected to {source_entity.get('name', '')}"]
                },
                'confidence': relationship.get('confidence', 0.0),
                'evidence': relationship.get('evidence', ''),
                'relationship': relationship
            }
        
        return None
    
    def _generate_hierarchical_update(self, 
                                    source_entity: Dict[str, Any],
                                    target_entity: Dict[str, Any],
                                    relationship: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate hierarchical relationship update."""
        return {
            'target_entity_id': target_entity['id'],
            'target_entity_name': target_entity.get('name', ''),
            'target_entity_type': target_entity.get('entity_type', ''),
            'update_type': 'add_reference',
            'changes': {
                'hierarchy_notes': [f"Hierarchical relationship with {source_entity.get('name', '')}"]
            },
            'confidence': relationship.get('confidence', 0.0),
            'evidence': relationship.get('evidence', ''),
            'relationship': relationship
        }
    
    def _generate_generic_update(self, 
                               source_entity: Dict[str, Any],
                               target_entity: Dict[str, Any],
                               relationship: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate generic relationship update."""
        return {
            'target_entity_id': target_entity['id'],
            'target_entity_name': target_entity.get('name', ''),
            'target_entity_type': target_entity.get('entity_type', ''),
            'update_type': 'add_reference',
            'changes': {
                'cross_references': [f"Related to {source_entity.get('name', '')} ({relationship.get('relationship_type', 'unknown')})"]
            },
            'confidence': relationship.get('confidence', 0.0),
            'evidence': relationship.get('evidence', ''),
            'relationship': relationship
        }
    
    def _generate_tag_updates(self, 
                            relationships: List[Dict[str, Any]],
                            entities: List[Dict[str, Any]],
                            source_entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tag-based updates."""
        updates = []
        
        # Collect related entity names for tagging
        related_names = set()
        for rel in relationships:
            related_names.add(rel.get('target_entity', ''))
        
        for entity in entities:
            related_names.add(entity.get('entity_name', ''))
        
        # Generate tag update for source entity
        if related_names:
            new_tags = [name for name in related_names if name and len(name) > 2][:5]  # Limit to 5 tags
            
            if new_tags:
                updates.append({
                    'target_entity_id': source_entity['id'],
                    'target_entity_name': source_entity.get('name', ''),
                    'target_entity_type': source_entity.get('entity_type', ''),
                    'update_type': 'add_tags',
                    'changes': {
                        'tags': new_tags
                    },
                    'confidence': 0.7,
                    'evidence': f"Related to {len(new_tags)} other entities",
                    'relationship': None
                })
        
        return updates
    
    def _generate_cross_reference_updates(self, 
                                        relationships: List[Dict[str, Any]],
                                        source_entity: Dict[str, Any],
                                        novel_id: str) -> List[Dict[str, Any]]:
        """Generate cross-reference field updates."""
        updates = []
        
        # Group relationships by target entity
        target_groups = {}
        for rel in relationships:
            target_name = rel.get('target_entity', '')
            if target_name not in target_groups:
                target_groups[target_name] = []
            target_groups[target_name].append(rel)
        
        # Generate cross-reference updates for each target
        for target_name, target_relationships in target_groups.items():
            target_entity = self._find_entity_by_name(target_name, novel_id)
            
            if not target_entity:
                continue
            
            # Create cross-reference text
            ref_text = self._generate_cross_reference_text(
                source_entity, target_relationships
            )
            
            if ref_text:
                updates.append({
                    'target_entity_id': target_entity['id'],
                    'target_entity_name': target_entity.get('name', ''),
                    'target_entity_type': target_entity.get('entity_type', ''),
                    'update_type': 'add_cross_reference',
                    'changes': {
                        'cross_references': [ref_text]
                    },
                    'confidence': max([r.get('confidence', 0) for r in target_relationships]),
                    'evidence': f"Based on {len(target_relationships)} relationship(s)",
                    'relationship': target_relationships[0]  # Primary relationship
                })
        
        return updates
    
    def _generate_cross_reference_text(self, 
                                     source_entity: Dict[str, Any],
                                     relationships: List[Dict[str, Any]]) -> str:
        """Generate human-readable cross-reference text."""
        source_name = source_entity.get('name', '')
        
        if not relationships:
            return f"Related to {source_name}"
        
        # Use the highest confidence relationship for the description
        primary_rel = max(relationships, key=lambda r: r.get('confidence', 0))
        rel_type = primary_rel.get('relationship_type', 'related')
        
        # Generate descriptive text based on relationship type
        if rel_type == 'spatial':
            return f"Spatially connected to {source_name}"
        elif rel_type == 'social':
            return f"Has social connection with {source_name}"
        elif rel_type == 'hierarchical':
            return f"In hierarchical relationship with {source_name}"
        elif rel_type == 'causal':
            return f"Causally related to {source_name}"
        elif rel_type == 'temporal':
            return f"Temporally connected to {source_name}"
        elif rel_type == 'functional':
            return f"Functionally related to {source_name}"
        else:
            return f"Connected to {source_name}"
    
    def _find_entity_by_name(self, entity_name: str, novel_id: str) -> Optional[Dict[str, Any]]:
        """Find entity in database by name."""
        if not self.world_state:
            return None
        
        try:
            existing_entities = self.world_state.get_entities_by_novel(novel_id)
            
            for entity_type, entities in existing_entities.items():
                for entity in entities:
                    existing_name = entity.get('name') or entity.get('title', '')
                    if existing_name.lower() == entity_name.lower():
                        entity['entity_type'] = entity_type  # Add type for convenience
                        return entity
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find entity by name: {e}")
            return None
    
    def _prioritize_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize and deduplicate updates."""
        # Remove duplicates based on target entity and update type
        seen = {}
        for update in updates:
            key = f"{update['target_entity_id']}:{update['update_type']}"
            
            if key not in seen or update['confidence'] > seen[key]['confidence']:
                seen[key] = update
        
        # Sort by confidence and limit count
        prioritized = list(seen.values())
        prioritized.sort(key=lambda x: x['confidence'], reverse=True)
        
        return prioritized[:10]  # Limit to top 10 updates
    
    def generate_new_entity_suggestions(self, 
                                      verified_new_entities: List[Dict[str, Any]],
                                      novel_id: str) -> List[Dict[str, Any]]:
        """Generate suggestions for creating new entities."""
        suggestions = []
        
        for entity in verified_new_entities:
            suggestion = {
                'name': entity.get('name', ''),
                'type': entity.get('type', ''),
                'description': entity.get('description', ''),
                'confidence': entity.get('confidence', 0.0),
                'evidence': entity.get('evidence', ''),
                'suggested_fields': self._suggest_entity_fields(entity),
                'novel_id': novel_id
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_entity_fields(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest initial field values for new entity."""
        entity_type = entity.get('type', '')
        name = entity.get('name', '')
        description = entity.get('description', '')
        
        if entity_type == 'characters':
            return {
                'name': name,
                'description': description,
                'background': f"Background information about {name}",
                'personality': f"Personality traits of {name}",
                'tags': []
            }
        elif entity_type == 'locations':
            return {
                'name': name,
                'description': description,
                'geography': f"Geographical details of {name}",
                'culture': f"Cultural aspects of {name}",
                'tags': []
            }
        elif entity_type == 'lore':
            return {
                'title': name,
                'details': description,
                'significance': f"Significance of {name}",
                'related_events': f"Events related to {name}",
                'tags': []
            }
        
        return {}
