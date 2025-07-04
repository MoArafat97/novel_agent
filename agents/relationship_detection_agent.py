"""
Relationship Detection Agent

This specialized agent focuses solely on detecting and analyzing relationships
between entities using co-occurrence analysis, semantic search, and contextual analysis.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import openai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class RelationshipDetectionAgent:
    """
    Specialized agent for detecting relationships between worldbuilding entities.
    """
    
    def __init__(self, world_state=None, semantic_search=None):
        """Initialize the Relationship Detection Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        self.world_state = world_state
        self.semantic_search = semantic_search
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Relationship detection will use basic analysis only.")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client."""
        try:
            self.openrouter_client = openai.OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized for relationship detection agent")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None
    
    def is_available(self) -> bool:
        """Check if the agent is available."""
        return self.openrouter_client is not None and self.world_state is not None
    
    def detect_relationships(self, 
                           source_entity: Dict[str, Any],
                           detected_entities: List[Dict[str, Any]],
                           novel_id: str) -> List[Dict[str, Any]]:
        """
        Detect relationships between source entity and detected entities.
        
        Args:
            source_entity: The main entity being analyzed
            detected_entities: List of entities found in the content
            novel_id: Novel ID for context
            
        Returns:
            List of potential relationships with evidence and confidence
        """
        relationships = []
        
        if not self.is_available():
            return self._fallback_relationship_detection(source_entity, detected_entities)
        
        try:
            # Step 1: Co-occurrence analysis
            cooccurrence_relationships = self._analyze_cooccurrence(
                source_entity, detected_entities
            )
            
            # Step 2: Semantic similarity analysis
            semantic_relationships = self._analyze_semantic_similarity(
                source_entity, detected_entities, novel_id
            )
            
            # Step 3: Contextual relationship analysis
            contextual_relationships = self._analyze_contextual_relationships(
                source_entity, detected_entities
            )
            
            # Combine and deduplicate relationships
            all_relationships = (
                cooccurrence_relationships + 
                semantic_relationships + 
                contextual_relationships
            )
            
            relationships = self._deduplicate_relationships(all_relationships)
            
            logger.info(f"Detected {len(relationships)} potential relationships for {source_entity.get('name', 'Unknown')}")
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship detection failed: {e}")
            return self._fallback_relationship_detection(source_entity, detected_entities)
    
    def _analyze_cooccurrence(self, 
                             source_entity: Dict[str, Any], 
                             detected_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze co-occurrence patterns between entities."""
        relationships = []
        
        source_name = source_entity.get('name') or source_entity.get('title', '')
        
        for detected in detected_entities:
            detected_name = detected.get('entity_name', '')
            
            if not detected_name or detected_name.lower() == source_name.lower():
                continue
            
            # Calculate co-occurrence strength based on context proximity
            context = detected.get('context', '')
            proximity_score = self._calculate_proximity_score(source_name, detected_name, context)
            
            if proximity_score > 0.3:  # Threshold for meaningful co-occurrence
                relationship = {
                    'source_entity': source_name,
                    'source_type': source_entity.get('entity_type', 'unknown'),
                    'target_entity': detected_name,
                    'target_type': detected.get('entity_type', 'unknown'),
                    'relationship_type': 'co_occurrence',
                    'confidence': proximity_score,
                    'evidence': f"Co-occurred in context: {context[:100]}...",
                    'detection_method': 'cooccurrence_analysis'
                }
                relationships.append(relationship)
        
        return relationships
    
    def _calculate_proximity_score(self, entity1: str, entity2: str, context: str) -> float:
        """Calculate proximity score between two entities in context."""
        if not context or not entity1 or not entity2:
            return 0.0
        
        context_lower = context.lower()
        entity1_lower = entity1.lower()
        entity2_lower = entity2.lower()
        
        # Find positions of entities in context
        pos1 = context_lower.find(entity1_lower)
        pos2 = context_lower.find(entity2_lower)
        
        if pos1 == -1 or pos2 == -1:
            return 0.0
        
        # Calculate distance between entities
        distance = abs(pos1 - pos2)
        context_length = len(context)
        
        # Normalize distance (closer = higher score)
        if distance == 0:
            return 1.0
        
        # Score decreases with distance, but considers context length
        normalized_distance = distance / context_length
        proximity_score = max(0.0, 1.0 - (normalized_distance * 2))
        
        return proximity_score
    
    def _analyze_semantic_similarity(self, 
                                   source_entity: Dict[str, Any],
                                   detected_entities: List[Dict[str, Any]],
                                   novel_id: str) -> List[Dict[str, Any]]:
        """Analyze semantic similarity between entities."""
        relationships = []
        
        if not self.semantic_search:
            return relationships
        
        try:
            source_name = source_entity.get('name') or source_entity.get('title', '')
            source_content = self._get_entity_content(source_entity)
            
            for detected in detected_entities:
                detected_name = detected.get('entity_name', '')
                
                if not detected_name or detected_name.lower() == source_name.lower():
                    continue
                
                # Search for semantic similarity
                search_results = self.semantic_search.search(
                    query=detected_name,
                    entity_types=[source_entity.get('entity_type', 'characters')],
                    novel_id=novel_id,
                    n_results=5,
                    min_similarity=0.6
                )
                
                # Check if source entity appears in results
                for result in search_results:
                    if (result.get('entity_name', '').lower() == source_name.lower() and
                        result.get('similarity', 0) > 0.6):
                        
                        relationship = {
                            'source_entity': source_name,
                            'source_type': source_entity.get('entity_type', 'unknown'),
                            'target_entity': detected_name,
                            'target_type': detected.get('entity_type', 'unknown'),
                            'relationship_type': 'semantic_similarity',
                            'confidence': result.get('similarity', 0.6),
                            'evidence': f"Semantic similarity score: {result.get('similarity', 0.6):.2f}",
                            'detection_method': 'semantic_analysis'
                        }
                        relationships.append(relationship)
                        break
            
            return relationships
            
        except Exception as e:
            logger.error(f"Semantic similarity analysis failed: {e}")
            return relationships
    
    def _analyze_contextual_relationships(self, 
                                        source_entity: Dict[str, Any],
                                        detected_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze contextual clues to determine relationship types."""
        relationships = []
        
        if not self.openrouter_client:
            return relationships
        
        try:
            source_name = source_entity.get('name') or source_entity.get('title', '')
            
            for detected in detected_entities:
                detected_name = detected.get('entity_name', '')
                context = detected.get('context', '')
                
                if not detected_name or not context or detected_name.lower() == source_name.lower():
                    continue
                
                # Use LLM to analyze relationship type from context
                relationship_analysis = self._llm_analyze_relationship(
                    source_name, 
                    source_entity.get('entity_type', 'unknown'),
                    detected_name,
                    detected.get('entity_type', 'unknown'),
                    context
                )
                
                if relationship_analysis and relationship_analysis.get('confidence', 0) > 0.5:
                    relationships.append(relationship_analysis)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Contextual relationship analysis failed: {e}")
            return relationships
    
    def _llm_analyze_relationship(self, 
                                source_name: str,
                                source_type: str, 
                                target_name: str,
                                target_type: str,
                                context: str) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze relationship type from context."""
        try:
            system_prompt = """You are an expert at analyzing relationships between entities in fantasy/fiction worldbuilding.

Given two entities and their context, determine:
1. The type of relationship between them
2. The confidence level (0.0-1.0)
3. A brief explanation

Relationship types:
- spatial: one entity is located in/near another (lives in, located at, etc.)
- social: personal relationships (friends, enemies, allies, family, etc.)
- hierarchical: power/authority relationships (rules, serves, commands, etc.)
- causal: one entity affects another (created by, destroyed by, influenced by, etc.)
- temporal: time-based relationships (happened during, before, after, etc.)
- functional: role-based relationships (guards, protects, teaches, etc.)

Respond with JSON:
{
  "relationship_type": "spatial|social|hierarchical|causal|temporal|functional|none",
  "confidence": 0.0-1.0,
  "explanation": "brief explanation",
  "evidence": "relevant quote from context"
}"""
            
            user_prompt = f"""Analyze the relationship between these entities:

Source Entity: "{source_name}" (type: {source_type})
Target Entity: "{target_name}" (type: {target_type})

Context: "{context}"

What is the relationship between {source_name} and {target_name} based on this context?"""
            
            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            ai_response = response.choices[0].message.content
            result = self._parse_relationship_response(ai_response)
            
            if result and result.get('relationship_type') != 'none':
                return {
                    'source_entity': source_name,
                    'source_type': source_type,
                    'target_entity': target_name,
                    'target_type': target_type,
                    'relationship_type': result.get('relationship_type', 'unknown'),
                    'confidence': result.get('confidence', 0.5),
                    'evidence': result.get('evidence', context[:100]),
                    'explanation': result.get('explanation', ''),
                    'detection_method': 'llm_contextual_analysis'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"LLM relationship analysis failed: {e}")
            return None
    
    def _parse_relationship_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response for relationship analysis."""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            return None
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse relationship response: {response}")
            return None
    
    def _get_entity_content(self, entity: Dict[str, Any]) -> str:
        """Extract searchable content from entity."""
        content_parts = []
        
        # Add name/title
        if entity.get('name'):
            content_parts.append(entity['name'])
        if entity.get('title'):
            content_parts.append(entity['title'])
        
        # Add description fields
        for field in ['description', 'background', 'personality', 'geography', 'culture', 'details', 'significance']:
            if entity.get(field):
                content_parts.append(entity[field])
        
        # Add tags
        if entity.get('tags'):
            content_parts.extend(entity['tags'])
        
        return ' '.join(content_parts)
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships and keep the highest confidence ones."""
        seen = {}
        
        for rel in relationships:
            key = f"{rel['source_entity']}:{rel['target_entity']}:{rel['relationship_type']}"
            
            if key not in seen or rel['confidence'] > seen[key]['confidence']:
                seen[key] = rel
        
        return list(seen.values())
    
    def _fallback_relationship_detection(self, 
                                       source_entity: Dict[str, Any],
                                       detected_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback relationship detection using simple heuristics."""
        relationships = []
        
        source_name = source_entity.get('name') or source_entity.get('title', '')
        
        for detected in detected_entities:
            detected_name = detected.get('entity_name', '')
            
            if not detected_name or detected_name.lower() == source_name.lower():
                continue
            
            # Simple co-occurrence relationship
            relationship = {
                'source_entity': source_name,
                'source_type': source_entity.get('entity_type', 'unknown'),
                'target_entity': detected_name,
                'target_type': detected.get('entity_type', 'unknown'),
                'relationship_type': 'mentioned_together',
                'confidence': 0.6,
                'evidence': f"Mentioned together in content",
                'detection_method': 'fallback_cooccurrence'
            }
            relationships.append(relationship)
        
        return relationships
