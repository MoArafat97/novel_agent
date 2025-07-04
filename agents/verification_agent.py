"""
Verification Agent

This specialized agent focuses on validating and filtering results from other agents,
ensuring high quality and logical consistency in cross-reference analysis.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Set
import openai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class VerificationAgent:
    """
    Specialized agent for verifying and validating cross-reference results.
    """
    
    def __init__(self, world_state=None):
        """Initialize the Verification Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        self.world_state = world_state
        
        # Quality thresholds
        self.min_confidence_threshold = 0.6
        self.min_relationship_confidence = 0.5
        self.max_entities_per_analysis = 20
        self.max_relationships_per_entity = 10
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Verification will use basic validation only.")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client."""
        try:
            self.openrouter_client = openai.OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized for verification agent")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None
    
    def is_available(self) -> bool:
        """Check if the agent is available."""
        return self.world_state is not None
    
    def verify_analysis_results(self, 
                               detected_entities: List[Dict[str, Any]],
                               relationships: List[Dict[str, Any]],
                               new_entities: List[Dict[str, Any]],
                               source_entity: Dict[str, Any],
                               novel_id: str) -> Dict[str, Any]:
        """
        Verify and validate all analysis results.
        
        Args:
            detected_entities: List of detected entities
            relationships: List of detected relationships
            new_entities: List of suggested new entities
            source_entity: The source entity being analyzed
            novel_id: Novel ID for context
            
        Returns:
            Verified and filtered results
        """
        try:
            logger.info(f"Starting verification of analysis results for {source_entity.get('name', 'Unknown')}")
            
            # Step 1: Validate detected entities
            verified_entities = self._verify_detected_entities(detected_entities, source_entity)
            
            # Step 2: Validate relationships
            verified_relationships = self._verify_relationships(relationships, source_entity, novel_id)
            
            # Step 3: Validate new entity suggestions
            verified_new_entities = self._verify_new_entities(new_entities, novel_id)
            
            # Step 4: Check logical consistency
            consistency_check = self._check_logical_consistency(
                verified_entities, verified_relationships, verified_new_entities
            )
            
            # Step 5: Apply quality filters
            final_results = self._apply_quality_filters(
                verified_entities, verified_relationships, verified_new_entities
            )
            
            verification_summary = {
                'original_entities': len(detected_entities),
                'verified_entities': len(final_results['entities']),
                'original_relationships': len(relationships),
                'verified_relationships': len(final_results['relationships']),
                'original_new_entities': len(new_entities),
                'verified_new_entities': len(final_results['new_entities']),
                'consistency_score': consistency_check.get('score', 0.0),
                'quality_issues': consistency_check.get('issues', [])
            }
            
            logger.info(f"Verification completed: {verification_summary}")
            
            return {
                'entities': final_results['entities'],
                'relationships': final_results['relationships'],
                'new_entities': final_results['new_entities'],
                'verification_summary': verification_summary,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                'entities': [],
                'relationships': [],
                'new_entities': [],
                'verification_summary': {'error': str(e)},
                'success': False
            }
    
    def _verify_detected_entities(self, 
                                 detected_entities: List[Dict[str, Any]],
                                 source_entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verify detected entities for quality and relevance."""
        verified = []
        source_name = source_entity.get('name', '').lower()
        
        for entity in detected_entities:
            entity_name = entity.get('entity_name', '').lower()
            confidence = entity.get('confidence', 0.0)
            recommendation = entity.get('recommendation', 'review')
            
            # Skip self-references
            if entity_name == source_name:
                continue
            
            # Apply confidence threshold
            if confidence < self.min_confidence_threshold:
                continue
            
            # Only accept entities with good recommendations
            if recommendation not in ['accept', 'review']:
                continue
            
            # Check for valid entity type
            entity_type = entity.get('entity_type')
            if entity_type not in ['characters', 'locations', 'lore']:
                continue
            
            # Add verification metadata
            entity['verified'] = True
            entity['verification_score'] = confidence
            verified.append(entity)
        
        # Limit number of entities
        verified = sorted(verified, key=lambda x: x.get('confidence', 0), reverse=True)
        return verified[:self.max_entities_per_analysis]
    
    def _verify_relationships(self, 
                            relationships: List[Dict[str, Any]],
                            source_entity: Dict[str, Any],
                            novel_id: str) -> List[Dict[str, Any]]:
        """Verify relationships for logical consistency and quality."""
        verified = []
        
        for relationship in relationships:
            confidence = relationship.get('confidence', 0.0)
            relationship_type = relationship.get('relationship_type', '')
            
            # Apply confidence threshold
            if confidence < self.min_relationship_confidence:
                continue
            
            # Validate relationship type
            valid_types = [
                'spatial', 'social', 'hierarchical', 'causal', 
                'temporal', 'functional', 'co_occurrence', 
                'semantic_similarity', 'mentioned_together'
            ]
            if relationship_type not in valid_types:
                continue
            
            # Check for valid entity references
            source_name = relationship.get('source_entity', '')
            target_name = relationship.get('target_entity', '')
            
            if not source_name or not target_name or source_name.lower() == target_name.lower():
                continue
            
            # Validate against existing entities if possible
            if self._validate_entity_existence(target_name, novel_id):
                relationship['target_exists'] = True
            else:
                relationship['target_exists'] = False
            
            # Add verification metadata
            relationship['verified'] = True
            relationship['verification_score'] = confidence
            verified.append(relationship)
        
        # Remove duplicates and limit count
        verified = self._deduplicate_relationships(verified)
        verified = sorted(verified, key=lambda x: x.get('confidence', 0), reverse=True)
        return verified[:self.max_relationships_per_entity]
    
    def _verify_new_entities(self, 
                           new_entities: List[Dict[str, Any]],
                           novel_id: str) -> List[Dict[str, Any]]:
        """Verify new entity suggestions for quality and uniqueness."""
        verified = []
        seen_names = set()
        
        for entity in new_entities:
            name = entity.get('name', '').strip()
            confidence = entity.get('confidence', 0.0)
            entity_type = entity.get('type', '')
            
            # Skip if already seen
            if name.lower() in seen_names:
                continue
            
            # Apply confidence threshold
            if confidence < self.min_confidence_threshold:
                continue
            
            # Validate entity type
            if entity_type not in ['characters', 'locations', 'lore']:
                continue
            
            # Check if entity already exists
            if self._validate_entity_existence(name, novel_id):
                continue  # Skip existing entities
            
            # Validate name quality
            if not self._validate_entity_name_quality(name):
                continue
            
            seen_names.add(name.lower())
            entity['verified'] = True
            entity['verification_score'] = confidence
            verified.append(entity)
        
        # Limit number of new entities
        verified = sorted(verified, key=lambda x: x.get('confidence', 0), reverse=True)
        return verified[:5]  # Conservative limit for new entities
    
    def _validate_entity_existence(self, entity_name: str, novel_id: str) -> bool:
        """Check if entity already exists in the database."""
        if not self.world_state:
            return False
        
        try:
            existing_entities = self.world_state.get_entities_by_novel(novel_id)
            
            for entity_type, entities in existing_entities.items():
                for entity in entities:
                    existing_name = entity.get('name') or entity.get('title', '')
                    if existing_name.lower() == entity_name.lower():
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check entity existence: {e}")
            return False
    
    def _validate_entity_name_quality(self, name: str) -> bool:
        """Validate the quality of an entity name."""
        if not name or len(name.strip()) < 2:
            return False
        
        # Check for common words that shouldn't be entities
        common_words = {
            'the', 'and', 'or', 'but', 'with', 'from', 'to', 'of', 'in', 'on',
            'at', 'by', 'for', 'as', 'is', 'was', 'are', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        if name.lower().strip() in common_words:
            return False
        
        # Check for temporal words
        temporal_words = {
            'annual', 'yearly', 'monthly', 'daily', 'weekly', 'century', 'decade',
            'year', 'month', 'day', 'week', 'season', 'winter', 'summer', 'spring',
            'autumn', 'fall', 'morning', 'evening', 'night'
        }
        
        if name.lower().strip() in temporal_words:
            return False
        
        # Check for simple adjectives
        simple_adjectives = {
            'cold', 'hot', 'warm', 'cool', 'big', 'small', 'large', 'tiny',
            'old', 'new', 'young', 'ancient', 'modern', 'good', 'bad', 'evil'
        }
        
        if name.lower().strip() in simple_adjectives:
            return False
        
        return True
    
    def _check_logical_consistency(self, 
                                 entities: List[Dict[str, Any]],
                                 relationships: List[Dict[str, Any]],
                                 new_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check logical consistency of the analysis results."""
        issues = []
        score = 1.0
        
        # Check for contradictory relationships
        relationship_pairs = {}
        for rel in relationships:
            key = f"{rel['source_entity']}:{rel['target_entity']}"
            if key in relationship_pairs:
                if relationship_pairs[key] != rel['relationship_type']:
                    issues.append(f"Contradictory relationships for {key}")
                    score -= 0.1
            else:
                relationship_pairs[key] = rel['relationship_type']
        
        # Check for impossible relationships
        for rel in relationships:
            source_type = rel.get('source_type', '')
            target_type = rel.get('target_type', '')
            rel_type = rel.get('relationship_type', '')
            
            # Example: locations can't have social relationships with characters
            if (source_type == 'locations' and target_type == 'characters' and 
                rel_type == 'social'):
                issues.append(f"Impossible relationship: {rel_type} between {source_type} and {target_type}")
                score -= 0.2
        
        # Check entity name quality
        all_entity_names = [e.get('entity_name', '') for e in entities]
        all_entity_names.extend([e.get('name', '') for e in new_entities])
        
        for name in all_entity_names:
            if not self._validate_entity_name_quality(name):
                issues.append(f"Low quality entity name: {name}")
                score -= 0.1
        
        return {
            'score': max(0.0, score),
            'issues': issues
        }
    
    def _apply_quality_filters(self, 
                             entities: List[Dict[str, Any]],
                             relationships: List[Dict[str, Any]],
                             new_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply final quality filters to results."""
        
        # Filter entities by verification score
        filtered_entities = [
            e for e in entities 
            if e.get('verification_score', 0) >= self.min_confidence_threshold
        ]
        
        # Filter relationships by verification score
        filtered_relationships = [
            r for r in relationships 
            if r.get('verification_score', 0) >= self.min_relationship_confidence
        ]
        
        # Filter new entities by verification score
        filtered_new_entities = [
            e for e in new_entities 
            if e.get('verification_score', 0) >= self.min_confidence_threshold
        ]
        
        return {
            'entities': filtered_entities,
            'relationships': filtered_relationships,
            'new_entities': filtered_new_entities
        }
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships."""
        seen = {}
        
        for rel in relationships:
            key = f"{rel['source_entity']}:{rel['target_entity']}:{rel['relationship_type']}"
            
            if key not in seen or rel['confidence'] > seen[key]['confidence']:
                seen[key] = rel
        
        return list(seen.values())
