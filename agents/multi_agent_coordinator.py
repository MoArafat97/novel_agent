"""
Multi-Agent Coordinator

This coordinator orchestrates the specialized agents to perform comprehensive
cross-reference analysis with proper error handling and quality control.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from utils.entity_type_classifier import EntityTypeClassifier
from .relationship_detection_agent import RelationshipDetectionAgent
from .verification_agent import VerificationAgent
from .update_generation_agent import UpdateGenerationAgent

logger = logging.getLogger(__name__)


class MultiAgentCoordinator:
    """
    Coordinates multiple specialized agents for cross-reference analysis.
    """
    
    def __init__(self, world_state=None, semantic_search=None):
        """Initialize the Multi-Agent Coordinator."""
        self.world_state = world_state
        self.semantic_search = semantic_search
        
        # Initialize specialized agents
        self.entity_classifier = EntityTypeClassifier()
        self.relationship_detector = RelationshipDetectionAgent(world_state, semantic_search)
        self.verification_agent = VerificationAgent(world_state)
        self.update_generator = UpdateGenerationAgent(world_state)
        
        logger.info("Multi-Agent Coordinator initialized with specialized agents")
    
    def is_available(self) -> bool:
        """Check if the coordinator and its agents are available."""
        return (
            self.world_state is not None and
            self.entity_classifier.is_available() and
            self.verification_agent.is_available()
        )
    
    def analyze_content(self, 
                       entity_type: str,
                       entity_id: str,
                       entity_data: Dict[str, Any],
                       novel_id: str) -> Dict[str, Any]:
        """
        Perform comprehensive cross-reference analysis using multiple specialized agents.
        
        Args:
            entity_type: Type of the source entity
            entity_id: ID of the source entity
            entity_data: Data of the source entity
            novel_id: Novel ID for context
            
        Returns:
            Comprehensive analysis results
        """
        try:
            analysis_start = datetime.now()
            entity_name = entity_data.get('name') or entity_data.get('title', 'Unknown')
            
            logger.info(f"Starting multi-agent analysis for {entity_type} '{entity_name}'")
            
            # Step 1: Extract content for analysis
            content_text = self._extract_entity_content(entity_data, entity_type)
            
            if not content_text or len(content_text.strip()) < 10:
                return self._create_empty_result(entity_type, entity_id, entity_name, novel_id, "Insufficient content for analysis")
            
            # Step 2: Entity Recognition and Classification
            logger.info("Step 1: Entity Recognition and Classification")
            detected_entities = self._detect_and_classify_entities(content_text, novel_id)
            
            if not detected_entities:
                return self._create_empty_result(entity_type, entity_id, entity_name, novel_id, "No entities detected")
            
            # Step 3: Relationship Detection
            logger.info("Step 2: Relationship Detection")
            relationships = self._detect_relationships(entity_data, detected_entities, novel_id)
            
            # Step 4: Generate New Entity Suggestions
            logger.info("Step 3: New Entity Generation")
            new_entities = self._generate_new_entity_suggestions(detected_entities, novel_id)
            
            # Step 5: Verification and Quality Control
            logger.info("Step 4: Verification and Quality Control")
            verification_results = self.verification_agent.verify_analysis_results(
                detected_entities, relationships, new_entities, entity_data, novel_id
            )
            
            if not verification_results.get('success'):
                return self._create_error_result(entity_type, entity_id, entity_name, novel_id, 
                                               verification_results.get('verification_summary', {}).get('error', 'Verification failed'))
            
            # Step 6: Update Generation
            logger.info("Step 5: Update Generation")
            suggested_updates = self.update_generator.generate_updates(
                verification_results['relationships'],
                verification_results['entities'],
                entity_data,
                novel_id
            )
            
            # Generate new entity creation suggestions
            new_entity_suggestions = self.update_generator.generate_new_entity_suggestions(
                verification_results['new_entities'],
                novel_id
            )
            
            # Step 7: Compile Final Results
            analysis_end = datetime.now()
            analysis_duration = (analysis_end - analysis_start).total_seconds()
            
            final_result = {
                'entity_type': entity_type,
                'entity_id': entity_id,
                'entity_name': entity_name,
                'novel_id': novel_id,
                'detected_entities': verification_results['entities'],
                'semantic_matches': [],  # Populated by semantic search if available
                'verified_relationships': verification_results['relationships'],
                'suggested_updates': suggested_updates,
                'new_entities': new_entity_suggestions,
                'analysis_timestamp': analysis_end.isoformat(),
                'analysis_duration': analysis_duration,
                'verification_summary': verification_results['verification_summary'],
                'agent_performance': self._get_agent_performance(),
                'success': True
            }
            
            logger.info(f"Multi-agent analysis completed in {analysis_duration:.2f}s. "
                       f"Found {len(verification_results['relationships'])} relationships, "
                       f"{len(suggested_updates)} updates, {len(new_entity_suggestions)} new entities")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Multi-agent analysis failed: {e}")
            return self._create_error_result(entity_type, entity_id, entity_name, novel_id, str(e))
    
    def _extract_entity_content(self, entity_data: Dict[str, Any], entity_type: str) -> str:
        """Extract searchable content from entity data."""
        text_parts = []
        
        def add_field_content(content):
            if isinstance(content, str) and content.strip():
                text_parts.append(content.strip())
            elif isinstance(content, list):
                text_parts.extend([str(item) for item in content if str(item).strip()])
        
        # Add name/title
        if entity_data.get('name'):
            add_field_content(entity_data['name'])
        if entity_data.get('title'):
            add_field_content(entity_data['title'])
        
        # Add type-specific fields (consistent with cross-reference agent)
        if entity_type == 'characters':
            for field in ['description', 'personality', 'backstory', 'background', 'occupation']:
                if entity_data.get(field):
                    add_field_content(entity_data[field])
        elif entity_type == 'locations':
            for field in ['description', 'geography', 'climate', 'culture', 'history', 'notable_features']:
                if entity_data.get(field):
                    add_field_content(entity_data[field])
        elif entity_type == 'lore':
            for field in ['description', 'details', 'significance', 'related_events']:
                if entity_data.get(field):
                    add_field_content(entity_data[field])
        
        # Add tags
        if entity_data.get('tags'):
            add_field_content(entity_data['tags'])
        
        return ' '.join(text_parts)
    
    def _detect_and_classify_entities(self, content_text: str, novel_id: str) -> List[Dict[str, Any]]:
        """Use entity classifier to detect and classify entities."""
        try:
            # Extract potential entity names using NLP patterns
            potential_entities = self._extract_potential_entity_names(content_text)
            
            if not potential_entities:
                return []
            
            # Use semantic classifier to validate and classify entities (OPTIMIZED: Batch Processing)
            detected = []
            logger.info(f"Classifying {len(potential_entities)} potential entities using batch semantic classifier")

            # Prepare entities for batch processing
            entities_for_batch = [(entity_info['name'], entity_info['context']) for entity_info in potential_entities]

            # Use batch classification for improved performance
            classifications = self.entity_classifier.classify_entities_batch_optimized(
                entities_for_batch,
                novel_id,
                batch_size=8,  # Optimize batch size for cost/speed balance
                max_workers=3   # Limit concurrent API calls
            )

            # Process batch results
            for entity_info, classification in zip(potential_entities, classifications):
                # Only accept entities with good classification
                if (classification['entity_type'] and
                    classification['confidence'] > 0.6 and
                    classification['recommendation'] in ['accept', 'review']):

                    detected.append({
                        'entity_type': classification['entity_type'],
                        'entity_id': '',  # Will be filled if matched to existing entity
                        'entity_name': entity_info['name'],
                        'mention_count': 1,
                        'detection_method': 'multi_agent_batch_semantic_classification',
                        'confidence': classification['confidence'],
                        'context': entity_info['context'],
                        'evidence': classification.get('reasoning', []),
                        'recommendation': classification['recommendation'],
                        'start_pos': entity_info.get('start_pos', 0),
                        'end_pos': entity_info.get('end_pos', 0)
                    })
            
            logger.info(f"Semantic classifier validated {len(detected)} entities from {len(potential_entities)} candidates")
            return detected
            
        except Exception as e:
            logger.error(f"Entity detection and classification failed: {e}")
            return []
    
    def _extract_potential_entity_names(self, content_text: str) -> List[Dict[str, Any]]:
        """Extract potential entity names using NLP patterns with selective processing."""
        potential_entities = []

        try:
            # Use regex patterns to extract proper nouns (limit to max 3 words)
            import re
            proper_noun_pattern = r'\b[A-Z][a-zA-Z]{1,}(?:\s+[A-Z][a-zA-Z]{1,}){0,2}\b'

            # Enhanced filtering lists for selective processing
            common_words = {
                'the', 'and', 'or', 'but', 'with', 'from', 'to', 'in', 'on', 'at', 'by', 'for',
                'of', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
                'what', 'who', 'which', 'whose', 'whom', 'all', 'any', 'some', 'many', 'much',
                'few', 'little', 'more', 'most', 'other', 'another', 'such', 'no', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just', 'now', 'also', 'still',
                # Add common descriptive words that shouldn't be entities
                'character', 'protagonist', 'story', 'main', 'during', 'their', 'adventures',
                'together', 'later', 'both', 'under', 'wise', 'ancient', 'legendary'
            }

            # Low-confidence patterns that should be deprioritized
            low_confidence_patterns = [
                r'^[A-Z][a-z]$',  # Single letter + lowercase (e.g., "It", "He")
                r'^\d+$',  # Pure numbers
                r'^[A-Z]{1,3}$',  # Short acronyms (often false positives)
                r'^(Mr|Mrs|Ms|Dr|Prof|Sir|Lady|Lord)$',  # Titles without names
                r'^(January|February|March|April|May|June|July|August|September|October|November|December)$',  # Months
                r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)$',  # Days
                r'^(North|South|East|West|Up|Down|Left|Right|Over|Under)$',  # Directions
            ]

            for match in re.finditer(proper_noun_pattern, content_text):
                name = match.group().strip()

                # Basic filtering
                if len(name) < 2 or name.lower() in common_words:
                    continue

                # Filter out names containing common descriptive words
                name_words = name.lower().split()
                if any(word in common_words for word in name_words):
                    continue

                # Skip if it's just "The" + something
                if name.startswith('The ') and len(name_words) == 2:
                    continue

                # Get enhanced context with larger window
                start_pos = match.start()
                end_pos = match.end()
                context = self._extract_enhanced_context(content_text, start_pos, end_pos, name)

                # Calculate pre-classification confidence score
                pre_confidence = self._calculate_pre_classification_confidence(name, context)

                # Skip very low confidence entities to save API calls
                if pre_confidence < 0.3:
                    logger.debug(f"Skipping low-confidence entity '{name}' (confidence: {pre_confidence:.2f})")
                    continue

                potential_entities.append({
                    'name': name,
                    'context': context,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'detection_method': 'regex_proper_noun',
                    'pre_confidence': pre_confidence
                })

            # Sort by pre-confidence (highest first) to prioritize better candidates
            potential_entities.sort(key=lambda x: x.get('pre_confidence', 0.5), reverse=True)

            # Limit the number of entities to process (cost optimization)
            max_entities = 50  # Configurable limit
            if len(potential_entities) > max_entities:
                logger.info(f"Limiting entity processing from {len(potential_entities)} to {max_entities} highest-confidence candidates")
                potential_entities = potential_entities[:max_entities]

            return potential_entities

        except Exception as e:
            logger.error(f"Failed to extract potential entity names: {e}")
            return []

    def _calculate_pre_classification_confidence(self, name: str, context: str) -> float:
        """Calculate a pre-classification confidence score to filter entities before API calls."""
        confidence = 0.5  # Base confidence

        # Name-based scoring
        if len(name) >= 3:
            confidence += 0.1
        if len(name) >= 5:
            confidence += 0.1

        # Check for multiple words (often better entities)
        if ' ' in name:
            confidence += 0.15

        # Context-based scoring
        context_lower = context.lower()

        # Character indicators
        character_indicators = ['said', 'spoke', 'told', 'asked', 'replied', 'whispered', 'shouted', 'he ', 'she ', 'they ', 'king', 'queen', 'prince', 'princess', 'lord', 'lady', 'sir', 'captain', 'wizard', 'knight']
        for indicator in character_indicators:
            if indicator in context_lower:
                confidence += 0.1
                break

        # Location indicators
        location_indicators = ['in ', 'at ', 'to ', 'from ', 'near ', 'castle', 'city', 'town', 'village', 'forest', 'mountain', 'river', 'sea', 'kingdom', 'realm', 'land', 'temple', 'tower', 'palace']
        for indicator in location_indicators:
            if indicator in context_lower:
                confidence += 0.1
                break

        # Lore indicators
        lore_indicators = ['magic', 'spell', 'curse', 'prophecy', 'legend', 'myth', 'order', 'guild', 'council', 'war', 'battle', 'ceremony', 'ritual', 'artifact', 'relic']
        for indicator in lore_indicators:
            if indicator in context_lower:
                confidence += 0.1
                break

        # Penalty for common false positives
        false_positive_patterns = ['annual', 'founded', 'cold', 'military', 'general', 'special', 'local', 'national', 'international', 'global']
        for pattern in false_positive_patterns:
            if pattern in name.lower():
                confidence -= 0.2
                break

        # Ensure confidence stays within bounds
        return max(0.0, min(1.0, confidence))

    def _extract_enhanced_context(self, content_text: str, start_pos: int, end_pos: int, entity_name: str) -> str:
        """Extract enhanced context with intelligent boundaries and larger windows."""
        import re

        # Base context window (larger than before)
        base_window = 150  # Increased from 50

        # Try to find sentence boundaries for more natural context
        sentence_start = start_pos
        sentence_end = end_pos

        # Look backwards for sentence start
        for i in range(start_pos - 1, max(0, start_pos - base_window * 2), -1):
            if content_text[i] in '.!?':
                # Found sentence boundary, but make sure it's not an abbreviation
                if i + 1 < len(content_text) and content_text[i + 1].isspace():
                    sentence_start = i + 1
                    break

        # Look forwards for sentence end
        for i in range(end_pos, min(len(content_text), end_pos + base_window * 2)):
            if content_text[i] in '.!?':
                # Found sentence boundary
                if i + 1 < len(content_text):
                    sentence_end = i + 1
                    break

        # If no sentence boundaries found, use paragraph boundaries
        if sentence_start == start_pos:
            for i in range(start_pos - 1, max(0, start_pos - base_window * 3), -1):
                if content_text[i] == '\n' and i > 0 and content_text[i-1] == '\n':
                    sentence_start = i + 1
                    break

        if sentence_end == end_pos:
            for i in range(end_pos, min(len(content_text), end_pos + base_window * 3)):
                if content_text[i] == '\n' and i + 1 < len(content_text) and content_text[i+1] == '\n':
                    sentence_end = i
                    break

        # Fallback to fixed window if no boundaries found
        if sentence_start == start_pos:
            sentence_start = max(0, start_pos - base_window)
        if sentence_end == end_pos:
            sentence_end = min(len(content_text), end_pos + base_window)

        # Extract context
        context = content_text[sentence_start:sentence_end].strip()

        # Clean up context
        context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
        context = context.replace('\n', ' ')    # Remove line breaks

        # Ensure the entity name is highlighted in context for better classification
        if entity_name.lower() not in context.lower():
            # If entity not in context (shouldn't happen), add it
            context = f"...{entity_name}... {context}"

        # Limit context length to prevent token overflow
        max_context_length = 300  # Increased from previous limits
        if len(context) > max_context_length:
            # Try to keep the entity name in the center
            entity_pos = context.lower().find(entity_name.lower())
            if entity_pos != -1:
                # Center the context around the entity
                start_trim = max(0, entity_pos - max_context_length // 2)
                end_trim = min(len(context), start_trim + max_context_length)
                context = context[start_trim:end_trim]
                if start_trim > 0:
                    context = "..." + context
                if end_trim < len(content_text):
                    context = context + "..."
            else:
                # Fallback: just truncate
                context = context[:max_context_length] + "..."

        return context
    
    def _detect_relationships(self, 
                            source_entity: Dict[str, Any],
                            detected_entities: List[Dict[str, Any]],
                            novel_id: str) -> List[Dict[str, Any]]:
        """Use relationship detection agent to find relationships."""
        try:
            if self.relationship_detector.is_available():
                return self.relationship_detector.detect_relationships(
                    source_entity, detected_entities, novel_id
                )
            else:
                logger.warning("Relationship detector not available, using fallback")
                return self._fallback_relationship_detection(source_entity, detected_entities)
                
        except Exception as e:
            logger.error(f"Relationship detection failed: {e}")
            return []
    
    def _generate_new_entity_suggestions(self, 
                                       detected_entities: List[Dict[str, Any]],
                                       novel_id: str) -> List[Dict[str, Any]]:
        """Generate suggestions for new entities that don't exist yet."""
        suggestions = []
        
        try:
            for entity in detected_entities:
                entity_name = entity.get('entity_name', '')
                entity_type = entity.get('entity_type', '')
                confidence = entity.get('confidence', 0.0)
                
                # Check if entity already exists
                if self._entity_exists(entity_name, novel_id):
                    continue
                
                # Only suggest high-confidence entities
                if confidence < 0.7:
                    continue
                
                suggestion = {
                    'name': entity_name,
                    'type': entity_type,
                    'description': f"Detected {entity_type} mentioned in content",
                    'confidence': confidence,
                    'evidence': entity.get('context', ''),
                    'detection_method': entity.get('detection_method', 'unknown')
                }
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"New entity suggestion generation failed: {e}")
            return []
    
    def _entity_exists(self, entity_name: str, novel_id: str) -> bool:
        """Check if entity already exists in the database."""
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
    
    def _fallback_relationship_detection(self, 
                                       source_entity: Dict[str, Any],
                                       detected_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback relationship detection using simple co-occurrence."""
        relationships = []
        source_name = source_entity.get('name') or source_entity.get('title', '')
        
        for entity in detected_entities:
            entity_name = entity.get('entity_name', '')
            
            if entity_name.lower() != source_name.lower():
                relationship = {
                    'source_entity': source_name,
                    'source_type': source_entity.get('entity_type', 'unknown'),
                    'target_entity': entity_name,
                    'target_type': entity.get('entity_type', 'unknown'),
                    'relationship_type': 'mentioned_together',
                    'confidence': 0.6,
                    'evidence': 'Mentioned in the same content',
                    'detection_method': 'fallback_cooccurrence'
                }
                relationships.append(relationship)
        
        return relationships
    
    def _get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for each agent."""
        return {
            'entity_classifier_available': self.entity_classifier.is_available(),
            'relationship_detector_available': self.relationship_detector.is_available(),
            'verification_agent_available': self.verification_agent.is_available(),
            'update_generator_available': self.update_generator.is_available(),
            'semantic_search_available': self.semantic_search is not None
        }
    
    def _create_empty_result(self, entity_type: str, entity_id: str, entity_name: str, novel_id: str, reason: str) -> Dict[str, Any]:
        """Create empty result with reason."""
        return {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'entity_name': entity_name,
            'novel_id': novel_id,
            'detected_entities': [],
            'semantic_matches': [],
            'verified_relationships': [],
            'suggested_updates': [],
            'new_entities': [],
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_duration': 0.0,
            'verification_summary': {'reason': reason},
            'agent_performance': self._get_agent_performance(),
            'success': True
        }
    
    def _create_error_result(self, entity_type: str, entity_id: str, entity_name: str, novel_id: str, error: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'entity_name': entity_name,
            'novel_id': novel_id,
            'detected_entities': [],
            'semantic_matches': [],
            'verified_relationships': [],
            'suggested_updates': [],
            'new_entities': [],
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_duration': 0.0,
            'verification_summary': {'error': error},
            'agent_performance': self._get_agent_performance(),
            'success': False
        }
