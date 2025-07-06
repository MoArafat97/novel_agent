"""
AI Cross-Reference Agent

This agent analyzes newly created or edited content (characters, locations, lore) to identify
relationships with existing entities. It uses lightweight text analysis for initial detection,
leverages existing semantic search infrastructure, and uses DeepSeek LLM for verification
and update generation.
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
import openai
from dotenv import load_dotenv
from utils.entity_detection import EntityDetectionUtils
from utils.stanza_entity_recognizer import OptimizedEntityRecognizer
from utils.entity_type_classifier import EntityTypeClassifier
from utils.streaming_analysis import get_streaming_manager, AnalysisStage
from utils.change_history import get_change_history_manager
from .multi_agent_coordinator import MultiAgentCoordinator

load_dotenv()
logger = logging.getLogger(__name__)

class CrossReferenceAgent:
    """
    AI agent that analyzes content to identify cross-references and relationships
    between worldbuilding entities using hybrid text analysis and LLM verification.
    """
    
    def __init__(self, world_state=None, semantic_search=None):
        """Initialize the Cross-Reference Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        self.world_state = world_state
        self.semantic_search = semantic_search
        self.entity_detector = EntityDetectionUtils()

        # Initialize streaming manager
        self.streaming_manager = get_streaming_manager()

        # Initialize change history manager
        self.change_history_manager = get_change_history_manager()

        # Initialize the multi-agent coordinator (Phase 2)
        try:
            self.multi_agent_coordinator = MultiAgentCoordinator(world_state, semantic_search)
            logger.info("Multi-Agent Coordinator initialized for cross-reference agent")
            self.use_multi_agent = True
        except Exception as e:
            logger.warning(f"Failed to initialize Multi-Agent Coordinator: {e}. Falling back to Phase 1 system.")
            self.multi_agent_coordinator = None
            self.use_multi_agent = False

        # Initialize Phase 1 components as fallback
        try:
            self.entity_classifier = EntityTypeClassifier()
            logger.info("EntityTypeClassifier initialized for cross-reference agent")
        except Exception as e:
            logger.warning(f"Failed to initialize EntityTypeClassifier: {e}. Falling back to basic detection.")
            self.entity_classifier = None

        # Initialize the optimized entity recognizer
        try:
            self.optimized_recognizer = OptimizedEntityRecognizer(
                world_state=world_state,
                cache_size=1000
            )
            logger.info("OptimizedEntityRecognizer initialized for cross-reference agent")
        except Exception as e:
            logger.warning(f"Failed to initialize OptimizedEntityRecognizer: {e}. Falling back to basic detection.")
            self.optimized_recognizer = None

        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Cross-reference agent will use basic analysis only.")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client for LLM operations."""
        try:
            self.openrouter_client = openai.OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized for cross-reference agent")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None
    
    def is_available(self) -> bool:
        """Check if the agent is available for use."""
        return self.openrouter_client is not None and self.world_state is not None
    
    def analyze_content(self, 
                       entity_type: str,
                       entity_id: str,
                       entity_data: Dict[str, Any],
                       novel_id: str) -> Dict[str, Any]:
        """
        Analyze content to identify cross-references and relationships.
        
        Args:
            entity_type: Type of entity being analyzed (characters, locations, lore)
            entity_id: ID of the entity being analyzed
            entity_data: Data of the entity being analyzed
            novel_id: ID of the novel this entity belongs to
            
        Returns:
            Dictionary containing analysis results with suggested updates
        """
        if not self.is_available():
            logger.error("Cross-reference agent not available")
            return self._create_empty_analysis()
        
        try:
            entity_name = entity_data.get('name') or entity_data.get('title', 'Unknown')
            logger.info(f"Starting cross-reference analysis for {entity_type} '{entity_name}' (ID: {entity_id}) in novel {novel_id}")
            logger.info(f"Current entity will be excluded from cross-reference suggestions")

            # Phase 2: Use Multi-Agent System if available
            if self.use_multi_agent and self.multi_agent_coordinator and self.multi_agent_coordinator.is_available():
                logger.info("Using Phase 2 Multi-Agent System for analysis")
                result = self.multi_agent_coordinator.analyze_content(entity_type, entity_id, entity_data, novel_id)

                # Add analysis metadata
                result['analysis_method'] = 'multi_agent_phase2'
                result['phase'] = 2

                return result

            # Phase 1: Fallback to original system with improvements
            else:
                logger.info("Using Phase 1 System (fallback) for analysis")
                return self._analyze_content_phase1(entity_type, entity_id, entity_data, novel_id)

        except Exception as e:
            logger.error(f"Cross-reference analysis failed: {e}")
            return {
                'entity_type': entity_type,
                'entity_id': entity_id,
                'entity_name': entity_data.get('name', 'Unknown'),
                'novel_id': novel_id,
                'detected_entities': [],
                'semantic_matches': [],
                'verified_relationships': [],
                'suggested_updates': [],
                'new_entities': [],
                'analysis_timestamp': self._get_timestamp(),
                'error': str(e),
                'success': False,
                'analysis_method': 'error',
                'phase': 0
            }

    def analyze_content_streaming(self, entity_type: str, entity_id: str, entity_data: Dict[str, Any], novel_id: str) -> str:
        """
        Perform cross-reference analysis with streaming progress updates.

        Args:
            entity_type: Type of the entity
            entity_id: ID of the entity
            entity_data: Entity data
            novel_id: Novel ID

        Returns:
            Job ID for tracking progress
        """
        # Create streaming job
        job_id = self.streaming_manager.create_job(entity_type, entity_id, novel_id)

        # Start analysis in background thread
        import threading
        analysis_thread = threading.Thread(
            target=self._run_streaming_analysis,
            args=(job_id, entity_type, entity_id, entity_data, novel_id),
            daemon=True
        )
        analysis_thread.start()

        return job_id

    def _run_streaming_analysis(self, job_id: str, entity_type: str, entity_id: str, entity_data: Dict[str, Any], novel_id: str):
        """Run the actual analysis with progress updates."""
        try:
            entity_name = entity_data.get('name') or entity_data.get('title', 'Unknown')

            # Stage 1: Initialize
            self.streaming_manager.update_progress(
                job_id, AnalysisStage.INITIALIZING, 0.05,
                f"Starting analysis for {entity_type} '{entity_name}'"
            )

            # Stage 2: Extract content
            self.streaming_manager.update_progress(
                job_id, AnalysisStage.EXTRACTING_CONTENT, 0.10,
                "Extracting entity content for analysis"
            )

            # Check if we should use multi-agent system
            if self.use_multi_agent and self.multi_agent_coordinator and self.multi_agent_coordinator.is_available():
                result = self._run_multi_agent_streaming_analysis(job_id, entity_type, entity_id, entity_data, novel_id)
            else:
                result = self._run_phase1_streaming_analysis(job_id, entity_type, entity_id, entity_data, novel_id)

            # Final stage: Complete
            self.streaming_manager.update_progress(
                job_id, AnalysisStage.COMPLETED, 1.0,
                "Analysis completed successfully",
                data={'result': result}
            )

            self.streaming_manager.set_job_result(job_id, result)

        except Exception as e:
            logger.error(f"Streaming analysis failed for job {job_id}: {e}")
            self.streaming_manager.update_progress(
                job_id, AnalysisStage.ERROR, 0.0,
                f"Analysis failed: {str(e)}"
            )
            self.streaming_manager.set_job_error(job_id, str(e))

    def _run_multi_agent_streaming_analysis(self, job_id: str, entity_type: str, entity_id: str, entity_data: Dict[str, Any], novel_id: str) -> Dict[str, Any]:
        """Run multi-agent analysis with streaming progress updates."""
        entity_name = entity_data.get('name') or entity_data.get('title', 'Unknown')

        # Stage 3: Entity Detection
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.DETECTING_ENTITIES, 0.20,
            "Detecting potential entities in content"
        )

        content_text = self._extract_entity_content(entity_data, entity_type)
        if not content_text or len(content_text.strip()) < 10:
            return self._create_empty_result(entity_type, entity_id, entity_name, novel_id, "Insufficient content for analysis")

        # Stage 4: Entity Classification
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.CLASSIFYING_ENTITIES, 0.40,
            "Classifying detected entities using AI"
        )

        detected_entities = self.multi_agent_coordinator._detect_and_classify_entities(content_text, novel_id)
        if not detected_entities:
            return self._create_empty_result(entity_type, entity_id, entity_name, novel_id, "No entities detected")

        # Stage 5: Relationship Detection
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.FINDING_RELATIONSHIPS, 0.60,
            f"Finding relationships between {len(detected_entities)} entities"
        )

        relationships = self.multi_agent_coordinator._detect_relationships(entity_data, detected_entities, novel_id)

        # Stage 6: Generate Updates
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.GENERATING_UPDATES, 0.80,
            "Generating suggested updates"
        )

        new_entities = self.multi_agent_coordinator._generate_new_entity_suggestions(detected_entities, novel_id)

        # Stage 7: Verification
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.VERIFYING_RESULTS, 0.90,
            "Verifying analysis results"
        )

        verification_results = self.multi_agent_coordinator.verification_agent.verify_analysis_results(
            detected_entities, relationships, new_entities, entity_data, novel_id
        )

        if not verification_results.get('success'):
            return self._create_error_result(entity_type, entity_id, entity_name, novel_id,
                                           verification_results.get('verification_summary', {}).get('error', 'Verification failed'))

        suggested_updates = self.multi_agent_coordinator.update_generator.generate_updates(
            verification_results['relationships'],
            verification_results['entities'],
            entity_data,
            novel_id
        )

        new_entity_suggestions = self.multi_agent_coordinator.update_generator.generate_new_entity_suggestions(
            verification_results['new_entities'],
            novel_id
        )

        # Compile final results
        return {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'entity_name': entity_name,
            'novel_id': novel_id,
            'detected_entities': verification_results['entities'],
            'semantic_matches': [],
            'verified_relationships': verification_results['relationships'],
            'suggested_updates': suggested_updates,
            'new_entities': new_entity_suggestions,
            'analysis_timestamp': self._get_timestamp(),
            'success': True,
            'analysis_method': 'multi_agent_streaming',
            'phase': 2
        }

    def _run_phase1_streaming_analysis(self, job_id: str, entity_type: str, entity_id: str, entity_data: Dict[str, Any], novel_id: str) -> Dict[str, Any]:
        """Run Phase 1 analysis with streaming progress updates."""
        entity_name = entity_data.get('name') or entity_data.get('title', 'Unknown')

        # Stage 3: Entity Detection
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.DETECTING_ENTITIES, 0.25,
            "Extracting entity content and detecting mentions"
        )

        content_text = self._extract_entity_content(entity_data, entity_type)
        detected_entities = self._detect_entity_mentions(content_text, novel_id)

        # Stage 4: Semantic Search
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.FINDING_RELATIONSHIPS, 0.50,
            "Finding semantic matches and relationships"
        )

        semantic_matches = self._find_semantic_matches(content_text, novel_id, entity_id)

        # Stage 5: LLM Verification
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.VERIFYING_RESULTS, 0.75,
            "Verifying relationships with AI analysis"
        )

        verified_relationships = self._verify_relationships_with_llm(
            entity_type, entity_data, detected_entities, semantic_matches, novel_id
        )

        # Stage 6: Generate Updates
        self.streaming_manager.update_progress(
            job_id, AnalysisStage.GENERATING_UPDATES, 0.90,
            "Generating suggested updates"
        )

        suggested_updates = self._generate_suggested_updates(verified_relationships, novel_id)

        if not suggested_updates and (detected_entities or semantic_matches):
            suggested_updates = self._generate_basic_updates(
                entity_type, entity_data, detected_entities, semantic_matches, novel_id
            )

        new_entities = self._detect_new_entities(content_text, novel_id)

        return {
            'entity_type': entity_type,
            'entity_id': entity_id,
            'entity_name': entity_name,
            'novel_id': novel_id,
            'detected_entities': detected_entities,
            'semantic_matches': semantic_matches,
            'verified_relationships': verified_relationships,
            'suggested_updates': suggested_updates,
            'new_entities': new_entities,
            'analysis_timestamp': self._get_timestamp(),
            'success': True,
            'analysis_method': 'phase1_streaming',
            'phase': 1
        }

    def _analyze_content_phase1(self, entity_type: str, entity_id: str, entity_data: Dict[str, Any], novel_id: str) -> Dict[str, Any]:
        """
        Phase 1 analysis method with semantic classification improvements.
        """
        try:
            entity_name = entity_data.get('name') or entity_data.get('title', 'Unknown')
            logger.info(f"Starting Phase 1 cross-reference analysis for {entity_type} '{entity_name}'")

            # Step 1: Extract text content for analysis
            content_text = self._extract_content_text(entity_type, entity_data)

            # Step 2: Lightweight entity detection using regex and patterns (with Phase 1 improvements)
            detected_entities = self._detect_entity_mentions(content_text, novel_id)

            # Step 3: Semantic search for related entities
            semantic_matches = self._find_semantic_matches(content_text, novel_id, entity_id)

            # Step 4: LLM verification and relationship analysis
            verified_relationships = self._verify_relationships_with_llm(
                entity_type, entity_data, detected_entities, semantic_matches, novel_id
            )

            # Step 5: Generate suggested updates
            suggested_updates = self._generate_suggested_updates(
                verified_relationships, novel_id
            )

            # Step 5b: If no LLM updates, generate basic updates from detected entities
            if not suggested_updates and (detected_entities or semantic_matches):
                suggested_updates = self._generate_basic_updates(
                    entity_type, entity_data, detected_entities, semantic_matches, novel_id
                )

            # Step 6: Detect potential new entities
            new_entities = self._detect_new_entities(content_text, novel_id)

            analysis_result = {
                'entity_type': entity_type,
                'entity_id': entity_id,
                'entity_name': entity_name,
                'novel_id': novel_id,
                'detected_entities': detected_entities,
                'semantic_matches': semantic_matches,
                'verified_relationships': verified_relationships,
                'suggested_updates': suggested_updates,
                'new_entities': new_entities,
                'analysis_timestamp': self._get_timestamp(),
                'success': True,
                'analysis_method': 'phase1_with_improvements',
                'phase': 1
            }

            logger.info(f"Phase 1 cross-reference analysis completed successfully. Found {len(verified_relationships)} relationships, {len(suggested_updates)} updates, {len(new_entities)} new entities")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Cross-reference analysis failed: {e}")
            return self._create_error_analysis(str(e))
    
    def _extract_content_text(self, entity_type: str, entity_data: Dict[str, Any]) -> str:
        """Extract all relevant text content from an entity for analysis."""
        text_parts = []

        # Helper function to safely add field content
        def add_field_content(field_value):
            if isinstance(field_value, list):
                # If it's a list, join the items and add as a single string
                text_parts.append(' '.join(str(item) for item in field_value))
            elif isinstance(field_value, str):
                text_parts.append(field_value)
            else:
                # Convert other types to string
                text_parts.append(str(field_value))

        # Common fields
        if entity_data.get('name'):
            add_field_content(entity_data['name'])
        if entity_data.get('title'):
            add_field_content(entity_data['title'])
        if entity_data.get('description'):
            add_field_content(entity_data['description'])

        # Entity-specific fields
        if entity_type == 'characters':
            for field in ['personality', 'backstory', 'occupation']:
                if entity_data.get(field):
                    add_field_content(entity_data[field])
        elif entity_type == 'locations':
            for field in ['geography', 'climate', 'culture', 'history', 'notable_features']:
                if entity_data.get(field):
                    add_field_content(entity_data[field])
        elif entity_type == 'lore':
            for field in ['details', 'significance', 'related_events']:
                if entity_data.get(field):
                    add_field_content(entity_data[field])

        # Tags
        if entity_data.get('tags'):
            add_field_content(entity_data['tags'])

        return ' '.join(text_parts)
    
    def _detect_entity_mentions(self, content_text: str, novel_id: str) -> List[Dict[str, Any]]:
        """Use semantic classification to detect and validate entity mentions."""
        detected = []

        if not self.world_state:
            return detected

        try:
            # Step 1: Extract potential entity names using NLP patterns
            potential_entities = self._extract_potential_entity_names(content_text)

            # Step 2: Use semantic classifier to validate and classify entities
            if self.entity_classifier and self.entity_classifier.is_available():
                logger.info(f"Classifying {len(potential_entities)} potential entities using semantic classifier")

                for entity_info in potential_entities:
                    classification = self.entity_classifier.classify_entity(
                        entity_info['name'],
                        entity_info['context'],
                        novel_id
                    )

                    # Only accept entities with good classification
                    if (classification['entity_type'] and
                        classification['confidence'] > 0.6 and  # Higher threshold for better quality
                        classification['recommendation'] in ['accept', 'review']):

                        detected.append({
                            'entity_type': classification['entity_type'],
                            'entity_id': '',  # Will be filled if matched to existing entity
                            'entity_name': entity_info['name'],
                            'mention_count': 1,
                            'detection_method': f"semantic_classification",
                            'confidence': classification['confidence'],
                            'context': entity_info['context'],
                            'evidence': classification.get('reasoning', []),
                            'recommendation': classification['recommendation'],
                            'start_pos': entity_info.get('start_pos', 0),
                            'end_pos': entity_info.get('end_pos', 0)
                        })

                logger.info(f"Semantic classifier validated {len(detected)} entities from {len(potential_entities)} candidates")

            else:
                # Fallback: Use existing entity matching when classifier unavailable
                logger.warning("Semantic classifier not available, falling back to existing entity matching")
                detected = self._fallback_entity_detection(content_text, novel_id)

            return detected

        except Exception as e:
            logger.error(f"Entity mention detection failed: {e}")
            return []

    def _extract_potential_entity_names(self, content_text: str) -> List[Dict[str, Any]]:
        """Extract potential entity names using NLP patterns."""
        potential_entities = []

        try:
            # Use Stanza if available through optimized recognizer
            if self.optimized_recognizer and hasattr(self.optimized_recognizer, 'nlp'):
                doc = self.optimized_recognizer.nlp(content_text)

                # Extract named entities from Stanza
                for ent in doc.ents:
                    if len(ent.text.strip()) >= 2:  # Minimum length filter
                        # Get context around the entity
                        start_char = ent.start_char
                        end_char = ent.end_char
                        context_start = max(0, start_char - 50)
                        context_end = min(len(content_text), end_char + 50)
                        context = content_text[context_start:context_end]

                        potential_entities.append({
                            'name': ent.text.strip(),
                            'context': context,
                            'start_pos': start_char,
                            'end_pos': end_char,
                            'stanza_type': ent.type
                        })

            # Also extract proper nouns using regex patterns
            import re
            proper_noun_pattern = r'\b[A-Z][a-zA-Z]{1,}(?:\s+[A-Z][a-zA-Z]{1,})*\b'

            for match in re.finditer(proper_noun_pattern, content_text):
                name = match.group().strip()

                # Skip if already found by Stanza
                if any(e['name'] == name for e in potential_entities):
                    continue

                # Skip common words and short names
                if len(name) < 2 or name.lower() in {'the', 'and', 'or', 'but', 'with', 'from', 'to'}:
                    continue

                # Get context
                start_pos = match.start()
                end_pos = match.end()
                context_start = max(0, start_pos - 50)
                context_end = min(len(content_text), end_pos + 50)
                context = content_text[context_start:context_end]

                potential_entities.append({
                    'name': name,
                    'context': context,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'detection_method': 'regex_proper_noun'
                })

            return potential_entities

        except Exception as e:
            logger.error(f"Failed to extract potential entity names: {e}")
            return []

    def _fallback_entity_detection(self, content_text: str, novel_id: str) -> List[Dict[str, Any]]:
        """Fallback entity detection using existing entity matching."""
        detected = []

        try:
            novel_entities = self.world_state.get_entities_by_novel(novel_id)

            # Create search patterns for each existing entity
            for entity_type, entities in novel_entities.items():
                for entity in entities:
                    entity_name = entity.get('name') or entity.get('title', '')
                    if not entity_name or len(entity_name) < 2:
                        continue

                    # Simple case-insensitive search
                    pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
                    matches = list(pattern.finditer(content_text))

                    if matches:
                        detected.append({
                            'entity_type': entity_type,
                            'entity_id': entity['id'],
                            'entity_name': entity_name,
                            'mention_count': len(matches),
                            'detection_method': 'regex_exact_match',
                            'confidence': 0.9,  # High confidence for exact matches
                            'context': content_text[max(0, matches[0].start()-50):matches[0].end()+50]
                        })

            return detected

        except Exception as e:
            logger.error(f"Fallback entity detection failed: {e}")
            return []
    
    def _find_semantic_matches(self, content_text: str, novel_id: str, exclude_entity_id: str) -> List[Dict[str, Any]]:
        """Use semantic search to find related entities."""
        if not self.semantic_search:
            return []
        
        try:
            # Perform semantic search
            results = self.semantic_search.search(
                query=content_text,
                novel_id=novel_id,
                n_results=10,
                min_similarity=0.3
            )
            
            # Filter out the entity being analyzed
            filtered_results = [
                r for r in results 
                if r.get('entity_id') != exclude_entity_id
            ]
            
            return filtered_results[:5]  # Limit to top 5 matches
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return []
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        """Create empty analysis result."""
        return {
            'detected_entities': [],
            'semantic_matches': [],
            'verified_relationships': [],
            'suggested_updates': [],
            'new_entities': [],
            'analysis_timestamp': self._get_timestamp(),
            'success': False,
            'error': 'Agent not available'
        }
    
    def _create_error_analysis(self, error_message: str) -> Dict[str, Any]:
        """Create error analysis result."""
        return {
            'detected_entities': [],
            'semantic_matches': [],
            'verified_relationships': [],
            'suggested_updates': [],
            'new_entities': [],
            'analysis_timestamp': self._get_timestamp(),
            'success': False,
            'error': error_message
        }

    def _verify_relationships_with_llm(self,
                                     entity_type: str,
                                     entity_data: Dict[str, Any],
                                     detected_entities: List[Dict[str, Any]],
                                     semantic_matches: List[Dict[str, Any]],
                                     novel_id: str) -> List[Dict[str, Any]]:
        """Use LLM to verify and analyze relationships between entities."""
        if not self.openrouter_client:
            return []

        try:
            # Combine detected entities and semantic matches
            all_candidates = detected_entities.copy()  # Start with detected entities

            # Add semantic matches with proper entity type handling
            for match in semantic_matches:
                entity_data = match.get('data', {})
                entity_name = entity_data.get('name') or entity_data.get('title', '')
                entity_type = match.get('entity_type', '')

                # Only add if we have valid data
                if entity_name and entity_type:
                    all_candidates.append({
                        'entity_type': entity_type,
                        'entity_id': match.get('entity_id'),
                        'entity_name': entity_name,
                        'similarity_score': match.get('similarity_score', 0),
                        'detection_method': 'semantic_search'
                    })

            if not all_candidates:
                return []

            # Build LLM prompt for relationship verification
            system_prompt = self._build_relationship_verification_prompt()
            user_prompt = self._build_relationship_user_prompt(
                entity_type, entity_data, all_candidates, novel_id
            )

            # Call LLM
            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Parse LLM response
            ai_response = response.choices[0].message.content
            verified_relationships = self._parse_relationship_response(ai_response)

            return verified_relationships

        except Exception as e:
            logger.error(f"LLM relationship verification failed: {e}")
            return []

    def _build_relationship_verification_prompt(self) -> str:
        """Build system prompt for relationship verification."""
        return """You are a worldbuilding expert analyzing relationships between entities in a novel.

Your task is to examine potential connections between a primary entity and other entities, then determine:
1. Whether genuine relationships exist
2. The type and strength of relationships
3. What new information can be inferred about existing entities

Respond with a JSON array of verified relationships. Each relationship should have:
- "target_entity_id": ID of the related entity
- "target_entity_name": Name of the related entity
- "relationship_type": Type of relationship (knows, lives_in, works_at, related_to, enemy_of, etc.)
- "relationship_strength": Strength (weak, moderate, strong)
- "evidence": Brief explanation of the evidence
- "new_information": Any new information discovered about the target entity
- "confidence": Confidence level (low, medium, high)

Only include relationships you are confident about. Be conservative and avoid speculation."""

    def _build_relationship_user_prompt(self,
                                      entity_type: str,
                                      entity_data: Dict[str, Any],
                                      candidates: List[Dict[str, Any]],
                                      novel_id: str) -> str:
        """Build user prompt for relationship verification."""
        entity_name = entity_data.get('name') or entity_data.get('title')

        prompt_parts = [
            f"Primary Entity: {entity_name} ({entity_type})",
            f"Description: {entity_data.get('description', 'No description')}",
        ]

        # Add entity-specific details
        if entity_type == 'characters':
            if entity_data.get('personality'):
                prompt_parts.append(f"Personality: {entity_data['personality']}")
            if entity_data.get('backstory'):
                prompt_parts.append(f"Backstory: {entity_data['backstory']}")
            if entity_data.get('occupation'):
                prompt_parts.append(f"Occupation: {entity_data['occupation']}")
        elif entity_type == 'locations':
            if entity_data.get('geography'):
                prompt_parts.append(f"Geography: {entity_data['geography']}")
            if entity_data.get('culture'):
                prompt_parts.append(f"Culture: {entity_data['culture']}")
        elif entity_type == 'lore':
            if entity_data.get('details'):
                prompt_parts.append(f"Details: {entity_data['details']}")

        prompt_parts.append("\nPotential Related Entities:")
        for i, candidate in enumerate(candidates[:10], 1):  # Limit to 10 candidates
            prompt_parts.append(
                f"{i}. {candidate['entity_name']} ({candidate['entity_type']}) "
                f"[ID: {candidate['entity_id']}] "
                f"[Detection: {candidate['detection_method']}]"
            )

        prompt_parts.append("\nAnalyze these potential relationships and respond with verified connections in JSON format.")

        return '\n'.join(prompt_parts)

    def _parse_relationship_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for verified relationships."""
        try:
            # Try to extract JSON from the response
            json_start = ai_response.find('[')
            json_end = ai_response.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                relationships = json.loads(json_str)

                # Validate and clean the relationships
                validated_relationships = []
                for rel in relationships:
                    if (isinstance(rel, dict) and
                        'target_entity_id' in rel and
                        'relationship_type' in rel):
                        validated_relationships.append(rel)

                return validated_relationships

            return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse relationship JSON: {e}")
            logger.error(f"AI Response: {ai_response}")
            return []
        except Exception as e:
            logger.error(f"Error parsing relationship response: {e}")
            return []

    def _generate_suggested_updates(self,
                                  verified_relationships: List[Dict[str, Any]],
                                  novel_id: str) -> List[Dict[str, Any]]:
        """Generate suggested updates for related entities based on verified relationships."""
        suggested_updates = []

        if not self.openrouter_client or not verified_relationships:
            return suggested_updates

        try:
            for relationship in verified_relationships:
                target_entity_id = relationship.get('target_entity_id')
                new_information = relationship.get('new_information')

                if not target_entity_id or not new_information:
                    continue

                # Get the target entity
                target_entity = None
                target_entity_type = None

                for entity_type in ['characters', 'locations', 'lore']:
                    entity = self.world_state.get(entity_type, target_entity_id)
                    if entity:
                        target_entity = entity
                        target_entity_type = entity_type
                        break

                if not target_entity:
                    continue

                # Generate specific update suggestions using LLM
                update_suggestion = self._generate_entity_update(
                    target_entity_type, target_entity, new_information, relationship
                )

                if update_suggestion:
                    suggested_updates.append({
                        'target_entity_id': target_entity_id,
                        'target_entity_type': target_entity_type,
                        'target_entity_name': target_entity.get('name') or target_entity.get('title'),
                        'relationship': relationship,
                        'suggested_changes': update_suggestion,
                        'confidence': relationship.get('confidence', 'medium')
                    })

            return suggested_updates

        except Exception as e:
            logger.error(f"Failed to generate suggested updates: {e}")
            return []

    def _generate_entity_update(self,
                              entity_type: str,
                              entity_data: Dict[str, Any],
                              new_information: str,
                              relationship: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate specific update suggestions for an entity."""
        if not self.openrouter_client:
            return None

        try:
            system_prompt = """You are a worldbuilding expert. Given an existing entity and new information about it, suggest specific updates to the entity's fields.

Respond with a JSON object containing only the fields that should be updated, with their new values. Do not include fields that don't need changes.

Available fields by entity type:
- Characters: description, personality, backstory, occupation, tags
- Locations: description, geography, climate, culture, history, notable_features, tags
- Lore: description, details, significance, related_events, tags

Be conservative and only suggest changes that are clearly supported by the new information."""

            user_prompt = f"""Entity Type: {entity_type}
Entity Name: {entity_data.get('name') or entity_data.get('title')}

Current Entity Data:
{json.dumps(entity_data, indent=2)}

New Information: {new_information}
Relationship Context: {relationship.get('relationship_type')} - {relationship.get('evidence')}

Suggest specific field updates based on this new information."""

            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            ai_response = response.choices[0].message.content

            # Parse JSON response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                updates = json.loads(json_str)
                return updates if isinstance(updates, dict) else None

            return None

        except Exception as e:
            logger.error(f"Failed to generate entity update: {e}")
            return None

    def _detect_new_entities(self, content_text: str, novel_id: str) -> List[Dict[str, Any]]:
        """Detect potential new entities that should be added to the database."""
        try:
            # Get existing entities for this novel
            existing_entities = self.world_state.get_entities_by_novel(novel_id) if self.world_state else {}

            # Use lightweight detection first
            lightweight_detections = self.entity_detector.detect_potential_entities(content_text, existing_entities)

            # If LLM is available, enhance with AI analysis
            if self.openrouter_client:
                ai_detections = self._detect_new_entities_with_llm(content_text, novel_id)

                # Merge and deduplicate results
                all_detections = self._merge_entity_detections(lightweight_detections, ai_detections)
                return all_detections
            else:
                # Convert lightweight detections to expected format
                converted_detections = []
                for entity_type, entities in lightweight_detections.items():
                    for entity in entities:
                        converted_detections.append({
                            'name': entity['name'],
                            'type': entity_type,
                            'evidence': entity['context'],
                            'description': f"Detected {entity_type[:-1]} based on context analysis",
                            'confidence': entity['confidence'],
                            'detection_method': entity['detection_method']
                        })

                return converted_detections

        except Exception as e:
            logger.error(f"Failed to detect new entities: {e}")
            return []

    def _detect_new_entities_with_llm(self, content_text: str, novel_id: str) -> List[Dict[str, Any]]:
        """Use LLM to detect new entities."""
        try:
            system_prompt = """You are a worldbuilding expert. Analyze the given text and identify potential new entities (characters, locations, or lore concepts) that are mentioned but might not exist in the database yet.

Look for:
- Character names or titles
- Location names or places
- Important concepts, organizations, or lore elements

Respond with a JSON array of potential new entities. Each should have:
- "name": The entity name/title
- "type": Entity type (characters, locations, or lore)
- "evidence": Brief quote or context from the text
- "description": Brief description of what this entity appears to be
- "confidence": Confidence level (low, medium, high)

Only include entities that seem significant and well-defined. Avoid common words or vague references."""

            user_prompt = f"""Analyze this text for potential new entities:

{content_text}

Identify significant characters, locations, or lore concepts that appear to be important but might not be in the database yet."""

            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            ai_response = response.choices[0].message.content

            # Parse JSON response
            json_start = ai_response.find('[')
            json_end = ai_response.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                new_entities = json.loads(json_str)

                # Validate and filter results
                validated_entities = []
                for entity in new_entities:
                    if (isinstance(entity, dict) and
                        'name' in entity and
                        'type' in entity and
                        entity['type'] in ['characters', 'locations', 'lore']):

                        # Check if entity already exists
                        if not self._entity_exists(entity['name'], entity['type'], novel_id):
                            entity['detection_method'] = 'llm_analysis'
                            validated_entities.append(entity)

                return validated_entities

            return []

        except Exception as e:
            logger.error(f"LLM entity detection failed: {e}")
            return []

    def _merge_entity_detections(self, lightweight: Dict[str, List[Dict[str, Any]]],
                               ai_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge lightweight and AI detections, removing duplicates."""
        merged = []
        seen_names = set()

        # Add AI detections first (higher priority)
        for entity in ai_detections:
            name_key = f"{entity['name'].lower()}_{entity['type']}"
            if name_key not in seen_names:
                seen_names.add(name_key)
                merged.append(entity)

        # Add lightweight detections that weren't found by AI
        for entity_type, entities in lightweight.items():
            for entity in entities:
                name_key = f"{entity['name'].lower()}_{entity_type}"
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    merged.append({
                        'name': entity['name'],
                        'type': entity_type,
                        'evidence': entity['context'],
                        'description': f"Detected {entity_type[:-1]} based on pattern analysis",
                        'confidence': entity['confidence'],
                        'detection_method': entity['detection_method']
                    })

        return merged

    def _generate_basic_updates(self,
                              entity_type: str,
                              entity_data: Dict[str, Any],
                              detected_entities: List[Dict[str, Any]],
                              semantic_matches: List[Dict[str, Any]],
                              novel_id: str) -> List[Dict[str, Any]]:
        """Generate basic updates when LLM is not available."""
        basic_updates = []

        try:
            entity_name = entity_data.get('name') or entity_data.get('title', '')

            # Process detected entities for basic relationship updates
            for detected in detected_entities:
                target_entity_id = detected.get('entity_id')
                target_entity_type = detected.get('entity_type')
                target_entity_name = detected.get('entity_name')

                if not target_entity_id or not target_entity_type:
                    continue

                # Get the target entity
                target_entity = self.world_state.get(target_entity_type, target_entity_id)
                if not target_entity:
                    continue

                # Generate basic update based on entity types
                suggested_changes = self._generate_basic_relationship_update(
                    entity_type, entity_name, target_entity_type, target_entity, target_entity_name
                )

                if suggested_changes:
                    basic_updates.append({
                        'target_entity_id': target_entity_id,
                        'target_entity_type': target_entity_type,
                        'target_entity_name': target_entity_name,
                        'relationship': {
                            'relationship_type': 'mentioned_with',
                            'evidence': f'Mentioned together in {entity_name}\'s content',
                            'confidence': 'medium'
                        },
                        'suggested_changes': suggested_changes,
                        'confidence': 'medium'
                    })

            # Process semantic matches for basic updates
            for match in semantic_matches:
                entity_data_match = match.get('data', {})
                target_entity_id = match.get('entity_id')
                target_entity_type = match.get('entity_type')
                target_entity_name = entity_data_match.get('name') or entity_data_match.get('title', '')

                if not target_entity_id or not target_entity_type or not target_entity_name:
                    continue

                # Generate basic semantic relationship update
                suggested_changes = self._generate_basic_semantic_update(
                    entity_type, entity_name, target_entity_type, entity_data_match, target_entity_name
                )

                if suggested_changes:
                    basic_updates.append({
                        'target_entity_id': target_entity_id,
                        'target_entity_type': target_entity_type,
                        'target_entity_name': target_entity_name,
                        'relationship': {
                            'relationship_type': 'semantically_related',
                            'evidence': f'Semantically similar to {entity_name}',
                            'confidence': 'low'
                        },
                        'suggested_changes': suggested_changes,
                        'confidence': 'low'
                    })

            return basic_updates[:3]  # Limit to 3 basic updates

        except Exception as e:
            logger.error(f"Failed to generate basic updates: {e}")
            return []

    def _generate_basic_relationship_update(self,
                                          source_type: str,
                                          source_name: str,
                                          target_type: str,
                                          target_entity: Dict[str, Any],
                                          target_name: str) -> Optional[Dict[str, Any]]:
        """Generate basic relationship update between entities."""
        try:
            # Simple relationship-based updates
            if source_type == 'characters' and target_type == 'locations':
                # Character mentioned with location - add to character's known locations
                current_desc = target_entity.get('description', '')
                if source_name not in current_desc:
                    return {
                        'description': f"{current_desc} Known to {source_name}.".strip()
                    }

            elif source_type == 'characters' and target_type == 'characters':
                # Character mentioned with another character - add relationship note
                current_desc = target_entity.get('description', '')
                if source_name not in current_desc:
                    return {
                        'description': f"{current_desc} Associated with {source_name}.".strip()
                    }

            elif source_type == 'locations' and target_type == 'characters':
                # Location mentioned with character - add location to character's background
                current_backstory = target_entity.get('backstory', '')
                if source_name not in current_backstory:
                    return {
                        'backstory': f"{current_backstory} Connected to {source_name}.".strip()
                    }

            return None

        except Exception as e:
            logger.error(f"Failed to generate basic relationship update: {e}")
            return None

    def _generate_basic_semantic_update(self,
                                      source_type: str,
                                      source_name: str,
                                      target_type: str,
                                      target_entity: Dict[str, Any],
                                      target_name: str) -> Optional[Dict[str, Any]]:
        """Generate basic semantic relationship update."""
        try:
            # Add semantic relationship tags
            current_tags = target_entity.get('tags', [])
            new_tags = current_tags.copy()

            # Add relationship tag if not already present
            relationship_tag = f"related_to_{source_name.lower().replace(' ', '_')}"
            if relationship_tag not in new_tags:
                new_tags.append(relationship_tag)
                return {'tags': new_tags}

            return None

        except Exception as e:
            logger.error(f"Failed to generate basic semantic update: {e}")
            return None

    def _entity_exists(self, entity_name: str, entity_type: str, novel_id: str) -> bool:
        """Check if an entity already exists in the database."""
        if not self.world_state:
            return False

        try:
            novel_entities = self.world_state.get_entities_by_novel(novel_id)
            entities = novel_entities.get(entity_type, [])

            for entity in entities:
                existing_name = entity.get('name') or entity.get('title', '')
                if existing_name.lower() == entity_name.lower():
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking entity existence: {e}")
            return False

    def apply_updates(self,
                     updates_to_apply: List[Dict[str, Any]],
                     novel_id: str) -> Dict[str, Any]:
        """Apply approved updates to entities."""
        if not self.world_state:
            return {'success': False, 'error': 'WorldState not available'}

        try:
            applied_updates = []
            failed_updates = []

            for update in updates_to_apply:
                try:
                    entity_id = update['target_entity_id']
                    entity_type = update['target_entity_type']
                    changes = update['suggested_changes']

                    # Get current entity
                    current_entity = self.world_state.get(entity_type, entity_id)
                    if not current_entity:
                        failed_updates.append({
                            'entity_id': entity_id,
                            'error': 'Entity not found'
                        })
                        continue

                    # Apply changes
                    updated_entity = current_entity.copy()
                    for field, new_value in changes.items():
                        if field in ['name', 'title', 'description', 'personality', 'backstory',
                                   'occupation', 'geography', 'climate', 'culture', 'history',
                                   'notable_features', 'details', 'significance', 'related_events']:
                            updated_entity[field] = new_value
                        elif field == 'tags' and isinstance(new_value, list):
                            # Merge tags instead of replacing
                            existing_tags = set(updated_entity.get('tags', []))
                            new_tags = set(new_value)
                            updated_entity['tags'] = list(existing_tags.union(new_tags))

                    # Add update metadata
                    updated_entity['updated_at'] = self._get_timestamp()
                    updated_entity['cross_reference_updated'] = True

                    # Save to database
                    success = self.world_state.add_or_update(
                        entity_type, entity_id, updated_entity
                    )

                    if success:
                        applied_updates.append({
                            'entity_id': entity_id,
                            'entity_type': entity_type,
                            'entity_name': updated_entity.get('name') or updated_entity.get('title'),
                            'changes_applied': changes
                        })
                    else:
                        failed_updates.append({
                            'entity_id': entity_id,
                            'error': 'Database update failed'
                        })

                except Exception as e:
                    failed_updates.append({
                        'entity_id': update.get('target_entity_id', 'unknown'),
                        'error': str(e)
                    })

            return {
                'success': True,
                'applied_updates': applied_updates,
                'failed_updates': failed_updates,
                'total_applied': len(applied_updates),
                'total_failed': len(failed_updates)
            }

        except Exception as e:
            logger.error(f"Failed to apply updates: {e}")
            return {'success': False, 'error': str(e)}
