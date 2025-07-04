"""
LLM-based Entity Type Classifier

This module provides semantic classification of entities using DeepSeek LLM
with comprehensive entity type definitions and validation.
"""

import os
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple
import openai
from dotenv import load_dotenv
from .entity_type_definitions import EntityTypeDefinitions
from .cross_reference_cache import get_cache_manager
from .api_rate_limiter import get_rate_limiter, RateLimitConfig
from .confidence_calibration import get_confidence_calibrator
from .genre_specific_prompts import get_genre_prompts, Genre

load_dotenv()
logger = logging.getLogger(__name__)


class EntityTypeClassifier:
    """
    LLM-based entity type classifier with semantic understanding.
    """
    
    def __init__(self, enable_caching=True, enable_smart_fallbacks=True):
        """Initialize the classifier."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        self.entity_definitions = EntityTypeDefinitions()
        self.enable_caching = enable_caching
        self.enable_smart_fallbacks = enable_smart_fallbacks

        # Smart fallback model configuration
        self.cheap_model = 'deepseek/deepseek-chat:free'  # For initial filtering
        self.premium_model = 'anthropic/claude-3.5-sonnet'  # For final validation (if needed)
        self.fallback_confidence_threshold = 0.8  # Use premium model if cheap model confidence < threshold

        # Initialize cache manager
        if self.enable_caching:
            self.cache_manager = get_cache_manager()
        else:
            self.cache_manager = None

        # Initialize rate limiter
        rate_limit_config = RateLimitConfig(
            requests_per_minute=50,  # Conservative limit for DeepSeek
            requests_per_hour=800,
            max_concurrent_requests=3,  # Limit concurrent requests
            base_delay=1.0,
            max_delay=30.0
        )
        self.rate_limiter = get_rate_limiter(rate_limit_config)

        # Initialize confidence calibrator
        self.confidence_calibrator = get_confidence_calibrator()

        # Initialize genre-specific prompts
        self.genre_prompts = get_genre_prompts()
        self.current_genre = Genre.GENERAL  # Default genre

        # Update thresholds from calibrator
        self._update_thresholds_from_calibrator()

        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Classifier will use fallback methods only.")

    def _update_thresholds_from_calibrator(self):
        """Update internal thresholds from the confidence calibrator."""
        thresholds = self.confidence_calibrator.get_current_thresholds()
        self.fallback_confidence_threshold = thresholds.premium_model_trigger
        logger.debug(f"Updated thresholds from calibrator: premium_trigger={self.fallback_confidence_threshold:.3f}")

    def record_classification_feedback(self,
                                     entity_name: str,
                                     confidence: float,
                                     predicted_type: str,
                                     user_feedback: str = None):
        """Record classification feedback for calibration."""
        self.confidence_calibrator.record_classification(
            confidence=confidence,
            predicted_type=predicted_type,
            user_feedback=user_feedback
        )
        logger.debug(f"Recorded feedback for '{entity_name}': confidence={confidence:.3f}, type={predicted_type}, feedback={user_feedback}")

    def set_genre(self, genre: Genre = None, content: str = None, novel_metadata: Dict[str, Any] = None):
        """Set the genre for classification optimization."""
        if genre:
            self.current_genre = genre
        elif content or novel_metadata:
            # Auto-detect genre
            self.current_genre = self.genre_prompts.detect_genre(content or "", novel_metadata or {})
        else:
            self.current_genre = Genre.GENERAL

        logger.debug(f"Set classification genre to: {self.current_genre.value}")

    def get_current_genre(self) -> Genre:
        """Get the current genre setting."""
        return self.current_genre
    
    def _init_openrouter(self):
        """Initialize OpenRouter client."""
        try:
            self.openrouter_client = openai.OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized for entity type classifier")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None
    
    def is_available(self) -> bool:
        """Check if the classifier is available."""
        return self.openrouter_client is not None
    
    def classify_entity(self,
                       entity_name: str,
                       context: str,
                       novel_id: str = None) -> Dict[str, Any]:
        """
        Classify an entity using LLM with semantic understanding.

        Args:
            entity_name: The entity name to classify
            context: Surrounding context text
            novel_id: Optional novel ID for context

        Returns:
            Classification result with type, confidence, and reasoning
        """
        # Check cache first
        if self.cache_manager:
            cache_key = self.cache_manager._generate_cache_key(
                'entity_classification',
                entity_name=entity_name,
                context=context[:200],  # Limit context for cache key
                novel_id=novel_id
            )

            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result

        if not self.is_available():
            result = self._fallback_classification(entity_name, context)
        else:
            try:
                # First, do basic validation
                basic_validation = self._basic_validation(entity_name)
                if not basic_validation['valid']:
                    result = {
                        'entity_type': None,
                        'confidence': 0.0,
                        'reasoning': basic_validation['reasons'],
                        'recommendation': 'reject',
                        'method': 'basic_validation'
                    }
                else:
                    # Use LLM for semantic classification
                    llm_result = self._llm_classification(entity_name, context)

                    # Combine with rule-based validation
                    result = self._combine_results(entity_name, llm_result, basic_validation)

            except Exception as e:
                logger.error(f"Entity classification failed for '{entity_name}': {e}")
                result = self._fallback_classification(entity_name, context)

        # Cache the result
        if self.cache_manager and result:
            # Cache for 1 hour for successful classifications, 10 minutes for failures
            ttl = 3600.0 if result.get('entity_type') else 600.0
            self.cache_manager.set(cache_key, result, ttl)

        return result

    def classify_entities_batch(self,
                               entities: List[Tuple[str, str]],
                               novel_id: str = None,
                               max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Classify multiple entities in parallel for improved performance.

        Args:
            entities: List of (entity_name, context) tuples
            novel_id: Optional novel ID for context
            max_workers: Maximum number of parallel workers

        Returns:
            List of classification results in the same order as input
        """
        if not entities:
            return []

        if not self.is_available():
            # Fallback to sequential processing with fallback classification
            return [self._fallback_classification(name, context) for name, context in entities]

        results = [None] * len(entities)  # Pre-allocate results list

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all classification tasks
            future_to_index = {}
            for i, (entity_name, context) in enumerate(entities):
                future = executor.submit(self.classify_entity, entity_name, context, novel_id)
                future_to_index[future] = i

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    entity_name, context = entities[index]
                    logger.error(f"Batch classification failed for '{entity_name}': {e}")
                    results[index] = self._fallback_classification(entity_name, context)

        return results

    def classify_entities_batch_optimized(self,
                                        entities: List[Tuple[str, str]],
                                        novel_id: str = None,
                                        batch_size: int = 10,
                                        max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        Classify entities using batch API calls for cost optimization with caching.

        Args:
            entities: List of (entity_name, context) tuples
            novel_id: Optional novel ID for context
            batch_size: Number of entities to classify in a single API call
            max_workers: Maximum number of parallel batch workers

        Returns:
            List of classification results in the same order as input
        """
        if not entities:
            return []

        # Check cache for individual entities first
        results = [None] * len(entities)
        uncached_entities = []
        uncached_indices = []

        if self.cache_manager:
            for i, (entity_name, context) in enumerate(entities):
                cache_key = self.cache_manager._generate_cache_key(
                    'entity_classification',
                    entity_name=entity_name,
                    context=context[:200],
                    novel_id=novel_id
                )

                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    results[i] = cached_result
                else:
                    uncached_entities.append((entity_name, context))
                    uncached_indices.append(i)
        else:
            uncached_entities = entities
            uncached_indices = list(range(len(entities)))

        # Process uncached entities
        if uncached_entities:
            if not self.is_available():
                uncached_results = [self._fallback_classification(name, context) for name, context in uncached_entities]
            else:
                # Split uncached entities into batches
                batches = [uncached_entities[i:i + batch_size] for i in range(0, len(uncached_entities), batch_size)]
                uncached_results = []

                # Process batches in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_batch = {}
                    for batch in batches:
                        future = executor.submit(self._classify_batch_api_call, batch, novel_id)
                        future_to_batch[future] = batch

                    # Collect results maintaining order
                    batch_results = {}
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            batch_result = future.result()
                            batch_results[id(batch)] = batch_result
                        except Exception as e:
                            logger.error(f"Batch API call failed: {e}")
                            # Fallback to individual classification for this batch
                            batch_result = [self._fallback_classification(name, context) for name, context in batch]
                            batch_results[id(batch)] = batch_result

                    # Reconstruct results in original order
                    for batch in batches:
                        uncached_results.extend(batch_results[id(batch)])

            # Cache the new results and fill in the results array
            for i, result in enumerate(uncached_results):
                original_index = uncached_indices[i]
                results[original_index] = result

                # Cache the result
                if self.cache_manager and result:
                    entity_name, context = uncached_entities[i]
                    cache_key = self.cache_manager._generate_cache_key(
                        'entity_classification',
                        entity_name=entity_name,
                        context=context[:200],
                        novel_id=novel_id
                    )
                    ttl = 3600.0 if result.get('entity_type') else 600.0
                    self.cache_manager.set(cache_key, result, ttl)

        return results

    def _basic_validation(self, entity_name: str) -> Dict[str, Any]:
        """Perform basic validation checks."""
        validation_result = {
            'valid': True,
            'reasons': [],
            'score': 0.5
        }
        
        # Length check
        if len(entity_name.strip()) < 2:
            validation_result['valid'] = False
            validation_result['reasons'].append('Entity name too short')
            return validation_result
        
        # Common word check
        common_words = {
            'the', 'and', 'or', 'but', 'with', 'from', 'to', 'of', 'in', 'on', 'at',
            'by', 'for', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that',
            'these', 'those', 'a', 'an', 'all', 'any', 'some', 'no', 'not'
        }
        
        if entity_name.lower().strip() in common_words:
            validation_result['valid'] = False
            validation_result['reasons'].append('Entity is a common English word')
            return validation_result
        
        # Temporal word check
        temporal_words = {
            'annual', 'yearly', 'monthly', 'daily', 'weekly', 'century', 'decade',
            'year', 'month', 'day', 'week', 'season', 'winter', 'summer', 'spring',
            'autumn', 'fall', 'morning', 'evening', 'night', 'today', 'yesterday',
            'tomorrow', 'now', 'then', 'when', 'while', 'during', 'before', 'after'
        }
        
        if entity_name.lower().strip() in temporal_words:
            validation_result['valid'] = False
            validation_result['reasons'].append('Entity appears to be a temporal concept')
            return validation_result
        
        # Simple adjective check
        simple_adjectives = {
            'cold', 'hot', 'warm', 'cool', 'big', 'small', 'large', 'tiny', 'huge',
            'old', 'new', 'young', 'ancient', 'modern', 'recent', 'past', 'future',
            'good', 'bad', 'evil', 'holy', 'dark', 'light', 'bright', 'dim',
            'high', 'low', 'tall', 'short', 'long', 'wide', 'narrow', 'thick',
            'military', 'royal', 'noble', 'sacred', 'blessed', 'cursed'
        }
        
        if entity_name.lower().strip() in simple_adjectives:
            validation_result['valid'] = False
            validation_result['reasons'].append('Entity appears to be a simple adjective')
            return validation_result
        
        return validation_result
    
    def _llm_classification(self, entity_name: str, context: str) -> Dict[str, Any]:
        """Use LLM to classify the entity type with smart fallbacks and rate limiting."""
        if self.enable_smart_fallbacks:
            return self._smart_fallback_classification(entity_name, context)
        else:
            return self._single_model_classification(entity_name, context, self.chat_model)

    def _smart_fallback_classification(self, entity_name: str, context: str) -> Dict[str, Any]:
        """Use smart fallback strategy: cheap model first, premium model if needed."""
        # Step 1: Try cheap model first
        cheap_result = self._single_model_classification(entity_name, context, self.cheap_model)

        # If cheap model is confident enough, use its result
        if (cheap_result.get('confidence', 0.0) >= self.fallback_confidence_threshold and
            cheap_result.get('entity_type') is not None):
            cheap_result['method'] = 'smart_fallback_cheap'
            logger.debug(f"Using cheap model result for '{entity_name}' (confidence: {cheap_result.get('confidence', 0.0):.2f})")
            return cheap_result

        # Step 2: If cheap model is not confident, try premium model
        logger.debug(f"Cheap model not confident enough for '{entity_name}' (confidence: {cheap_result.get('confidence', 0.0):.2f}), trying premium model")
        premium_result = self._single_model_classification(entity_name, context, self.premium_model)

        if premium_result.get('entity_type') is not None:
            premium_result['method'] = 'smart_fallback_premium'
            return premium_result
        else:
            # If both models fail, return the cheap model result with fallback method
            cheap_result['method'] = 'smart_fallback_both_failed'
            return cheap_result

    def _single_model_classification(self, entity_name: str, context: str, model: str) -> Dict[str, Any]:
        """Classify using a single model with rate limiting."""
        system_prompt = self._build_classification_system_prompt()
        user_prompt = self._build_classification_user_prompt(entity_name, context)

        # Acquire rate limit permission
        if not self.rate_limiter.acquire(timeout=30.0):
            logger.warning(f"Rate limit timeout for entity classification: {entity_name}")
            return {
                'entity_type': None,
                'confidence': 0.0,
                'reasoning': ['Rate limit timeout'],
                'recommendation': 'retry',
                'method': 'rate_limit_timeout'
            }

        try:
            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=300  # Reduced from 500 for cost optimization
            )

            ai_response = response.choices[0].message.content
            result = self._parse_llm_response(ai_response)

            # Release rate limiter with success
            self.rate_limiter.release(success=True, response_headers=getattr(response, 'headers', {}))
            return result

        except Exception as e:
            logger.error(f"LLM classification failed with model {model}: {e}")
            # Release rate limiter with failure
            self.rate_limiter.release(success=False)
            return {
                'entity_type': None,
                'confidence': 0.0,
                'reasoning': [str(e)],
                'recommendation': 'review',
                'method': 'llm_error'
            }

    def _classify_batch_api_call(self,
                                entities: List[Tuple[str, str]],
                                novel_id: str = None) -> List[Dict[str, Any]]:
        """
        Classify a batch of entities with smart fallbacks for cost optimization.

        Args:
            entities: List of (entity_name, context) tuples
            novel_id: Optional novel ID for context

        Returns:
            List of classification results
        """
        if not entities:
            return []

        if self.enable_smart_fallbacks:
            return self._smart_fallback_batch_classification(entities)
        else:
            return self._single_model_batch_classification(entities, self.chat_model)

    def _smart_fallback_batch_classification(self, entities: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Use smart fallback strategy for batch classification."""
        # Step 1: Try cheap model for the entire batch
        cheap_results = self._single_model_batch_classification(entities, self.cheap_model)

        # Step 2: Identify entities that need premium model validation
        needs_premium = []
        needs_premium_indices = []
        final_results = cheap_results.copy()

        for i, result in enumerate(cheap_results):
            if (result.get('confidence', 0.0) < self.fallback_confidence_threshold or
                result.get('entity_type') is None):
                needs_premium.append(entities[i])
                needs_premium_indices.append(i)

        # Step 3: Use premium model for uncertain entities
        if needs_premium:
            logger.info(f"Using premium model for {len(needs_premium)} uncertain entities out of {len(entities)}")
            premium_results = self._single_model_batch_classification(needs_premium, self.premium_model)

            # Replace uncertain results with premium results
            for i, premium_result in enumerate(premium_results):
                original_index = needs_premium_indices[i]
                premium_result['method'] = 'smart_fallback_batch_premium'
                final_results[original_index] = premium_result

        # Mark cheap model results
        for i, result in enumerate(final_results):
            if i not in needs_premium_indices:
                result['method'] = 'smart_fallback_batch_cheap'

        return final_results

    def _single_model_batch_classification(self, entities: List[Tuple[str, str]], model: str) -> List[Dict[str, Any]]:
        """Classify a batch using a single model."""
        # Acquire rate limit permission for batch request
        if not self.rate_limiter.acquire(timeout=30.0):
            logger.warning(f"Rate limit timeout for batch classification of {len(entities)} entities")
            return [self._fallback_classification(name, context) for name, context in entities]

        try:
            # Build batch classification prompt
            system_prompt = self._build_batch_classification_system_prompt()
            user_prompt = self._build_batch_classification_user_prompt(entities)

            # Optimize token usage based on batch size
            max_tokens = min(2000, 500 + len(entities) * 30)  # More conservative token usage

            response = self.openrouter_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens
            )

            ai_response = response.choices[0].message.content
            result = self._parse_batch_classification_response(ai_response, entities)

            # Release rate limiter with success
            self.rate_limiter.release(success=True, response_headers=getattr(response, 'headers', {}))
            return result

        except Exception as e:
            logger.error(f"Batch classification API call failed with model {model}: {e}")
            # Release rate limiter with failure
            self.rate_limiter.release(success=False)
            # Fallback to individual classification
            return [self._fallback_classification(name, context) for name, context in entities]

    def _build_batch_classification_system_prompt(self) -> str:
        """Build genre-optimized system prompt for batch classification."""
        return self.genre_prompts.get_batch_system_prompt(self.current_genre)

    def _build_batch_classification_user_prompt(self, entities: List[Tuple[str, str]]) -> str:
        """Build optimized user prompt for batch classification."""
        # Optimized batch user prompt - reduced context length and simplified format
        prompt_parts = ["Entities:"]

        for i, (entity_name, context) in enumerate(entities, 1):
            # Reduced context from 200 to 100 characters to save tokens
            prompt_parts.append(f"{i}. '{entity_name}' - {context[:100]}")

        return "\n".join(prompt_parts)

    def _parse_batch_classification_response(self,
                                           response: str,
                                           entities: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Parse batch classification response."""
        try:
            # Find JSON array in response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                classifications = json.loads(json_str)

                # Validate and format results
                results = []
                for i, classification in enumerate(classifications):
                    if i < len(entities):
                        entity_name, context = entities[i]

                        # Ensure required fields
                        result = {
                            'entity_type': classification.get('entity_type'),
                            'confidence': float(classification.get('confidence', 0.0)),
                            'reasoning': classification.get('reasoning', ''),
                            'recommendation': classification.get('recommendation', 'review'),
                            'method': 'batch_llm_classification'
                        }

                        # Validate entity_type
                        if result['entity_type'] not in ['characters', 'locations', 'lore']:
                            result['entity_type'] = None
                            result['recommendation'] = 'reject'

                        results.append(result)

                # Fill missing results with fallback
                while len(results) < len(entities):
                    entity_name, context = entities[len(results)]
                    results.append(self._fallback_classification(entity_name, context))

                return results

        except Exception as e:
            logger.error(f"Failed to parse batch classification response: {e}")

        # Fallback to individual classification
        return [self._fallback_classification(name, context) for name, context in entities]

    def _build_classification_system_prompt(self) -> str:
        """Build genre-optimized system prompt for entity classification."""
        return self.genre_prompts.get_system_prompt(self.current_genre)
    
    def _build_classification_user_prompt(self, entity_name: str, context: str) -> str:
        """Build optimized, token-efficient user prompt with enhanced context."""
        # Enhanced context processing - extract most relevant parts
        enhanced_context = self._extract_relevant_context_features(entity_name, context)

        return f"""Entity: "{entity_name}"
Context: "{enhanced_context}"

Classify as characters/locations/lore/null with JSON response."""

    def _extract_relevant_context_features(self, entity_name: str, context: str) -> str:
        """Extract the most relevant context features for classification."""
        # Limit context length but prioritize relevant information
        max_length = 200  # Increased from 150 for better accuracy

        if len(context) <= max_length:
            return context

        # Find the entity in context
        entity_lower = entity_name.lower()
        context_lower = context.lower()
        entity_pos = context_lower.find(entity_lower)

        if entity_pos == -1:
            # Entity not found, return beginning of context
            return context[:max_length]

        # Extract context around the entity mention
        start_pos = max(0, entity_pos - max_length // 3)
        end_pos = min(len(context), entity_pos + len(entity_name) + max_length * 2 // 3)

        relevant_context = context[start_pos:end_pos]

        # Add ellipsis if truncated
        if start_pos > 0:
            relevant_context = "..." + relevant_context
        if end_pos < len(context):
            relevant_context = relevant_context + "..."

        return relevant_context
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        try:
            # Find JSON in the response - look for the first complete JSON object
            json_start = response.find('{')
            if json_start == -1:
                logger.warning(f"No JSON found in LLM response: {response}")
                return self._extract_classification_from_text(response)

            # Find the matching closing brace
            brace_count = 0
            json_end = json_start

            for i in range(json_start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if brace_count != 0:
                logger.warning(f"Unmatched braces in JSON response: {response}")
                return self._extract_classification_from_text(response)

            json_str = response[json_start:json_end]
            result = json.loads(json_str)

            # Validate required fields
            if 'entity_type' in result and 'confidence' in result:
                return result
            else:
                logger.warning(f"Missing required fields in JSON: {result}")
                return self._extract_classification_from_text(response)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Attempted to parse: {response[json_start:json_end] if 'json_start' in locals() else 'N/A'}")
            return self._extract_classification_from_text(response)
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            return self._extract_classification_from_text(response)
    
    def _extract_classification_from_text(self, response: str) -> Dict[str, Any]:
        """Extract classification from unstructured text response."""
        response_lower = response.lower()
        
        # Look for entity type mentions
        if 'character' in response_lower:
            entity_type = 'characters'
        elif 'location' in response_lower:
            entity_type = 'locations'
        elif 'lore' in response_lower:
            entity_type = 'lore'
        elif 'reject' in response_lower or 'invalid' in response_lower:
            entity_type = None
        else:
            entity_type = None
        
        # Estimate confidence based on language
        confidence = 0.5
        if 'clearly' in response_lower or 'definitely' in response_lower:
            confidence = 0.8
        elif 'probably' in response_lower or 'likely' in response_lower:
            confidence = 0.7
        elif 'maybe' in response_lower or 'possibly' in response_lower:
            confidence = 0.4
        elif 'reject' in response_lower or 'invalid' in response_lower:
            confidence = 0.1
        
        return {
            'entity_type': entity_type,
            'confidence': confidence,
            'reasoning': [f'Extracted from text: {response[:100]}...'],
            'recommendation': 'review'
        }
    
    def _combine_results(self, entity_name: str, llm_result: Dict[str, Any], basic_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Combine LLM results with basic validation."""
        if not basic_validation['valid']:
            return {
                'entity_type': None,
                'confidence': 0.0,
                'reasoning': basic_validation['reasons'],
                'recommendation': 'reject',
                'method': 'basic_validation'
            }

        # Use rule-based validation to adjust LLM confidence
        if llm_result.get('entity_type'):
            validation = self.entity_definitions.validate_entity_name(
                entity_name, llm_result['entity_type']
            )

            # More conservative confidence adjustment
            llm_confidence = llm_result.get('confidence', 0.5)
            validation_score = validation.get('score', 0.5)

            # Don't penalize too heavily for minor validation issues
            if validation_score > 0.7:
                adjusted_confidence = llm_confidence  # Keep original confidence
            elif validation_score > 0.5:
                adjusted_confidence = llm_confidence * 0.9  # Minor penalty
            else:
                adjusted_confidence = llm_confidence * validation_score  # Larger penalty

            # Determine final recommendation
            if adjusted_confidence > 0.8:
                recommendation = 'accept'
            elif adjusted_confidence > 0.5:
                recommendation = 'review'
            else:
                recommendation = 'reject'

            return {
                'entity_type': llm_result['entity_type'],
                'confidence': adjusted_confidence,
                'reasoning': llm_result.get('reasoning', []) + validation.get('reasons', []),
                'recommendation': recommendation,
                'method': 'llm_with_validation'
            }

        return llm_result
    
    def _fallback_classification(self, entity_name: str, context: str) -> Dict[str, Any]:
        """Fallback classification when LLM is not available."""
        # Use basic rule-based classification
        basic_validation = self._basic_validation(entity_name)
        if not basic_validation['valid']:
            return {
                'entity_type': None,
                'confidence': 0.0,
                'reasoning': basic_validation['reasons'],
                'recommendation': 'reject',
                'method': 'fallback_validation'
            }
        
        # Simple rule-based classification
        context_lower = context.lower()
        name_lower = entity_name.lower()
        
        # Check for character indicators
        character_indicators = ['said', 'spoke', 'he', 'she', 'they', 'king', 'queen', 'wizard']
        character_score = sum(1 for indicator in character_indicators if indicator in context_lower)
        
        # Check for location indicators  
        location_indicators = ['city', 'castle', 'forest', 'mountain', 'in', 'at', 'located']
        location_score = sum(1 for indicator in location_indicators if indicator in context_lower)
        
        # Check for lore indicators
        lore_indicators = ['magic', 'spell', 'legend', 'war', 'battle', 'organization']
        lore_score = sum(1 for indicator in lore_indicators if indicator in context_lower)
        
        scores = {
            'characters': character_score,
            'locations': location_score, 
            'lore': lore_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            entity_type = max(scores, key=scores.get)
            confidence = min(0.6, max_score * 0.2)  # Conservative confidence
        else:
            entity_type = 'characters'  # Default fallback
            confidence = 0.3
        
        return {
            'entity_type': entity_type,
            'confidence': confidence,
            'reasoning': [f'Fallback classification based on context indicators'],
            'recommendation': 'review',
            'method': 'fallback_rules'
        }
    
    def batch_classify(self, entities: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Classify multiple entities in batch for efficiency."""
        results = []
        for entity_name, context in entities:
            result = self.classify_entity(entity_name, context)
            results.append(result)
        return results
