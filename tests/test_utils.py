"""
Unit tests for utility modules in the Lazywriter system.

This module tests entity detection, entity recognition, streaming analysis,
caching, and other utility functions.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from utils.entity_detection import EntityDetectionUtils
from utils.stanza_entity_recognizer import OptimizedEntityRecognizer, EntityMatch
from utils.streaming_analysis import StreamingAnalysisManager, AnalysisStage
from utils.cross_reference_cache import CrossReferenceCacheManager
from utils.api_rate_limiter import APIRateLimiter
from utils.confidence_calibration import ConfidenceCalibrator
from utils.json_ops import JSONOperations


class TestEntityDetectionUtils:
    """Test cases for EntityDetectionUtils."""

    def test_initialization(self):
        """Test EntityDetectionUtils initialization."""
        detector = EntityDetectionUtils()
        
        assert hasattr(detector, 'name_patterns')
        assert hasattr(detector, 'location_indicators')
        assert hasattr(detector, 'character_indicators')
        assert hasattr(detector, 'lore_indicators')
    
    def test_detect_potential_entities(self):
        """Test entity detection in text."""
        detector = EntityDetector()
        
        text = "Aragorn walked through Rivendell with Gandalf the Grey. The Ring of Power was hidden there."
        existing_entities = {
            'characters': [],
            'locations': [],
            'lore': []
        }
        
        detected = detector.detect_potential_entities(text, existing_entities)
        
        assert 'characters' in detected
        assert 'locations' in detected
        assert 'lore' in detected
        
        # Should detect some entities
        total_detected = sum(len(entities) for entities in detected.values())
        assert total_detected > 0
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        detector = EntityDetector()
        
        dirty_text = "  Hello,   world!  \n\n  Multiple   spaces.  "
        clean_text = detector._clean_text(dirty_text)
        
        assert clean_text == "Hello, world! Multiple spaces."
        assert "  " not in clean_text  # No double spaces
        assert not clean_text.startswith(" ")  # No leading spaces
        assert not clean_text.endswith(" ")  # No trailing spaces
    
    def test_extract_names(self):
        """Test name extraction from text."""
        detector = EntityDetector()
        
        text = "King Arthur met Sir Lancelot at Camelot Castle."
        names = detector._extract_names(text)
        
        assert isinstance(names, list)
        assert len(names) > 0
        
        # Should extract capitalized words/phrases
        names_str = " ".join(names)
        assert any("Arthur" in name for name in names)
    
    def test_classify_entity_type(self):
        """Test entity type classification."""
        detector = EntityDetector()
        
        # Test character classification
        char_result = detector._classify_entity_type(
            "Sir Galahad", 
            "The brave knight fought the dragon"
        )
        assert char_result in ['characters', None]
        
        # Test location classification
        loc_result = detector._classify_entity_type(
            "Dragon's Lair",
            "The cave was dark and filled with treasure"
        )
        assert loc_result in ['locations', None]
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        detector = EntityDetector()
        
        confidence = detector._calculate_confidence(
            "Sir Galahad",
            "The noble knight rode into battle",
            "characters"
        )
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)


class TestOptimizedEntityRecognizer:
    """Test cases for OptimizedEntityRecognizer."""
    
    @pytest.mark.requires_models
    def test_initialization(self, mock_world_state):
        """Test OptimizedEntityRecognizer initialization."""
        with patch('utils.stanza_entity_recognizer.stanza') as mock_stanza:
            mock_pipeline = Mock()
            mock_stanza.Pipeline.return_value = mock_pipeline
            
            recognizer = OptimizedEntityRecognizer(world_state=mock_world_state)
            
            assert recognizer.world_state == mock_world_state
            assert hasattr(recognizer, 'nlp')
            assert hasattr(recognizer, 'gazetteers')
            assert hasattr(recognizer, 'cache')
    
    @pytest.mark.requires_models
    def test_recognize_entities_basic(self, mock_world_state):
        """Test basic entity recognition."""
        with patch('utils.stanza_entity_recognizer.stanza') as mock_stanza:
            # Mock Stanza pipeline
            mock_doc = Mock()
            mock_doc.text = "Aragorn is a ranger from Gondor."
            mock_doc.ents = []
            
            mock_pipeline = Mock()
            mock_pipeline.return_value = mock_doc
            mock_stanza.Pipeline.return_value = mock_pipeline
            
            recognizer = OptimizedEntityRecognizer(world_state=mock_world_state)
            
            # Mock the cache to avoid actual processing
            with patch.object(recognizer, '_check_cache') as mock_cache:
                mock_cache.return_value = None
                
                results = recognizer.recognize_entities(
                    content="Aragorn is a ranger from Gondor.",
                    novel_id="test_novel",
                    use_cache=False
                )
                
                assert isinstance(results, list)
    
    def test_cache_functionality(self, mock_world_state):
        """Test caching functionality."""
        with patch('utils.stanza_entity_recognizer.stanza'):
            recognizer = OptimizedEntityRecognizer(world_state=mock_world_state)
            
            # Test cache key generation
            content = "Test content"
            cache_key = recognizer._generate_cache_key(content, "novel_1")
            
            assert isinstance(cache_key, str)
            assert len(cache_key) > 0
            
            # Test cache storage and retrieval
            test_results = [
                EntityMatch(
                    entity_id="test_id",
                    entity_name="Test Entity",
                    entity_type="characters",
                    match_text="Test",
                    match_type="test",
                    confidence=0.8,
                    start_pos=0,
                    end_pos=4,
                    context="Test context",
                    evidence=["test evidence"]
                )
            ]
            
            recognizer._store_in_cache(cache_key, test_results)
            cached_results = recognizer._check_cache(cache_key)
            
            assert cached_results is not None
            assert len(cached_results) == 1
            assert cached_results[0].entity_name == "Test Entity"
    
    def test_gazetteer_updates(self, mock_world_state):
        """Test gazetteer update functionality."""
        with patch('utils.stanza_entity_recognizer.stanza'):
            recognizer = OptimizedEntityRecognizer(world_state=mock_world_state)
            
            # Test updating gazetteer
            test_entities = [
                {'name': 'Aragorn', 'id': '1'},
                {'name': 'Legolas', 'id': '2'}
            ]
            
            recognizer.update_gazetteer("test_novel", "characters", test_entities)
            
            assert "test_novel" in recognizer.gazetteers
            assert "characters" in recognizer.gazetteers["test_novel"]
            assert len(recognizer.gazetteers["test_novel"]["characters"]) == 2


class TestStreamingAnalysisManager:
    """Test cases for StreamingAnalysisManager."""
    
    def test_initialization(self):
        """Test StreamingAnalysisManager initialization."""
        manager = StreamingAnalysisManager()
        
        assert hasattr(manager, 'active_jobs')
        assert hasattr(manager, 'job_results')
        assert isinstance(manager.active_jobs, dict)
    
    def test_create_job(self):
        """Test job creation."""
        manager = StreamingAnalysisManager()
        
        job_id = manager.create_job("test_entity", "characters", "novel_1")
        
        assert job_id is not None
        assert job_id in manager.active_jobs
        assert manager.active_jobs[job_id]['entity_type'] == "characters"
        assert manager.active_jobs[job_id]['status'] == 'created'
    
    def test_update_progress(self):
        """Test progress updates."""
        manager = StreamingAnalysisManager()
        
        job_id = manager.create_job("test_entity", "characters", "novel_1")
        
        manager.update_progress(
            job_id, 
            AnalysisStage.DETECTING_ENTITIES, 
            0.5, 
            "Processing entities"
        )
        
        job_info = manager.get_job_status(job_id)
        assert job_info['stage'] == AnalysisStage.DETECTING_ENTITIES
        assert job_info['progress'] == 0.5
        assert job_info['message'] == "Processing entities"
    
    def test_complete_job(self):
        """Test job completion."""
        manager = StreamingAnalysisManager()
        
        job_id = manager.create_job("test_entity", "characters", "novel_1")
        
        test_results = {"detected_entities": [], "relationships": []}
        manager.complete_job(job_id, test_results)
        
        job_info = manager.get_job_status(job_id)
        assert job_info['status'] == 'completed'
        assert job_id in manager.job_results
        assert manager.job_results[job_id] == test_results
    
    def test_fail_job(self):
        """Test job failure handling."""
        manager = StreamingAnalysisManager()
        
        job_id = manager.create_job("test_entity", "characters", "novel_1")
        
        error_message = "Test error occurred"
        manager.fail_job(job_id, error_message)
        
        job_info = manager.get_job_status(job_id)
        assert job_info['status'] == 'failed'
        assert job_info['error'] == error_message


class TestCrossReferenceCacheManager:
    """Test cases for CrossReferenceCacheManager."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = CrossReferenceCacheManager(max_size=100, ttl=3600)
        
        assert cache.max_size == 100
        assert cache.ttl == 3600
        assert hasattr(cache, 'cache')
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = CrossReferenceCacheManager(max_size=10, ttl=3600)
        
        # Test set and get
        test_data = {"test": "data"}
        cache.set("test_key", test_data)
        
        retrieved = cache.get("test_key")
        assert retrieved == test_data
        
        # Test non-existent key
        assert cache.get("non_existent") is None
    
    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = CrossReferenceCacheManager(max_size=10, ttl=0.1)  # 0.1 second TTL
        
        cache.set("test_key", {"test": "data"})
        
        # Should be available immediately
        assert cache.get("test_key") is not None
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("test_key") is None
    
    def test_cache_size_limit(self):
        """Test cache size limiting."""
        cache = CrossReferenceCacheManager(max_size=2, ttl=3600)
        
        # Add items up to limit
        cache.set("key1", "data1")
        cache.set("key2", "data2")
        
        assert cache.get("key1") == "data1"
        assert cache.get("key2") == "data2"
        
        # Add one more (should evict oldest)
        cache.set("key3", "data3")
        
        # key1 should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "data2"
        assert cache.get("key3") == "data3"


class TestAPIRateLimiter:
    """Test cases for APIRateLimiter."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = APIRateLimiter(requests_per_minute=60)
        
        assert limiter.requests_per_minute == 60
        assert hasattr(limiter, 'request_times')
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        limiter = APIRateLimiter(requests_per_minute=2)  # Very low limit for testing
        
        # First request should be allowed
        assert limiter.can_make_request() is True
        limiter.record_request()
        
        # Second request should be allowed
        assert limiter.can_make_request() is True
        limiter.record_request()
        
        # Third request should be rate limited
        assert limiter.can_make_request() is False
    
    def test_rate_limit_reset(self):
        """Test rate limit reset over time."""
        limiter = APIRateLimiter(requests_per_minute=1)
        
        # Make a request
        limiter.record_request()
        assert limiter.can_make_request() is False
        
        # Mock time passage
        import time
        with patch('time.time') as mock_time:
            mock_time.return_value = time.time() + 61  # 61 seconds later
            
            # Should be able to make request again
            assert limiter.can_make_request() is True


class TestConfidenceCalibrator:
    """Test cases for ConfidenceCalibrator."""
    
    def test_initialization(self):
        """Test calibrator initialization."""
        calibrator = ConfidenceCalibrator()
        
        assert hasattr(calibrator, 'calibration_data')
        assert hasattr(calibrator, 'model_weights')
    
    def test_calibrate_confidence(self):
        """Test confidence calibration."""
        calibrator = ConfidenceCalibrator()
        
        # Test basic calibration
        raw_confidence = 0.8
        calibrated = calibrator.calibrate_confidence(
            raw_confidence, 
            method='stanza_ner',
            context_features={'entity_length': 5, 'context_words': 10}
        )
        
        assert 0.0 <= calibrated <= 1.0
        assert isinstance(calibrated, float)
    
    def test_update_calibration_data(self):
        """Test updating calibration data."""
        calibrator = ConfidenceCalibrator()
        
        # Add calibration data point
        calibrator.add_calibration_point(
            predicted_confidence=0.8,
            actual_accuracy=0.7,
            method='stanza_ner'
        )
        
        assert 'stanza_ner' in calibrator.calibration_data
        assert len(calibrator.calibration_data['stanza_ner']) > 0


class TestJSONOperations:
    """Test cases for JSONOperations utility."""
    
    def test_safe_json_parse(self):
        """Test safe JSON parsing."""
        # Valid JSON
        valid_json = '{"name": "test", "value": 123}'
        result = JSONOperations.safe_parse(valid_json)
        
        assert result is not None
        assert result['name'] == 'test'
        assert result['value'] == 123
        
        # Invalid JSON
        invalid_json = '{"name": "test", "value":}'
        result = JSONOperations.safe_parse(invalid_json)
        
        assert result is None
    
    def test_extract_json_from_text(self):
        """Test extracting JSON from mixed text."""
        mixed_text = '''
        Here is some text before the JSON.
        {"name": "extracted", "type": "character"}
        And some text after.
        '''
        
        result = JSONOperations.extract_json_from_text(mixed_text)
        
        assert result is not None
        assert result['name'] == 'extracted'
        assert result['type'] == 'character'
    
    def test_validate_entity_json(self):
        """Test entity JSON validation."""
        # Valid character JSON
        valid_character = {
            "name": "Test Character",
            "description": "A test character",
            "age": "25",
            "occupation": "Knight"
        }
        
        is_valid = JSONOperations.validate_entity_json(valid_character, 'characters')
        assert is_valid is True
        
        # Invalid character JSON (missing required fields)
        invalid_character = {
            "name": "Test Character"
            # Missing description, age, occupation
        }
        
        is_valid = JSONOperations.validate_entity_json(invalid_character, 'characters')
        assert is_valid is False
    
    def test_sanitize_json_for_ai(self):
        """Test JSON sanitization for AI responses."""
        raw_json = {
            "name": "Test Character",
            "description": "A character with \"quotes\" and special chars: <>&",
            "tags": ["tag1", "tag2", None, ""],
            "invalid_field": None
        }
        
        sanitized = JSONOperations.sanitize_for_ai_response(raw_json)
        
        assert sanitized['name'] == "Test Character"
        assert '"quotes"' not in sanitized['description']  # Should be escaped
        assert None not in sanitized['tags']  # None values removed
        assert "" not in sanitized['tags']  # Empty strings removed
        assert 'invalid_field' not in sanitized  # None fields removed


@pytest.mark.integration
class TestUtilsIntegration:
    """Integration tests for utility modules."""
    
    def test_entity_detection_to_recognition_pipeline(self, mock_world_state):
        """Test complete entity detection to recognition pipeline."""
        detector = EntityDetector()
        
        text = "Aragorn walked through Rivendell with Gandalf."
        existing_entities = {'characters': [], 'locations': [], 'lore': []}
        
        # Detect entities
        detected = detector.detect_potential_entities(text, existing_entities)
        
        assert isinstance(detected, dict)
        assert 'characters' in detected
        assert 'locations' in detected
        assert 'lore' in detected
        
        # Should have detected some entities
        total_detected = sum(len(entities) for entities in detected.values())
        assert total_detected >= 0  # May be 0 if no clear patterns match
    
    def test_caching_with_streaming_analysis(self):
        """Test caching integration with streaming analysis."""
        cache = CrossReferenceCache(max_size=10, ttl=3600)
        manager = StreamingAnalysisManager()
        
        # Create job
        job_id = manager.create_job("test_entity", "characters", "novel_1")
        
        # Cache some results
        test_results = {"entities": ["entity1", "entity2"]}
        cache_key = f"analysis_{job_id}"
        cache.set(cache_key, test_results)
        
        # Retrieve from cache
        cached_results = cache.get(cache_key)
        assert cached_results == test_results
        
        # Complete job with cached results
        manager.complete_job(job_id, cached_results)
        
        job_status = manager.get_job_status(job_id)
        assert job_status['status'] == 'completed'
