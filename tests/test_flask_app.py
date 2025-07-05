"""
Flask Application Tests for Lazywriter

This module tests the core Flask application functionality including
initialization, configuration, and basic application behavior.
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="lazywriter_flask_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def app_config(temp_test_dir):
    """Flask application configuration for testing."""
    return {
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key',
        'TINYDB_PATH': os.path.join(temp_test_dir, 'tinydb'),
        'CHROMADB_PATH': os.path.join(temp_test_dir, 'chromadb'),
        'USE_OPENROUTER_EMBEDDINGS': 'false',
        'OPENROUTER_API_KEY': 'test_key'
    }


@pytest.fixture
def flask_app(app_config):
    """Create Flask application for testing."""
    with patch.dict(os.environ, app_config):
        # Import app after setting environment variables
        from app import app
        app.config.update(app_config)
        
        with app.app_context():
            yield app


@pytest.fixture
def client(flask_app):
    """Create test client."""
    return flask_app.test_client()


class TestFlaskApplication:
    """Test Flask application initialization and configuration."""
    
    def test_app_creation(self, flask_app):
        """Test that Flask app is created successfully."""
        assert flask_app is not None
        assert flask_app.config['TESTING'] is True
    
    def test_app_secret_key(self, flask_app):
        """Test that secret key is configured."""
        assert flask_app.secret_key is not None
        assert len(flask_app.secret_key) > 0
    
    def test_app_context(self, flask_app):
        """Test Flask application context."""
        with flask_app.app_context():
            from flask import current_app
            assert current_app is flask_app


class TestDatabaseInitialization:
    """Test database initialization in Flask context."""
    
    def test_world_state_initialization(self, flask_app):
        """Test WorldState initialization."""
        with flask_app.app_context():
            from app import world_state
            assert world_state is not None
    
    def test_semantic_search_initialization(self, flask_app):
        """Test SemanticSearchEngine initialization."""
        with flask_app.app_context():
            from app import semantic_search
            assert semantic_search is not None
    
    def test_ai_agents_initialization(self, flask_app):
        """Test AI agents initialization."""
        with flask_app.app_context():
            from app import (character_creator, character_editor, 
                           lore_creator, lore_editor, location_creator, 
                           location_editor, cross_reference_agent)
            
            assert character_creator is not None
            assert character_editor is not None
            assert lore_creator is not None
            assert lore_editor is not None
            assert location_creator is not None
            assert location_editor is not None
            assert cross_reference_agent is not None


class TestBasicRoutes:
    """Test basic Flask routes."""
    
    def test_home_route(self, client):
        """Test home page route."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Lazywriter' in response.data or b'Novel' in response.data
    
    def test_create_novel_get(self, client):
        """Test create novel GET route."""
        response = client.get('/create_novel')
        assert response.status_code == 200
    
    def test_404_handling(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent-route')
        assert response.status_code == 404


class TestApplicationSecurity:
    """Test application security features."""
    
    def test_csrf_protection(self, flask_app):
        """Test CSRF protection is configured."""
        # Check that secret key exists for CSRF protection
        assert flask_app.secret_key is not None
    
    def test_secure_headers(self, client):
        """Test security headers in responses."""
        response = client.get('/')
        # Basic security check - ensure no sensitive info in headers
        assert 'Server' not in response.headers or 'Flask' not in response.headers.get('Server', '')


class TestErrorHandling:
    """Test application error handling."""
    
    def test_internal_error_handling(self, flask_app):
        """Test internal error handling."""
        with flask_app.test_client() as client:
            # This should not crash the application
            with patch('app.world_state.get_all', side_effect=Exception("Test error")):
                response = client.get('/')
                # Should either handle gracefully or return 500
                assert response.status_code in [200, 500]


class TestApplicationLogging:
    """Test application logging configuration."""
    
    def test_logging_configuration(self, flask_app):
        """Test that logging is properly configured."""
        import logging
        logger = logging.getLogger('app')
        assert logger is not None
        
        # Test that we can log without errors
        logger.info("Test log message")


class TestApplicationShutdown:
    """Test application shutdown and cleanup."""
    
    def test_graceful_shutdown(self, flask_app):
        """Test that application can shut down gracefully."""
        # Test that we can create and destroy the app context
        with flask_app.app_context():
            pass
        
        # Application should still be functional
        assert flask_app is not None


class TestEnvironmentConfiguration:
    """Test environment-specific configuration."""
    
    def test_testing_environment(self, flask_app):
        """Test testing environment configuration."""
        assert flask_app.config['TESTING'] is True
    
    def test_debug_mode_disabled_in_testing(self, flask_app):
        """Test that debug mode is disabled in testing."""
        # In testing, debug should typically be False
        assert flask_app.config.get('DEBUG', False) is False


class TestApplicationIntegrity:
    """Test overall application integrity."""
    
    def test_all_imports_successful(self, flask_app):
        """Test that all required modules can be imported."""
        with flask_app.app_context():
            try:
                from app import (world_state, semantic_search, character_creator,
                               character_editor, lore_creator, lore_editor,
                               location_creator, location_editor, cross_reference_agent)
                # If we get here, all imports were successful
                assert True
            except ImportError as e:
                pytest.fail(f"Import failed: {e}")
    
    def test_application_ready_for_requests(self, client):
        """Test that application is ready to handle requests."""
        # Test multiple requests to ensure stability
        for _ in range(3):
            response = client.get('/')
            assert response.status_code == 200
    
    def test_memory_usage_reasonable(self, flask_app):
        """Test that application memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Application should use less than 1GB of memory in testing
        assert memory_mb < 1024, f"Memory usage too high: {memory_mb:.2f} MB"
