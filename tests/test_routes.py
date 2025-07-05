"""
Route Functionality Tests for Lazywriter

This module tests all Flask routes to ensure they respond correctly
and handle various scenarios including error conditions.
"""

import pytest
import os
import sys
import tempfile
import shutil
import uuid
import json
from unittest.mock import patch, Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="lazywriter_routes_test_")
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
        from app import app
        app.config.update(app_config)
        with app.app_context():
            yield app


@pytest.fixture
def client(flask_app):
    """Create test client."""
    return flask_app.test_client()


@pytest.fixture
def sample_novel_id():
    """Generate a sample novel ID."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_character_id():
    """Generate a sample character ID."""
    return str(uuid.uuid4())


class TestNovelRoutes:
    """Test novel-related routes."""
    
    def test_home_route_displays_novels(self, client):
        """Test home route displays novels list."""
        response = client.get('/')
        assert response.status_code == 200
        # Should contain some indication of novels section
        assert b'novel' in response.data.lower() or b'Novel' in response.data
    
    def test_create_novel_get(self, client):
        """Test create novel GET route."""
        response = client.get('/create_novel')
        assert response.status_code == 200
        assert b'form' in response.data.lower()
    
    def test_create_novel_post_valid_data(self, client):
        """Test create novel POST with valid data."""
        novel_data = {
            'title': 'Test Novel',
            'description': 'A test novel description',
            'genre': 'Fantasy'
        }
        
        response = client.post('/create_novel', data=novel_data, follow_redirects=True)
        assert response.status_code == 200
    
    def test_create_novel_post_missing_data(self, client):
        """Test create novel POST with missing data."""
        novel_data = {
            'title': '',  # Missing title
            'description': 'A test novel description',
            'genre': 'Fantasy'
        }
        
        response = client.post('/create_novel', data=novel_data)
        # Should handle missing data gracefully
        assert response.status_code in [200, 400, 422]


class TestWorldbuildingRoutes:
    """Test worldbuilding-related routes."""
    
    def test_novel_worldbuilding_route(self, client, sample_novel_id):
        """Test novel worldbuilding main page."""
        url = f'/novel/{sample_novel_id}/worldbuilding'
        response = client.get(url)
        # Should either show worldbuilding page or redirect if novel doesn't exist
        assert response.status_code in [200, 302, 404]
    
    def test_novel_characters_route(self, client, sample_novel_id):
        """Test novel characters page."""
        url = f'/novel/{sample_novel_id}/worldbuilding/characters'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]
    
    def test_novel_locations_route(self, client, sample_novel_id):
        """Test novel locations page."""
        url = f'/novel/{sample_novel_id}/worldbuilding/locations'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]
    
    def test_novel_lore_route(self, client, sample_novel_id):
        """Test novel lore page."""
        url = f'/novel/{sample_novel_id}/worldbuilding/lore'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]
    
    def test_novel_search_route(self, client, sample_novel_id):
        """Test novel search page."""
        url = f'/novel/{sample_novel_id}/worldbuilding/search'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]


class TestCharacterRoutes:
    """Test character-related routes."""
    
    def test_create_character_get(self, client, sample_novel_id):
        """Test create character GET route."""
        url = f'/novel/{sample_novel_id}/worldbuilding/characters/create'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]
    
    def test_character_detail_route(self, client, sample_novel_id, sample_character_id):
        """Test character detail route."""
        url = f'/novel/{sample_novel_id}/worldbuilding/characters/{sample_character_id}'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]
    
    def test_edit_character_route(self, client, sample_novel_id, sample_character_id):
        """Test edit character route."""
        url = f'/novel/{sample_novel_id}/worldbuilding/characters/{sample_character_id}/edit'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]


class TestLocationRoutes:
    """Test location-related routes."""
    
    def test_create_location_get(self, client, sample_novel_id):
        """Test create location GET route."""
        url = f'/novel/{sample_novel_id}/worldbuilding/locations/create'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]
    
    def test_location_detail_route(self, client, sample_novel_id, sample_character_id):
        """Test location detail route."""
        url = f'/novel/{sample_novel_id}/worldbuilding/locations/{sample_character_id}'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]


class TestLoreRoutes:
    """Test lore-related routes."""
    
    def test_create_lore_get(self, client, sample_novel_id):
        """Test create lore GET route."""
        url = f'/novel/{sample_novel_id}/worldbuilding/lore/create'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]
    
    def test_lore_detail_route(self, client, sample_novel_id, sample_character_id):
        """Test lore detail route."""
        url = f'/novel/{sample_novel_id}/worldbuilding/lore/{sample_character_id}'
        response = client.get(url)
        assert response.status_code in [200, 302, 404]


class TestAPIRoutes:
    """Test API routes and AJAX endpoints."""
    
    def test_character_create_api(self, client, sample_novel_id):
        """Test character creation API endpoint."""
        url = f'/novel/{sample_novel_id}/worldbuilding/characters/create'
        
        character_data = {
            'prompt': 'A brave knight with a mysterious past'
        }
        
        response = client.post(url, data=character_data)
        # Should handle API request appropriately
        assert response.status_code in [200, 302, 400, 404]
    
    def test_search_api(self, client, sample_novel_id):
        """Test search API endpoint."""
        url = f'/novel/{sample_novel_id}/worldbuilding/search'
        
        search_data = {
            'query': 'test search',
            'entity_type': 'all'
        }
        
        response = client.post(url, data=search_data)
        assert response.status_code in [200, 302, 400, 404]


class TestCrossReferenceRoutes:
    """Test cross-reference functionality routes."""
    
    def test_cross_reference_analysis(self, client, sample_novel_id, sample_character_id):
        """Test cross-reference analysis endpoint."""
        url = f'/novel/{sample_novel_id}/worldbuilding/characters/{sample_character_id}/cross-reference'
        
        response = client.post(url)
        # Should handle cross-reference request
        assert response.status_code in [200, 302, 400, 404]


class TestErrorHandling:
    """Test error handling in routes."""
    
    def test_invalid_novel_id(self, client):
        """Test routes with invalid novel ID."""
        invalid_id = 'invalid-uuid'
        url = f'/novel/{invalid_id}/worldbuilding'
        
        response = client.get(url)
        # Should handle invalid UUID gracefully
        assert response.status_code in [400, 404]
    
    def test_nonexistent_novel(self, client):
        """Test routes with nonexistent novel."""
        nonexistent_id = str(uuid.uuid4())
        url = f'/novel/{nonexistent_id}/worldbuilding'
        
        response = client.get(url)
        # Should handle nonexistent novel appropriately
        assert response.status_code in [302, 404]
    
    def test_malformed_requests(self, client, sample_novel_id):
        """Test malformed requests."""
        url = f'/novel/{sample_novel_id}/worldbuilding/characters/create'
        
        # Test with malformed JSON
        response = client.post(url, 
                             data='invalid json',
                             content_type='application/json')
        
        # Should handle malformed data gracefully
        assert response.status_code in [200, 400, 422]


class TestRoutePerformance:
    """Test route performance and responsiveness."""
    
    def test_route_response_times(self, client):
        """Test that routes respond within reasonable time."""
        import time
        
        start_time = time.time()
        response = client.get('/')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Route should respond within 5 seconds
        assert response_time < 5.0
        assert response.status_code == 200
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get('/')
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
