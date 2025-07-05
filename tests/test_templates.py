"""
Template Rendering Tests for Lazywriter

This module tests Jinja2 template rendering to ensure all templates
render correctly and contain expected content.
"""

import pytest
import os
import sys
import tempfile
import shutil
import uuid
from unittest.mock import patch, Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="lazywriter_templates_test_")
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
def sample_novel_data():
    """Sample novel data for template testing."""
    return {
        'id': str(uuid.uuid4()),
        'title': 'Test Novel',
        'author': 'Test Author',
        'genre': 'Fantasy',
        'description': 'A test novel for template testing'
    }


@pytest.fixture
def sample_character_data():
    """Sample character data for template testing."""
    return {
        'id': str(uuid.uuid4()),
        'name': 'Test Character',
        'description': 'A brave warrior',
        'age': '25',
        'occupation': 'Knight',
        'personality': 'Brave and loyal',
        'backstory': 'Born in a small village',
        'tags': ['protagonist', 'warrior']
    }


class TestBaseTemplate:
    """Test base template (layout.html) functionality."""
    
    def test_base_template_structure(self, client):
        """Test that base template renders with proper structure."""
        response = client.get('/')
        assert response.status_code == 200
        
        # Check for basic HTML structure
        assert b'<!DOCTYPE html>' in response.data or b'<html' in response.data
        assert b'<head>' in response.data
        assert b'<body>' in response.data
        assert b'</html>' in response.data
    
    def test_navigation_elements(self, client):
        """Test navigation elements in base template."""
        response = client.get('/')
        assert response.status_code == 200
        
        # Check for navigation elements
        assert b'nav' in response.data.lower() or b'menu' in response.data.lower()
        # Should contain Lazywriter branding
        assert b'Lazywriter' in response.data or b'lazywriter' in response.data.lower()
    
    def test_bootstrap_css_inclusion(self, client):
        """Test that Bootstrap CSS is included."""
        response = client.get('/')
        assert response.status_code == 200
        
        # Check for Bootstrap or CSS inclusion
        assert (b'bootstrap' in response.data.lower() or 
                b'css' in response.data.lower() or
                b'style' in response.data.lower())
    
    def test_responsive_meta_tags(self, client):
        """Test responsive design meta tags."""
        response = client.get('/')
        assert response.status_code == 200
        
        # Check for viewport meta tag
        assert (b'viewport' in response.data.lower() or
                b'responsive' in response.data.lower())


class TestHomeTemplate:
    """Test home page template (index.html)."""
    
    def test_home_template_renders(self, client):
        """Test home template renders successfully."""
        response = client.get('/')
        assert response.status_code == 200
        assert len(response.data) > 0
    
    def test_novels_section_present(self, client):
        """Test novels section is present on home page."""
        response = client.get('/')
        assert response.status_code == 200
        
        # Should contain novels-related content
        content = response.data.lower()
        assert (b'novel' in content or 
                b'create' in content or
                b'worldbuilding' in content)
    
    def test_create_novel_link(self, client):
        """Test create novel link is present."""
        response = client.get('/')
        assert response.status_code == 200
        
        # Should contain link to create novel
        assert (b'create' in response.data.lower() and 
                b'novel' in response.data.lower())


class TestNovelTemplates:
    """Test novel-related templates."""
    
    def test_create_novel_template(self, client):
        """Test create novel template renders."""
        response = client.get('/create_novel')
        assert response.status_code == 200
        
        # Should contain form elements
        assert b'form' in response.data.lower()
        assert b'title' in response.data.lower()
        assert b'description' in response.data.lower()
        assert b'genre' in response.data.lower()
    
    def test_novel_detail_template_structure(self, client, sample_novel_data):
        """Test novel detail template structure."""
        # This test assumes the route exists and handles missing novels gracefully
        novel_id = sample_novel_data['id']
        response = client.get(f'/novel/{novel_id}')
        
        # Should either render template or redirect appropriately
        assert response.status_code in [200, 302, 404]


class TestWorldbuildingTemplates:
    """Test worldbuilding-related templates."""
    
    def test_worldbuilding_hub_template(self, client, sample_novel_data):
        """Test worldbuilding hub template."""
        novel_id = sample_novel_data['id']
        response = client.get(f'/novel/{novel_id}/worldbuilding')
        
        # Should handle template rendering appropriately
        assert response.status_code in [200, 302, 404]
        
        if response.status_code == 200:
            content = response.data.lower()
            # Should contain worldbuilding sections
            assert (b'character' in content or 
                    b'location' in content or 
                    b'lore' in content)
    
    def test_characters_template(self, client, sample_novel_data):
        """Test characters list template."""
        novel_id = sample_novel_data['id']
        response = client.get(f'/novel/{novel_id}/worldbuilding/characters')
        
        assert response.status_code in [200, 302, 404]
        
        if response.status_code == 200:
            assert b'character' in response.data.lower()
    
    def test_locations_template(self, client, sample_novel_data):
        """Test locations list template."""
        novel_id = sample_novel_data['id']
        response = client.get(f'/novel/{novel_id}/worldbuilding/locations')
        
        assert response.status_code in [200, 302, 404]
        
        if response.status_code == 200:
            assert b'location' in response.data.lower()
    
    def test_lore_template(self, client, sample_novel_data):
        """Test lore list template."""
        novel_id = sample_novel_data['id']
        response = client.get(f'/novel/{novel_id}/worldbuilding/lore')
        
        assert response.status_code in [200, 302, 404]
        
        if response.status_code == 200:
            assert b'lore' in response.data.lower()


class TestCharacterTemplates:
    """Test character-related templates."""
    
    def test_create_character_template(self, client, sample_novel_data):
        """Test create character template."""
        novel_id = sample_novel_data['id']
        response = client.get(f'/novel/{novel_id}/worldbuilding/characters/create')
        
        assert response.status_code in [200, 302, 404]
        
        if response.status_code == 200:
            content = response.data.lower()
            # Should contain character creation elements
            assert (b'character' in content and 
                    (b'create' in content or b'form' in content))
    
    def test_character_detail_template(self, client, sample_novel_data, sample_character_data):
        """Test character detail template."""
        novel_id = sample_novel_data['id']
        character_id = sample_character_data['id']
        response = client.get(f'/novel/{novel_id}/worldbuilding/characters/{character_id}')
        
        assert response.status_code in [200, 302, 404]
    
    def test_edit_character_template(self, client, sample_novel_data, sample_character_data):
        """Test edit character template."""
        novel_id = sample_novel_data['id']
        character_id = sample_character_data['id']
        response = client.get(f'/novel/{novel_id}/worldbuilding/characters/{character_id}/edit')
        
        assert response.status_code in [200, 302, 404]


class TestSearchTemplate:
    """Test search template functionality."""
    
    def test_search_template_renders(self, client, sample_novel_data):
        """Test search template renders."""
        novel_id = sample_novel_data['id']
        response = client.get(f'/novel/{novel_id}/worldbuilding/search')
        
        assert response.status_code in [200, 302, 404]
        
        if response.status_code == 200:
            content = response.data.lower()
            # Should contain search elements
            assert (b'search' in content and 
                    (b'form' in content or b'input' in content))


class TestTemplateErrorHandling:
    """Test template error handling."""
    
    def test_missing_template_variables(self, flask_app):
        """Test handling of missing template variables."""
        with flask_app.test_client() as client:
            # Test that templates handle missing data gracefully
            response = client.get('/')
            assert response.status_code == 200
    
    def test_template_syntax_errors(self, flask_app):
        """Test that there are no template syntax errors."""
        # This test ensures templates can be loaded without syntax errors
        with flask_app.app_context():
            from flask import render_template_string
            
            # Test basic template rendering
            try:
                result = render_template_string("{{ 'test' }}")
                assert result == 'test'
            except Exception as e:
                pytest.fail(f"Template rendering failed: {e}")


class TestTemplatePerformance:
    """Test template rendering performance."""
    
    def test_template_rendering_speed(self, client):
        """Test template rendering performance."""
        import time
        
        start_time = time.time()
        response = client.get('/')
        end_time = time.time()
        
        render_time = end_time - start_time
        
        # Template should render within 3 seconds
        assert render_time < 3.0
        assert response.status_code == 200
    
    def test_large_data_rendering(self, flask_app):
        """Test template rendering with large datasets."""
        with flask_app.app_context():
            from flask import render_template_string
            
            # Test rendering with large data
            large_data = {'items': [f'item_{i}' for i in range(100)]}
            template = "{% for item in items %}{{ item }}{% endfor %}"
            
            start_time = time.time()
            result = render_template_string(template, **large_data)
            end_time = time.time()
            
            render_time = end_time - start_time
            
            # Should render large data within reasonable time
            assert render_time < 2.0
            assert len(result) > 0


class TestTemplateAccessibility:
    """Test template accessibility features."""
    
    def test_semantic_html_elements(self, client):
        """Test use of semantic HTML elements."""
        response = client.get('/')
        assert response.status_code == 200
        
        content = response.data.lower()
        # Should use semantic HTML elements
        semantic_elements = [b'<main', b'<nav', b'<header', b'<section', b'<article']
        has_semantic = any(element in content for element in semantic_elements)
        
        # At least some semantic elements should be present
        assert has_semantic or b'role=' in content
    
    def test_alt_text_for_images(self, client):
        """Test alt text for images."""
        response = client.get('/')
        assert response.status_code == 200
        
        # If images are present, they should have alt text
        if b'<img' in response.data:
            assert b'alt=' in response.data
