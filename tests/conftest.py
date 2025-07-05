"""
Pytest configuration and shared fixtures for Lazywriter tests.

This module provides common fixtures and test utilities used across
all test modules in the Lazywriter testing suite.
"""

import os
import sys
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.world_state import WorldState
from database.semantic_search import SemanticSearchEngine
from agents.character_creator import CharacterCreatorAgent
from agents.character_editor import CharacterEditorAgent
from agents.lore_creator import LoreCreatorAgent
from agents.lore_editor import LoreEditorAgent
from agents.location_creator import LocationCreatorAgent
from agents.location_editor import LocationEditorAgent
from agents.cross_reference_agent import CrossReferenceAgent


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="lazywriter_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_openrouter_client():
    """Mock OpenRouter client for testing AI agents."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '{"test": "response"}'
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_novel_data():
    """Sample novel data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "title": "Test Novel",
        "author": "Test Author",
        "genre": "Fantasy",
        "description": "A test novel for unit testing",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_character_data():
    """Sample character data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "novel_id": str(uuid.uuid4()),
        "name": "Test Character",
        "description": "A brave warrior with a mysterious past",
        "age": "25",
        "occupation": "Knight",
        "personality": "Brave, loyal, and determined",
        "backstory": "Born in a small village, trained as a knight",
        "tags": ["protagonist", "warrior", "noble"],
        "role": "protagonist",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_location_data():
    """Sample location data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "novel_id": str(uuid.uuid4()),
        "name": "Test Castle",
        "description": "A magnificent castle on a hill",
        "type": "Castle",
        "significance": "Royal residence and fortress",
        "history": "Built centuries ago by ancient kings",
        "tags": ["castle", "royal", "fortress"],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def sample_lore_data():
    """Sample lore data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "novel_id": str(uuid.uuid4()),
        "title": "Test Magic System",
        "content": "Magic flows through ancient crystals",
        "category": "Magic",
        "importance": "High",
        "connections": "Connected to the ancient civilization",
        "tags": ["magic", "crystals", "ancient"],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_world_state(temp_data_dir):
    """Mock WorldState for testing."""
    with patch.dict(os.environ, {
        'TINYDB_PATH': os.path.join(temp_data_dir, 'tinydb'),
        'CHROMADB_PATH': os.path.join(temp_data_dir, 'chromadb'),
        'OPENROUTER_API_KEY': 'test_key'
    }):
        world_state = WorldState()
        yield world_state


@pytest.fixture
def mock_semantic_search(mock_world_state):
    """Mock SemanticSearchEngine for testing."""
    return SemanticSearchEngine(mock_world_state)


@pytest.fixture
def character_creator_agent(mock_openrouter_client):
    """CharacterCreatorAgent with mocked OpenRouter client."""
    agent = CharacterCreatorAgent()
    agent.openrouter_client = mock_openrouter_client
    return agent


@pytest.fixture
def character_editor_agent(mock_openrouter_client):
    """CharacterEditorAgent with mocked OpenRouter client."""
    agent = CharacterEditorAgent()
    agent.openrouter_client = mock_openrouter_client
    return agent


@pytest.fixture
def lore_creator_agent(mock_openrouter_client):
    """LoreCreatorAgent with mocked OpenRouter client."""
    agent = LoreCreatorAgent()
    agent.openrouter_client = mock_openrouter_client
    return agent


@pytest.fixture
def lore_editor_agent(mock_openrouter_client):
    """LoreEditorAgent with mocked OpenRouter client."""
    agent = LoreEditorAgent()
    agent.openrouter_client = mock_openrouter_client
    return agent


@pytest.fixture
def location_creator_agent(mock_openrouter_client):
    """LocationCreatorAgent with mocked OpenRouter client."""
    agent = LocationCreatorAgent()
    agent.openrouter_client = mock_openrouter_client
    return agent


@pytest.fixture
def location_editor_agent(mock_openrouter_client):
    """LocationEditorAgent with mocked OpenRouter client."""
    agent = LocationEditorAgent()
    agent.openrouter_client = mock_openrouter_client
    return agent


@pytest.fixture
def cross_reference_agent(mock_world_state, mock_semantic_search):
    """CrossReferenceAgent with mocked dependencies."""
    return CrossReferenceAgent(mock_world_state, mock_semantic_search)


# Test data generators
def generate_test_entities(novel_id: str, count: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Generate test entities for a novel."""
    entities = {
        'characters': [],
        'locations': [],
        'lore': []
    }
    
    for i in range(count):
        entities['characters'].append({
            'id': str(uuid.uuid4()),
            'novel_id': novel_id,
            'name': f'Character {i+1}',
            'description': f'Test character {i+1}',
            'tags': [f'tag{i+1}', 'test']
        })
        
        entities['locations'].append({
            'id': str(uuid.uuid4()),
            'novel_id': novel_id,
            'name': f'Location {i+1}',
            'description': f'Test location {i+1}',
            'tags': [f'place{i+1}', 'test']
        })
        
        entities['lore'].append({
            'id': str(uuid.uuid4()),
            'novel_id': novel_id,
            'title': f'Lore {i+1}',
            'content': f'Test lore content {i+1}',
            'tags': [f'lore{i+1}', 'test']
        })
    
    return entities


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def assert_valid_entity_structure(entity: Dict[str, Any], entity_type: str):
        """Assert that an entity has the required structure."""
        required_fields = {
            'characters': ['id', 'novel_id', 'name', 'description'],
            'locations': ['id', 'novel_id', 'name', 'description'],
            'lore': ['id', 'novel_id', 'title', 'content'],
            'novels': ['id', 'title', 'author']
        }
        
        for field in required_fields.get(entity_type, []):
            assert field in entity, f"Missing required field '{field}' in {entity_type}"
            assert entity[field] is not None, f"Field '{field}' cannot be None in {entity_type}"
    
    @staticmethod
    def assert_valid_uuid(uuid_string: str):
        """Assert that a string is a valid UUID."""
        try:
            uuid.UUID(uuid_string)
        except ValueError:
            pytest.fail(f"'{uuid_string}' is not a valid UUID")
    
    @staticmethod
    def mock_ai_response(response_data: Dict[str, Any]) -> str:
        """Generate a mock AI response in JSON format."""
        import json
        return json.dumps(response_data)


@pytest.fixture
def test_utils():
    """Test utilities fixture."""
    return TestUtils
