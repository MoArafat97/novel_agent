"""
Unit tests for database operations in the Lazywriter system.

This module tests WorldState, TinyDB operations, ChromaDB operations,
and semantic search functionality.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from database.world_state import WorldState
from database.tinydb_manager import TinyDBManager
from database.chromadb_manager import ChromaDBManager
from database.semantic_search import SemanticSearchEngine


class TestWorldState:
    """Test cases for WorldState unified interface."""
    
    def test_initialization(self, temp_data_dir):
        """Test WorldState initialization."""
        with patch.dict(os.environ, {
            'TINYDB_PATH': os.path.join(temp_data_dir, 'tinydb'),
            'CHROMADB_PATH': os.path.join(temp_data_dir, 'chromadb'),
            'OPENROUTER_API_KEY': 'test_key'
        }):
            world_state = WorldState()
            
            assert world_state.tinydb_path == os.path.join(temp_data_dir, 'tinydb')
            assert world_state.chromadb_path == os.path.join(temp_data_dir, 'chromadb')
            assert len(world_state.entity_types) == 4
            assert 'characters' in world_state.entity_types
            assert hasattr(world_state, 'tinydb_connections')
            assert hasattr(world_state, 'chroma_collections')
    
    def test_add_novel(self, mock_world_state, sample_novel_data, test_utils):
        """Test adding a novel."""
        result = mock_world_state.add_or_update(
            'novels', 
            sample_novel_data['id'], 
            sample_novel_data
        )
        
        assert result is True
        
        # Verify novel was added
        retrieved = mock_world_state.get('novels', sample_novel_data['id'])
        assert retrieved is not None
        assert retrieved['title'] == sample_novel_data['title']
        test_utils.assert_valid_entity_structure(retrieved, 'novels')
    
    def test_add_character(self, mock_world_state, sample_character_data, test_utils):
        """Test adding a character."""
        result = mock_world_state.add_or_update(
            'characters',
            sample_character_data['id'],
            sample_character_data,
            skip_embeddings=True  # Skip embeddings for faster testing
        )
        
        assert result is True
        
        # Verify character was added
        retrieved = mock_world_state.get('characters', sample_character_data['id'])
        assert retrieved is not None
        assert retrieved['name'] == sample_character_data['name']
        test_utils.assert_valid_entity_structure(retrieved, 'characters')
    
    def test_update_entity(self, mock_world_state, sample_character_data):
        """Test updating an existing entity."""
        # Add character first
        mock_world_state.add_or_update(
            'characters',
            sample_character_data['id'],
            sample_character_data,
            skip_embeddings=True
        )
        
        # Update character
        updated_data = sample_character_data.copy()
        updated_data['description'] = 'Updated description'
        
        result = mock_world_state.add_or_update(
            'characters',
            sample_character_data['id'],
            updated_data,
            skip_embeddings=True
        )
        
        assert result is True
        
        # Verify update
        retrieved = mock_world_state.get('characters', sample_character_data['id'])
        assert retrieved['description'] == 'Updated description'
    
    def test_delete_entity(self, mock_world_state, sample_character_data):
        """Test deleting an entity."""
        # Add character first
        mock_world_state.add_or_update(
            'characters',
            sample_character_data['id'],
            sample_character_data,
            skip_embeddings=True
        )
        
        # Delete character
        result = mock_world_state.delete('characters', sample_character_data['id'])
        assert result is True
        
        # Verify deletion
        retrieved = mock_world_state.get('characters', sample_character_data['id'])
        assert retrieved is None
    
    def test_get_entities_by_novel(self, mock_world_state, sample_novel_data):
        """Test retrieving all entities for a novel."""
        novel_id = sample_novel_data['id']
        
        # Add test entities
        from conftest import generate_test_entities
        test_entities = generate_test_entities(novel_id, count=3)
        
        for entity_type, entities in test_entities.items():
            for entity in entities:
                mock_world_state.add_or_update(
                    entity_type,
                    entity['id'],
                    entity,
                    skip_embeddings=True
                )
        
        # Retrieve entities by novel
        result = mock_world_state.get_entities_by_novel(novel_id)
        
        assert len(result['characters']) == 3
        assert len(result['locations']) == 3
        assert len(result['lore']) == 3
        
        # Verify all entities belong to the novel
        for entity_type, entities in result.items():
            for entity in entities:
                assert entity['novel_id'] == novel_id
    
    def test_get_all_entities(self, mock_world_state, sample_character_data):
        """Test retrieving all entities of a type."""
        # Add multiple characters
        characters = []
        for i in range(3):
            char_data = sample_character_data.copy()
            char_data['id'] = f"char_{i}"
            char_data['name'] = f"Character {i}"
            characters.append(char_data)
            
            mock_world_state.add_or_update(
                'characters',
                char_data['id'],
                char_data,
                skip_embeddings=True
            )
        
        # Retrieve all characters
        result = mock_world_state.get_all('characters')
        
        assert len(result) >= 3
        names = [char['name'] for char in result]
        assert 'Character 0' in names
        assert 'Character 1' in names
        assert 'Character 2' in names
    
    @pytest.mark.requires_api
    def test_semantic_query(self, mock_world_state, sample_character_data):
        """Test semantic search functionality."""
        # Mock the embedding generation
        with patch.object(mock_world_state, '_generate_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 384  # Mock embedding vector
            
            # Add character with embeddings
            mock_world_state.add_or_update(
                'characters',
                sample_character_data['id'],
                sample_character_data
            )
            
            # Perform semantic query
            results = mock_world_state.semantic_query(
                "brave warrior knight",
                entity_type='characters',
                n_results=5
            )
            
            # Should return results (mocked)
            assert isinstance(results, list)


class TestTinyDBManager:
    """Test cases for TinyDBManager."""
    
    def test_initialization(self, temp_data_dir):
        """Test TinyDBManager initialization."""
        manager = TinyDBManager(temp_data_dir)
        
        assert manager.db_path == temp_data_dir
        assert len(manager.entity_types) == 4
        assert 'characters' in manager.databases
        assert 'locations' in manager.databases
        assert 'lore' in manager.databases
        assert 'novels' in manager.databases
    
    def test_create_entity(self, temp_data_dir, sample_character_data, test_utils):
        """Test creating an entity."""
        manager = TinyDBManager(temp_data_dir)
        
        entity_id = manager.create('characters', sample_character_data)
        
        assert entity_id is not None
        test_utils.assert_valid_uuid(entity_id)
        
        # Verify entity was created
        retrieved = manager.read('characters', entity_id)
        assert retrieved is not None
        assert retrieved['name'] == sample_character_data['name']
    
    def test_read_entity(self, temp_data_dir, sample_character_data):
        """Test reading an entity."""
        manager = TinyDBManager(temp_data_dir)
        
        # Create entity first
        entity_id = manager.create('characters', sample_character_data)
        
        # Read entity
        retrieved = manager.read('characters', entity_id)
        
        assert retrieved is not None
        assert retrieved['id'] == entity_id
        assert retrieved['name'] == sample_character_data['name']
    
    def test_update_entity(self, temp_data_dir, sample_character_data):
        """Test updating an entity."""
        manager = TinyDBManager(temp_data_dir)
        
        # Create entity first
        entity_id = manager.create('characters', sample_character_data)
        
        # Update entity
        updated_data = sample_character_data.copy()
        updated_data['id'] = entity_id
        updated_data['description'] = 'Updated description'
        
        result = manager.update('characters', entity_id, updated_data)
        assert result is True
        
        # Verify update
        retrieved = manager.read('characters', entity_id)
        assert retrieved['description'] == 'Updated description'
    
    def test_delete_entity(self, temp_data_dir, sample_character_data):
        """Test deleting an entity."""
        manager = TinyDBManager(temp_data_dir)
        
        # Create entity first
        entity_id = manager.create('characters', sample_character_data)
        
        # Delete entity
        result = manager.delete('characters', entity_id)
        assert result is True
        
        # Verify deletion
        retrieved = manager.read('characters', entity_id)
        assert retrieved is None
    
    def test_list_entities(self, temp_data_dir, sample_character_data):
        """Test listing entities."""
        manager = TinyDBManager(temp_data_dir)
        
        # Create multiple entities
        entity_ids = []
        for i in range(3):
            char_data = sample_character_data.copy()
            char_data['name'] = f'Character {i}'
            entity_id = manager.create('characters', char_data)
            entity_ids.append(entity_id)
        
        # List entities
        entities = manager.list('characters')
        
        assert len(entities) >= 3
        names = [entity['name'] for entity in entities]
        assert 'Character 0' in names
        assert 'Character 1' in names
        assert 'Character 2' in names
    
    def test_invalid_entity_type(self, temp_data_dir, sample_character_data):
        """Test handling of invalid entity types."""
        manager = TinyDBManager(temp_data_dir)
        
        # Try to create with invalid entity type
        result = manager.create('invalid_type', sample_character_data)
        assert result is None
        
        # Try to read with invalid entity type
        result = manager.read('invalid_type', 'some_id')
        assert result is None
    
    def test_entity_validation(self, temp_data_dir):
        """Test entity data validation."""
        manager = TinyDBManager(temp_data_dir)
        
        # Try to create character without required fields
        invalid_data = {'name': 'Test'}  # Missing required fields
        
        result = manager.create('characters', invalid_data)
        # Should still create but with generated ID and metadata
        assert result is not None


class TestSemanticSearchEngine:
    """Test cases for SemanticSearchEngine."""
    
    def test_initialization(self, mock_world_state):
        """Test SemanticSearchEngine initialization."""
        search_engine = SemanticSearchEngine(mock_world_state)
        
        assert search_engine.world_state == mock_world_state
        assert hasattr(search_engine, 'search')
    
    def test_basic_search(self, mock_semantic_search):
        """Test basic search functionality."""
        # Mock the world_state.semantic_query method
        mock_results = [
            {
                'entity_type': 'characters',
                'data': {'name': 'Test Character', 'description': 'A test character'},
                'similarity_score': 0.8
            }
        ]
        
        with patch.object(mock_semantic_search.world_state, 'semantic_query') as mock_query:
            mock_query.return_value = mock_results
            
            results = mock_semantic_search.search(
                query="test character",
                n_results=5
            )
            
            assert len(results) == 1
            assert results[0]['data']['name'] == 'Test Character'
    
    def test_filtered_search(self, mock_semantic_search):
        """Test search with entity type filtering."""
        mock_results = [
            {
                'entity_type': 'characters',
                'data': {'name': 'Test Character', 'novel_id': 'novel_1'},
                'similarity_score': 0.8
            }
        ]
        
        with patch.object(mock_semantic_search.world_state, 'semantic_query') as mock_query:
            mock_query.return_value = mock_results
            
            results = mock_semantic_search.search(
                query="test",
                entity_types=['characters'],
                novel_id='novel_1',
                n_results=5
            )
            
            assert len(results) == 1
            assert results[0]['entity_type'] == 'characters'
    
    def test_search_suggestions(self, mock_semantic_search):
        """Test search suggestions functionality."""
        # Mock entities for suggestions
        mock_entities = {
            'characters': [
                {'name': 'Aragorn', 'tags': ['king', 'ranger']},
                {'name': 'Gandalf', 'tags': ['wizard', 'grey']}
            ]
        }
        
        with patch.object(mock_semantic_search.world_state, 'get_entities_by_novel') as mock_get:
            mock_get.return_value = mock_entities
            
            suggestions = mock_semantic_search.get_search_suggestions(
                partial_query="ara",
                novel_id="test_novel"
            )
            
            assert isinstance(suggestions, list)
            # Should include matching names
            assert any('Aragorn' in str(suggestions))


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_worldstate_tinydb_chromadb_sync(self, temp_data_dir, sample_character_data):
        """Test synchronization between TinyDB and ChromaDB."""
        with patch.dict(os.environ, {
            'TINYDB_PATH': os.path.join(temp_data_dir, 'tinydb'),
            'CHROMADB_PATH': os.path.join(temp_data_dir, 'chromadb'),
            'OPENROUTER_API_KEY': 'test_key'
        }):
            # Mock embedding generation to avoid API calls
            with patch('database.world_state.WorldState._generate_embedding') as mock_embed:
                mock_embed.return_value = [0.1] * 384
                
                world_state = WorldState()
                
                # Add character
                result = world_state.add_or_update(
                    'characters',
                    sample_character_data['id'],
                    sample_character_data
                )
                
                assert result is True
                
                # Verify in TinyDB
                tinydb_result = world_state.get('characters', sample_character_data['id'])
                assert tinydb_result is not None
                assert tinydb_result['name'] == sample_character_data['name']
                
                # Verify embedding was generated (mocked)
                mock_embed.assert_called()
    
    def test_full_crud_workflow(self, mock_world_state, sample_character_data, test_utils):
        """Test complete CRUD workflow."""
        entity_id = sample_character_data['id']
        
        # Create
        result = mock_world_state.add_or_update(
            'characters', entity_id, sample_character_data, skip_embeddings=True
        )
        assert result is True
        
        # Read
        retrieved = mock_world_state.get('characters', entity_id)
        assert retrieved is not None
        test_utils.assert_valid_entity_structure(retrieved, 'characters')
        
        # Update
        updated_data = sample_character_data.copy()
        updated_data['description'] = 'Updated description'
        result = mock_world_state.add_or_update(
            'characters', entity_id, updated_data, skip_embeddings=True
        )
        assert result is True
        
        retrieved = mock_world_state.get('characters', entity_id)
        assert retrieved['description'] == 'Updated description'
        
        # Delete
        result = mock_world_state.delete('characters', entity_id)
        assert result is True
        
        retrieved = mock_world_state.get('characters', entity_id)
        assert retrieved is None
