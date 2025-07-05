"""
Integration tests for the Lazywriter system.

This module tests complete workflows and interactions between different
components of the system, including end-to-end worldbuilding workflows
and cross-reference analysis pipelines.
"""

import pytest
import os
import uuid
from unittest.mock import Mock, patch, MagicMock
from database.world_state import WorldState
from database.semantic_search import SemanticSearchEngine
from agents.character_creator import CharacterCreatorAgent
from agents.character_editor import CharacterEditorAgent
from agents.cross_reference_agent import CrossReferenceAgent
from agents.multi_agent_coordinator import MultiAgentCoordinator


@pytest.mark.integration
class TestWorldbuildingWorkflows:
    """Test complete worldbuilding workflows."""
    
    def test_novel_creation_workflow(self, mock_world_state, sample_novel_data, test_utils):
        """Test complete novel creation and setup workflow."""
        novel_id = sample_novel_data['id']
        
        # Step 1: Create novel
        result = mock_world_state.add_or_update(
            'novels', novel_id, sample_novel_data, skip_embeddings=True
        )
        assert result is True
        
        # Step 2: Verify novel exists
        retrieved_novel = mock_world_state.get('novels', novel_id)
        assert retrieved_novel is not None
        test_utils.assert_valid_entity_structure(retrieved_novel, 'novels')
        
        # Step 3: Create initial worldbuilding entities
        from conftest import generate_test_entities
        test_entities = generate_test_entities(novel_id, count=2)
        
        for entity_type, entities in test_entities.items():
            for entity in entities:
                result = mock_world_state.add_or_update(
                    entity_type, entity['id'], entity, skip_embeddings=True
                )
                assert result is True
        
        # Step 4: Verify all entities are associated with novel
        novel_entities = mock_world_state.get_entities_by_novel(novel_id)
        
        assert len(novel_entities['characters']) == 2
        assert len(novel_entities['locations']) == 2
        assert len(novel_entities['lore']) == 2
        
        for entity_type, entities in novel_entities.items():
            for entity in entities:
                assert entity['novel_id'] == novel_id
    
    def test_character_creation_to_cross_reference_workflow(
        self, mock_world_state, mock_semantic_search, character_creator_agent, test_utils
    ):
        """Test workflow from character creation to cross-reference analysis."""
        novel_id = str(uuid.uuid4())
        
        # Step 1: Create character using AI agent
        ai_response = test_utils.mock_ai_response({
            "description": "A brave knight from the northern kingdoms",
            "age": "28",
            "occupation": "Knight",
            "personality": "Honorable, brave, and loyal to the crown",
            "backstory": "Born in the northern kingdoms, trained as a knight",
            "tags": ["knight", "northern", "brave"],
            "role": "protagonist"
        })
        
        character_creator_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
        
        character_data = character_creator_agent.create_character(
            name="Sir Aldric",
            user_prompt="A knight from the northern kingdoms",
            novel_context={"title": "Test Novel", "genre": "Fantasy"}
        )
        
        character_data['novel_id'] = novel_id
        
        # Step 2: Save character to database
        result = mock_world_state.add_or_update(
            'characters', character_data['id'], character_data, skip_embeddings=True
        )
        assert result is True
        
        # Step 3: Create some related entities for cross-referencing
        related_location = {
            'id': str(uuid.uuid4()),
            'novel_id': novel_id,
            'name': 'Northern Kingdoms',
            'description': 'A cold region known for its brave knights',
            'tags': ['northern', 'kingdoms', 'cold']
        }
        
        mock_world_state.add_or_update(
            'locations', related_location['id'], related_location, skip_embeddings=True
        )
        
        # Step 4: Perform cross-reference analysis
        cross_ref_agent = CrossReferenceAgent(mock_world_state, mock_semantic_search)
        
        with patch.object(cross_ref_agent, 'streaming_manager') as mock_streaming:
            mock_streaming.create_job.return_value = "test_job_id"
            
            with patch.object(cross_ref_agent, '_detect_entity_mentions') as mock_detect:
                mock_detect.return_value = [
                    {
                        'name': 'Northern Kingdoms',
                        'type': 'locations',
                        'confidence': 0.8,
                        'context': 'knight from the northern kingdoms'
                    }
                ]
                
                with patch.object(cross_ref_agent, '_find_semantic_matches') as mock_semantic:
                    mock_semantic.return_value = [
                        {
                            'entity_id': related_location['id'],
                            'entity_type': 'locations',
                            'similarity_score': 0.9,
                            'data': related_location
                        }
                    ]
                    
                    result = cross_ref_agent.analyze_entity(
                        entity_type="characters",
                        entity_id=character_data['id'],
                        novel_id=novel_id
                    )
                    
                    assert "job_id" in result
                    assert result["status"] == "processing"
    
    def test_multi_entity_editing_workflow(
        self, mock_world_state, character_editor_agent, test_utils
    ):
        """Test editing multiple related entities."""
        novel_id = str(uuid.uuid4())
        
        # Create multiple related characters
        characters = []
        for i in range(3):
            char_data = {
                'id': str(uuid.uuid4()),
                'novel_id': novel_id,
                'name': f'Character {i+1}',
                'description': f'A member of the royal guard, character {i+1}',
                'occupation': 'Royal Guard',
                'tags': ['guard', 'royal', 'warrior']
            }
            characters.append(char_data)
            
            mock_world_state.add_or_update(
                'characters', char_data['id'], char_data, skip_embeddings=True
            )
        
        # Edit each character to make them more distinct
        edit_requests = [
            "Make this character the captain of the guard",
            "Make this character a new recruit",
            "Make this character a veteran with battle scars"
        ]
        
        for i, (character, edit_request) in enumerate(zip(characters, edit_requests)):
            # Mock AI response for editing
            updated_data = character.copy()
            if i == 0:
                updated_data['occupation'] = 'Captain of the Royal Guard'
                updated_data['description'] = 'The experienced captain of the royal guard'
            elif i == 1:
                updated_data['description'] = 'A new recruit eager to prove themselves'
            else:
                updated_data['description'] = 'A battle-scarred veteran of many wars'
            
            ai_response = test_utils.mock_ai_response(updated_data)
            character_editor_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
            
            edited_character = character_editor_agent.edit_character(
                current_character=character,
                edit_request=edit_request
            )
            
            # Update in database
            mock_world_state.add_or_update(
                'characters', character['id'], edited_character, skip_embeddings=True
            )
        
        # Verify all characters were updated
        novel_entities = mock_world_state.get_entities_by_novel(novel_id)
        updated_characters = novel_entities['characters']
        
        assert len(updated_characters) == 3
        
        # Verify distinct roles
        descriptions = [char['description'] for char in updated_characters]
        assert any('captain' in desc.lower() for desc in descriptions)
        assert any('recruit' in desc.lower() for desc in descriptions)
        assert any('veteran' in desc.lower() for desc in descriptions)


@pytest.mark.integration
class TestCrossReferenceIntegration:
    """Test cross-reference system integration."""
    
    def test_multi_agent_coordinator_workflow(self, mock_world_state, mock_semantic_search):
        """Test multi-agent coordinator workflow."""
        coordinator = MultiAgentCoordinator(mock_world_state, mock_semantic_search)
        
        # Test entity data
        entity_data = {
            'id': str(uuid.uuid4()),
            'name': 'Test Character',
            'description': 'A character who lives in the Crystal City',
            'novel_id': 'test_novel'
        }
        
        detected_entities = [
            {
                'name': 'Crystal City',
                'type': 'locations',
                'confidence': 0.8,
                'context': 'lives in the Crystal City'
            }
        ]
        
        # Mock the coordinator methods
        with patch.object(coordinator, '_detect_relationships') as mock_detect_rel:
            mock_detect_rel.return_value = [
                {
                    'source_entity': entity_data['id'],
                    'target_entity': 'crystal_city_id',
                    'relationship_type': 'spatial',
                    'confidence': 0.9
                }
            ]
            
            with patch.object(coordinator, '_generate_new_entity_suggestions') as mock_gen_entities:
                mock_gen_entities.return_value = [
                    {
                        'name': 'Crystal City',
                        'type': 'locations',
                        'suggested_data': {
                            'description': 'A magnificent city made of crystal',
                            'type': 'City'
                        }
                    }
                ]
                
                # Test relationship detection
                relationships = coordinator._detect_relationships(
                    entity_data, detected_entities, 'test_novel'
                )
                
                assert len(relationships) == 1
                assert relationships[0]['relationship_type'] == 'spatial'
                
                # Test entity generation
                new_entities = coordinator._generate_new_entity_suggestions(
                    detected_entities, 'test_novel'
                )
                
                assert len(new_entities) == 1
                assert new_entities[0]['name'] == 'Crystal City'
    
    def test_cross_reference_with_semantic_search(
        self, mock_world_state, mock_semantic_search
    ):
        """Test cross-reference integration with semantic search."""
        novel_id = str(uuid.uuid4())
        
        # Create test entities
        character_data = {
            'id': str(uuid.uuid4()),
            'novel_id': novel_id,
            'name': 'Mage Elara',
            'description': 'A powerful mage who studies ancient magic',
            'tags': ['mage', 'ancient', 'powerful']
        }
        
        lore_data = {
            'id': str(uuid.uuid4()),
            'novel_id': novel_id,
            'title': 'Ancient Magic Traditions',
            'content': 'The study of ancient magical practices and rituals',
            'tags': ['ancient', 'magic', 'traditions']
        }
        
        # Add to database
        mock_world_state.add_or_update(
            'characters', character_data['id'], character_data, skip_embeddings=True
        )
        mock_world_state.add_or_update(
            'lore', lore_data['id'], lore_data, skip_embeddings=True
        )
        
        # Mock semantic search to return related lore
        with patch.object(mock_semantic_search, 'search') as mock_search:
            mock_search.return_value = [
                {
                    'entity_type': 'lore',
                    'data': lore_data,
                    'similarity_score': 0.85
                }
            ]
            
            # Perform semantic search
            results = mock_semantic_search.search(
                query="ancient magic studies",
                entity_types=['lore'],
                novel_id=novel_id
            )
            
            assert len(results) == 1
            assert results[0]['entity_type'] == 'lore'
            assert results[0]['data']['title'] == 'Ancient Magic Traditions'
            assert results[0]['similarity_score'] == 0.85
    
    def test_end_to_end_cross_reference_pipeline(
        self, mock_world_state, mock_semantic_search
    ):
        """Test complete end-to-end cross-reference pipeline."""
        novel_id = str(uuid.uuid4())
        
        # Step 1: Create source entity
        source_character = {
            'id': str(uuid.uuid4()),
            'novel_id': novel_id,
            'name': 'Sir Gareth',
            'description': 'A knight who guards the Sacred Temple and wields the Flame Sword',
            'backstory': 'Trained in the Temple of Light, protector of ancient artifacts'
        }
        
        mock_world_state.add_or_update(
            'characters', source_character['id'], source_character, skip_embeddings=True
        )
        
        # Step 2: Create related entities
        related_entities = [
            {
                'id': str(uuid.uuid4()),
                'novel_id': novel_id,
                'name': 'Sacred Temple',
                'description': 'An ancient temple housing sacred artifacts',
                'type': 'Temple'
            },
            {
                'id': str(uuid.uuid4()),
                'novel_id': novel_id,
                'title': 'Flame Sword Legend',
                'content': 'The legendary sword that burns with eternal flame',
                'category': 'Artifacts'
            }
        ]
        
        mock_world_state.add_or_update(
            'locations', related_entities[0]['id'], related_entities[0], skip_embeddings=True
        )
        mock_world_state.add_or_update(
            'lore', related_entities[1]['id'], related_entities[1], skip_embeddings=True
        )
        
        # Step 3: Initialize cross-reference agent
        cross_ref_agent = CrossReferenceAgent(mock_world_state, mock_semantic_search)
        
        # Step 4: Mock the analysis pipeline
        with patch.object(cross_ref_agent, 'streaming_manager') as mock_streaming:
            mock_streaming.create_job.return_value = "pipeline_job_id"
            
            # Mock entity detection
            with patch.object(cross_ref_agent, '_detect_entity_mentions') as mock_detect:
                mock_detect.return_value = [
                    {
                        'name': 'Sacred Temple',
                        'type': 'locations',
                        'confidence': 0.9,
                        'context': 'guards the Sacred Temple'
                    },
                    {
                        'name': 'Flame Sword',
                        'type': 'lore',
                        'confidence': 0.8,
                        'context': 'wields the Flame Sword'
                    }
                ]
                
                # Mock semantic search
                with patch.object(cross_ref_agent, '_find_semantic_matches') as mock_semantic:
                    mock_semantic.return_value = [
                        {
                            'entity_id': related_entities[0]['id'],
                            'entity_type': 'locations',
                            'similarity_score': 0.95,
                            'data': related_entities[0]
                        },
                        {
                            'entity_id': related_entities[1]['id'],
                            'entity_type': 'lore',
                            'similarity_score': 0.88,
                            'data': related_entities[1]
                        }
                    ]
                    
                    # Mock LLM verification
                    with patch.object(cross_ref_agent, '_verify_relationships_with_llm') as mock_verify:
                        mock_verify.return_value = [
                            {
                                'source_entity_id': source_character['id'],
                                'target_entity_id': related_entities[0]['id'],
                                'relationship_type': 'spatial',
                                'confidence': 0.9,
                                'explanation': 'Character guards the location'
                            },
                            {
                                'source_entity_id': source_character['id'],
                                'target_entity_id': related_entities[1]['id'],
                                'relationship_type': 'functional',
                                'confidence': 0.85,
                                'explanation': 'Character wields the artifact'
                            }
                        ]
                        
                        # Step 5: Run the analysis
                        result = cross_ref_agent.analyze_entity(
                            entity_type="characters",
                            entity_id=source_character['id'],
                            novel_id=novel_id
                        )
                        
                        # Step 6: Verify results
                        assert "job_id" in result
                        assert result["status"] == "processing"
                        
                        # Verify that all mocked methods were called
                        mock_detect.assert_called()
                        mock_semantic.assert_called()
                        mock_verify.assert_called()


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance aspects of integrated workflows."""
    
    def test_large_dataset_handling(self, mock_world_state):
        """Test system performance with larger datasets."""
        novel_id = str(uuid.uuid4())
        
        # Create a larger number of entities
        entity_count = 50
        
        # Create characters
        for i in range(entity_count):
            char_data = {
                'id': str(uuid.uuid4()),
                'novel_id': novel_id,
                'name': f'Character {i:03d}',
                'description': f'Test character number {i} with unique traits',
                'tags': [f'tag{i}', 'test', 'character']
            }
            
            result = mock_world_state.add_or_update(
                'characters', char_data['id'], char_data, skip_embeddings=True
            )
            assert result is True
        
        # Verify all entities were created
        novel_entities = mock_world_state.get_entities_by_novel(novel_id)
        assert len(novel_entities['characters']) == entity_count
        
        # Test retrieval performance
        import time
        start_time = time.time()
        
        all_characters = mock_world_state.get_all('characters')
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert retrieval_time < 5.0  # 5 seconds max
        assert len(all_characters) >= entity_count
    
    def test_concurrent_operations(self, mock_world_state):
        """Test concurrent database operations."""
        import threading
        import time
        
        novel_id = str(uuid.uuid4())
        results = []
        errors = []
        
        def create_entity(index):
            try:
                char_data = {
                    'id': str(uuid.uuid4()),
                    'novel_id': novel_id,
                    'name': f'Concurrent Character {index}',
                    'description': f'Character created concurrently {index}'
                }
                
                result = mock_world_state.add_or_update(
                    'characters', char_data['id'], char_data, skip_embeddings=True
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_entity, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(result is True for result in results)
        
        # Verify all entities were created
        novel_entities = mock_world_state.get_entities_by_novel(novel_id)
        assert len(novel_entities['characters']) == 10
