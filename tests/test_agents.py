"""
Unit tests for AI agents in the Lazywriter system.

This module tests all AI agents including character, lore, and location
creators and editors, as well as the cross-reference system.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from agents.character_creator import CharacterCreatorAgent
from agents.character_editor import CharacterEditorAgent
from agents.lore_creator import LoreCreatorAgent
from agents.lore_editor import LoreEditorAgent
from agents.location_creator import LocationCreatorAgent
from agents.location_editor import LocationEditorAgent
from agents.cross_reference_agent import CrossReferenceAgent


class TestCharacterCreatorAgent:
    """Test cases for CharacterCreatorAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = CharacterCreatorAgent()
        assert agent.chat_model == 'deepseek/deepseek-chat:free'
        assert agent.openrouter_api_key is None  # No API key in test env
    
    def test_create_character_success(self, character_creator_agent, test_utils):
        """Test successful character creation."""
        # Mock AI response
        ai_response = test_utils.mock_ai_response({
            "description": "A brave knight with noble heart",
            "age": "28",
            "occupation": "Knight",
            "personality": "Brave, honorable, determined",
            "backstory": "Born to a noble family, trained in combat",
            "tags": ["protagonist", "knight", "noble"],
            "role": "protagonist"
        })
        
        character_creator_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
        
        result = character_creator_agent.create_character(
            name="Sir Galahad",
            user_prompt="A noble knight seeking the holy grail",
            novel_context={"title": "Test Novel", "genre": "Fantasy"}
        )
        
        assert result["name"] == "Sir Galahad"
        assert result["description"] == "A brave knight with noble heart"
        assert result["role"] == "protagonist"
        assert "knight" in result["tags"]
        test_utils.assert_valid_uuid(result["id"])
    
    def test_create_character_api_failure(self, character_creator_agent):
        """Test character creation with API failure."""
        # Mock API failure
        character_creator_agent.openrouter_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = character_creator_agent.create_character(
            name="Test Character",
            user_prompt="A test character"
        )
        
        # Should return fallback character
        assert result["name"] == "Test Character"
        assert "fallback" in result["description"].lower()
        assert result["role"] == "supporting character"
    
    def test_create_character_invalid_json(self, character_creator_agent):
        """Test character creation with invalid JSON response."""
        # Mock invalid JSON response
        character_creator_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = "Invalid JSON"
        
        result = character_creator_agent.create_character(
            name="Test Character",
            user_prompt="A test character"
        )
        
        # Should return fallback character
        assert result["name"] == "Test Character"
        assert "fallback" in result["description"].lower()


class TestCharacterEditorAgent:
    """Test cases for CharacterEditorAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = CharacterEditorAgent()
        assert agent.chat_model == 'deepseek/deepseek-chat:free'
        assert hasattr(agent, '_response_cache')
        assert agent._cache_max_size == 50
    
    def test_edit_character_success(self, character_editor_agent, sample_character_data, test_utils):
        """Test successful character editing."""
        # Mock AI response
        updated_data = sample_character_data.copy()
        updated_data["description"] = "An experienced warrior with battle scars"
        updated_data["age"] = "30"
        
        ai_response = test_utils.mock_ai_response(updated_data)
        character_editor_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
        
        result = character_editor_agent.edit_character(
            current_character=sample_character_data,
            edit_request="Make the character older and more experienced"
        )
        
        assert result["description"] == "An experienced warrior with battle scars"
        assert result["age"] == "30"
        assert result["name"] == sample_character_data["name"]  # Name should remain
    
    def test_edit_character_caching(self, character_editor_agent, sample_character_data, test_utils):
        """Test response caching functionality."""
        ai_response = test_utils.mock_ai_response(sample_character_data)
        character_editor_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
        
        edit_request = "Test edit request"
        
        # First call
        result1 = character_editor_agent.edit_character(
            current_character=sample_character_data,
            edit_request=edit_request
        )
        
        # Second call should use cache
        result2 = character_editor_agent.edit_character(
            current_character=sample_character_data,
            edit_request=edit_request
        )
        
        assert result1 == result2
        # API should only be called once
        assert character_editor_agent.openrouter_client.chat.completions.create.call_count == 1
    
    def test_edit_character_api_failure(self, character_editor_agent, sample_character_data):
        """Test character editing with API failure."""
        character_editor_agent.openrouter_client.chat.completions.create.side_effect = Exception("API Error")
        
        result = character_editor_agent.edit_character(
            current_character=sample_character_data,
            edit_request="Make changes"
        )
        
        # Should return original character unchanged
        assert result == sample_character_data


class TestLoreCreatorAgent:
    """Test cases for LoreCreatorAgent."""
    
    def test_create_lore_success(self, lore_creator_agent, test_utils):
        """Test successful lore creation."""
        ai_response = test_utils.mock_ai_response({
            "content": "Ancient magic flows through crystalline structures",
            "category": "Magic System",
            "importance": "High",
            "connections": "Connected to the ancient civilization",
            "tags": ["magic", "crystals", "ancient"]
        })
        
        lore_creator_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
        
        result = lore_creator_agent.create_lore(
            title="Crystal Magic",
            user_prompt="A magic system based on crystals",
            novel_context={"title": "Test Novel", "genre": "Fantasy"}
        )
        
        assert result["title"] == "Crystal Magic"
        assert "crystalline" in result["content"]
        assert result["category"] == "Magic System"
        assert "magic" in result["tags"]
        test_utils.assert_valid_uuid(result["id"])


class TestLocationCreatorAgent:
    """Test cases for LocationCreatorAgent."""
    
    def test_create_location_success(self, location_creator_agent, test_utils):
        """Test successful location creation."""
        ai_response = test_utils.mock_ai_response({
            "description": "A towering castle built on ancient foundations",
            "type": "Castle",
            "significance": "Royal seat of power and ancient fortress",
            "history": "Built by the first king over ancient ruins",
            "tags": ["castle", "royal", "ancient", "fortress"]
        })
        
        location_creator_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
        
        result = location_creator_agent.create_location(
            name="Dragonspire Castle",
            user_prompt="An ancient castle on a mountain peak",
            novel_context={"title": "Test Novel", "genre": "Fantasy"}
        )
        
        assert result["name"] == "Dragonspire Castle"
        assert "towering castle" in result["description"]
        assert result["type"] == "Castle"
        assert "castle" in result["tags"]
        test_utils.assert_valid_uuid(result["id"])


class TestCrossReferenceAgent:
    """Test cases for CrossReferenceAgent."""
    
    def test_initialization(self, mock_world_state, mock_semantic_search):
        """Test cross-reference agent initialization."""
        agent = CrossReferenceAgent(mock_world_state, mock_semantic_search)
        assert agent.world_state == mock_world_state
        assert agent.semantic_search == mock_semantic_search
        assert hasattr(agent, 'multi_agent_coordinator')
    
    @pytest.mark.slow
    def test_analyze_entity_basic(self, cross_reference_agent, sample_character_data):
        """Test basic entity analysis."""
        novel_id = sample_character_data["novel_id"]
        entity_id = sample_character_data["id"]
        
        # Mock the streaming manager
        with patch.object(cross_reference_agent, 'streaming_manager') as mock_streaming:
            mock_streaming.create_job.return_value = "test_job_id"
            
            # Mock entity detection
            with patch.object(cross_reference_agent, '_detect_entity_mentions') as mock_detect:
                mock_detect.return_value = []
                
                # Mock semantic search
                with patch.object(cross_reference_agent, '_find_semantic_matches') as mock_semantic:
                    mock_semantic.return_value = []
                    
                    result = cross_reference_agent.analyze_entity(
                        entity_type="characters",
                        entity_id=entity_id,
                        novel_id=novel_id
                    )
                    
                    assert "job_id" in result
                    assert result["status"] == "processing"
    
    def test_extract_entity_content(self, cross_reference_agent, sample_character_data):
        """Test entity content extraction."""
        content = cross_reference_agent._extract_entity_content(
            sample_character_data, "characters"
        )
        
        assert sample_character_data["name"] in content
        assert sample_character_data["description"] in content
        assert sample_character_data["backstory"] in content
    
    def test_classify_entity_type(self, cross_reference_agent):
        """Test entity type classification."""
        # Test character classification
        char_result = cross_reference_agent._classify_entity_type(
            "Aragorn", "The brave warrior fought valiantly"
        )
        assert char_result == "characters"
        
        # Test location classification  
        loc_result = cross_reference_agent._classify_entity_type(
            "Rivendell", "The elven city in the mountains"
        )
        assert loc_result == "locations"
        
        # Test unknown classification
        unknown_result = cross_reference_agent._classify_entity_type(
            "Something", "Generic text without clear indicators"
        )
        assert unknown_result == "unknown"


class TestAgentErrorHandling:
    """Test error handling across all agents."""
    
    def test_no_api_key_handling(self):
        """Test agent behavior when no API key is provided."""
        with patch.dict('os.environ', {}, clear=True):
            agent = CharacterCreatorAgent()
            assert agent.openrouter_client is None
            
            result = agent.create_character("Test", "Test prompt")
            assert "fallback" in result["description"].lower()
    
    def test_network_timeout_handling(self, character_creator_agent):
        """Test handling of network timeouts."""
        import requests
        character_creator_agent.openrouter_client.chat.completions.create.side_effect = requests.Timeout("Timeout")
        
        result = character_creator_agent.create_character("Test", "Test prompt")
        assert result["name"] == "Test"
        assert "fallback" in result["description"].lower()
    
    def test_rate_limit_handling(self, character_creator_agent):
        """Test handling of API rate limits."""
        from openai import RateLimitError
        character_creator_agent.openrouter_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=Mock(), body=None
        )
        
        result = character_creator_agent.create_character("Test", "Test prompt")
        assert result["name"] == "Test"
        assert "fallback" in result["description"].lower()


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for agent workflows."""
    
    def test_character_creation_to_editing_workflow(self, character_creator_agent, character_editor_agent, test_utils):
        """Test complete character creation and editing workflow."""
        # Create character
        create_response = test_utils.mock_ai_response({
            "description": "A young mage learning magic",
            "age": "20",
            "occupation": "Apprentice Mage",
            "personality": "Curious and eager to learn",
            "backstory": "Discovered magical abilities recently",
            "tags": ["mage", "apprentice", "young"],
            "role": "secondary protagonist"
        })
        
        character_creator_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = create_response
        
        created_character = character_creator_agent.create_character(
            name="Lyra",
            user_prompt="A young mage learning her powers"
        )
        
        # Edit character
        edit_response = test_utils.mock_ai_response({
            **created_character,
            "description": "An experienced mage with mastery over elements",
            "age": "25",
            "occupation": "Elemental Mage"
        })
        
        character_editor_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = edit_response
        
        edited_character = character_editor_agent.edit_character(
            current_character=created_character,
            edit_request="Make her more experienced and powerful"
        )
        
        assert edited_character["name"] == "Lyra"
        assert "experienced" in edited_character["description"]
        assert edited_character["age"] == "25"
        assert edited_character["id"] == created_character["id"]  # ID should remain same
