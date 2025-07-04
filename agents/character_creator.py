"""
AI Character Creator Agent

This agent uses DeepSeek LLM to intelligently create detailed characters
from simple user prompts. It analyzes the input and populates all character
fields with rich, contextual information.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import openai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class CharacterCreatorAgent:
    """
    AI agent that creates detailed characters from simple prompts using DeepSeek LLM.
    """
    
    def __init__(self):
        """Initialize the Character Creator Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Character creator will not work.")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client for chat completions."""
        try:
            self.openrouter_client = openai.OpenAI(
                base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
                api_key=self.openrouter_api_key
            )
            logger.info("Character Creator Agent initialized with DeepSeek")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter for character creation: {e}")
            self.openrouter_client = None
    
    def create_character(self, 
                        name: str, 
                        user_prompt: str, 
                        novel_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a detailed character from a simple prompt.
        
        Args:
            name: Character name
            user_prompt: User's description/prompt about the character
            novel_context: Context about the novel (title, genre, etc.)
            
        Returns:
            Dictionary with detailed character information
        """
        if not self.openrouter_client:
            logger.error("OpenRouter client not available")
            return self._create_fallback_character(name, user_prompt)
        
        try:
            # Build the AI prompt
            system_prompt = self._build_system_prompt(novel_context)
            user_message = self._build_user_prompt(name, user_prompt, novel_context)
            
            # Call DeepSeek LLM
            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=1500
            )
            
            # Parse the response
            ai_response = response.choices[0].message.content
            character_data = self._parse_ai_response(ai_response, name, user_prompt)
            
            logger.info(f"Successfully created character '{name}' using AI")
            return character_data
            
        except Exception as e:
            logger.error(f"AI character creation failed: {e}")
            return self._create_fallback_character(name, user_prompt)
    
    def _build_system_prompt(self, novel_context: Dict[str, Any] = None) -> str:
        """Build the system prompt for the AI."""
        base_prompt = """You are a character creator for fiction writing. Take a simple character description and expand it into a detailed character profile.

IMPORTANT WRITING STYLE:
- Use simple, clear English
- Avoid flowery or overly dramatic language
- Don't use AI-typical words like "enigmatic", "mysterious aura", "piercing gaze", "steely determination"
- Write naturally, like a human would describe someone they know
- Be specific and practical, not poetic

You must respond with a JSON object containing these exact fields:
- "description": Clear physical and personality description (2-3 sentences, simple language)
- "age": Character's age (number, range like "30s", or "Old" - keep it simple)
- "occupation": Character's job or role (be specific and practical)
- "personality": Main personality traits and what drives them (2-3 sentences, natural language)
- "backstory": Character's background and important events (3-4 sentences, straightforward)
- "tags": Array of 3-5 simple, relevant keywords
- "role": Character's story role (choose one: "protagonist", "secondary protagonist", "villain", "antagonist", "supporting character", "minor character")

Make the character feel real and relatable. Focus on:
- What they do and why they do it
- How they act around others
- What happened in their past that matters
- What they want and what stops them

Write like you're describing a real person to a friend. Keep it simple and clear.

Respond ONLY with valid JSON. No extra text."""

        if novel_context:
            novel_info = f"""

NOVEL CONTEXT:
- Title: {novel_context.get('title', 'Unknown')}
- Genre: {novel_context.get('genre', 'Unknown')}
- Description: {novel_context.get('description', 'No description available')}

Make sure the character fits naturally into this world and genre."""
            base_prompt += novel_info
        
        return base_prompt
    
    def _build_user_prompt(self, name: str, user_prompt: str, novel_context: Dict[str, Any] = None) -> str:
        """Build the user prompt for the AI."""
        prompt = f"""Create a detailed character profile for:

CHARACTER NAME: {name}

USER DESCRIPTION: {user_prompt}"""

        if novel_context:
            prompt += f"""

This character exists in the novel "{novel_context.get('title', 'Unknown')}" which is a {novel_context.get('genre', 'Unknown')} story."""

        prompt += """

Please create a comprehensive character profile that expands on the user's description while staying true to their vision. Return only the JSON object."""

        return prompt
    
    def _parse_ai_response(self, ai_response: str, name: str, user_prompt: str) -> Dict[str, Any]:
        """Parse the AI response and extract character data."""
        try:
            # Try to extract JSON from the response
            ai_response = ai_response.strip()
            
            # Remove any markdown code blocks
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.startswith('```'):
                ai_response = ai_response[3:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            
            # Parse JSON
            character_data = json.loads(ai_response.strip())
            
            # Validate required fields
            required_fields = ['description', 'age', 'occupation', 'personality', 'backstory', 'tags', 'role']
            for field in required_fields:
                if field not in character_data:
                    if field == 'tags':
                        character_data[field] = []
                    elif field == 'role':
                        character_data[field] = 'supporting character'
                    else:
                        character_data[field] = ""
            
            # Ensure tags is a list
            if not isinstance(character_data.get('tags'), list):
                character_data['tags'] = []

            # Add occupation as a tag if it exists and isn't already in tags
            if character_data.get('occupation'):
                if character_data['occupation'] not in character_data['tags']:
                    character_data['tags'].append(character_data['occupation'])

            # Add metadata
            character_data['name'] = name
            character_data['ai_generated'] = True
            character_data['user_prompt'] = user_prompt

            return character_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"AI Response: {ai_response}")
            return self._create_fallback_character(name, user_prompt)
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._create_fallback_character(name, user_prompt)
    
    def _create_fallback_character(self, name: str, user_prompt: str) -> Dict[str, Any]:
        """Create a basic character when AI fails."""
        return {
            'name': name,
            'description': user_prompt or f"{name} is a character in this story.",
            'age': '',
            'occupation': '',
            'personality': '',
            'backstory': '',
            'tags': [],
            'role': 'supporting character',
            'ai_generated': False,
            'user_prompt': user_prompt,
            'fallback_created': True
        }
    
    def enhance_character(self, character_data: Dict[str, Any], enhancement_request: str) -> Dict[str, Any]:
        """
        Enhance an existing character with additional details.
        
        Args:
            character_data: Existing character data
            enhancement_request: What to enhance or add
            
        Returns:
            Enhanced character data
        """
        if not self.openrouter_client:
            logger.error("OpenRouter client not available for character enhancement")
            return character_data
        
        try:
            system_prompt = """You are enhancing an existing character. Take the current character data and the enhancement request, then return an improved version.

WRITING STYLE:
- Use simple, clear English
- Avoid flowery or dramatic language
- Don't use AI words like "enigmatic", "piercing", "steely"
- Write naturally and practically

Respond with a complete JSON object containing all character fields with enhancements applied. Keep existing good details and add/improve based on the request.

Required fields: description, age, occupation, personality, backstory, tags, role

Respond ONLY with valid JSON."""

            user_message = f"""Current Character:
{json.dumps(character_data, indent=2)}

Enhancement Request: {enhancement_request}

Please enhance this character based on the request while preserving their core identity."""

            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content
            enhanced_data = self._parse_ai_response(ai_response, character_data['name'], character_data.get('user_prompt', ''))
            
            # Preserve original metadata
            enhanced_data['ai_generated'] = True
            enhanced_data['enhanced'] = True
            enhanced_data['original_prompt'] = character_data.get('user_prompt', '')
            enhanced_data['enhancement_request'] = enhancement_request
            
            logger.info(f"Successfully enhanced character '{character_data['name']}'")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Character enhancement failed: {e}")
            return character_data
    
    def is_available(self) -> bool:
        """Check if the character creator is available."""
        return self.openrouter_client is not None
