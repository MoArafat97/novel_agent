"""
AI Character Editor Agent

This agent uses DeepSeek LLM to intelligently edit and update existing characters
based on natural language requests. It can modify specific fields or enhance
the entire character while maintaining consistency.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import openai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class CharacterEditorAgent:
    """
    AI agent that edits and updates existing characters using natural language requests.
    """
    
    def __init__(self):
        """Initialize the Character Editor Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Character editor will not work.")

        # Simple response cache for speed optimization
        self._response_cache = {}
        self._cache_max_size = 50
    
    def _init_openrouter(self):
        """Initialize OpenRouter client for chat completions."""
        try:
            self.openrouter_client = openai.OpenAI(
                base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
                api_key=self.openrouter_api_key
            )
            logger.info("Character Editor Agent initialized with DeepSeek")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter for character editing: {e}")
            self.openrouter_client = None
    
    def edit_character(self, 
                      current_character: Dict[str, Any],
                      edit_request: str,
                      novel_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Edit an existing character based on a natural language request.
        
        Args:
            current_character: Current character data
            edit_request: Natural language description of what to change
            novel_context: Context about the novel
            
        Returns:
            Updated character data
        """
        if not self.openrouter_client:
            logger.error("OpenRouter client not available")
            return current_character

        # Check cache first for speed
        cache_key = f"{current_character.get('id', 'unknown')}:{edit_request[:50]}"
        if cache_key in self._response_cache:
            logger.info("Using cached character edit response")
            return self._response_cache[cache_key]

        try:
            # Build the AI prompt
            system_prompt = self._build_edit_system_prompt(novel_context)
            user_message = self._build_edit_user_prompt(current_character, edit_request)
            
            # Call DeepSeek LLM with optimized settings for speed
            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Lower temperature for faster, more deterministic responses
                max_tokens=600,   # Further reduced tokens for speed
                top_p=0.8,       # More focused token selection
                timeout=15.0     # 15 second timeout to prevent hanging
            )
            
            # Parse the response
            ai_response = response.choices[0].message.content
            updated_character = self._parse_edit_response(ai_response, current_character, edit_request)

            # Cache the successful response for speed
            if len(self._response_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            self._response_cache[cache_key] = updated_character

            logger.info(f"Successfully edited character '{current_character.get('name', 'Unknown')}' using AI")
            return updated_character
            
        except Exception as e:
            logger.error(f"AI character editing failed: {e}")
            # Return original character with failure flag
            failed_character = current_character.copy()
            failed_character['ai_edit_failed'] = True
            failed_character['ai_edit_error'] = str(e)
            return failed_character
    
    def _build_edit_system_prompt(self, novel_context: Dict[str, Any] = None) -> str:
        """Build the system prompt for character editing."""
        base_prompt = """You are a character editor for fiction writing. Your job is to take an existing character and modify them based on the user's request.

IMPORTANT WRITING STYLE:
- Use simple, clear English
- Avoid flowery or overly dramatic language
- Don't use AI-typical words like "enigmatic", "mysterious aura", "piercing gaze", "steely determination"
- Write naturally, like a human would describe someone they know
- Be specific and practical, not poetic

EDITING RULES:
- Keep the character's core identity unless specifically asked to change it
- Only modify what the user requests - don't change unrelated fields
- Maintain consistency across all character fields
- If changing personality, update backstory to support it
- If changing backstory, ensure personality still makes sense
- Keep the same writing style as the original character

You must respond with a COMPLETE JSON object containing ALL these fields:
- "name": Character's name (keep unless asked to change)
- "description": Physical and personality description (2-3 sentences, simple language)
- "age": Character's age (number, range, or descriptive)
- "occupation": Character's job or role
- "personality": Main personality traits and motivations (2-3 sentences, natural language)
- "backstory": Character's background and history (3-4 sentences, straightforward)
- "tags": Array of 3-5 simple, relevant keywords
- "role": Story role (protagonist, secondary protagonist, villain, antagonist, supporting character, minor character)

Apply the requested changes while keeping everything else consistent and natural.

Respond ONLY with valid JSON. No extra text."""

        if novel_context:
            novel_info = f"""

NOVEL CONTEXT:
- Title: {novel_context.get('title', 'Unknown')}
- Genre: {novel_context.get('genre', 'Unknown')}
- Description: {novel_context.get('description', 'No description available')}

Make sure any changes fit naturally into this world and genre."""
            base_prompt += novel_info
        
        return base_prompt
    
    def _build_edit_user_prompt(self, current_character: Dict[str, Any], edit_request: str) -> str:
        """Build the user prompt for character editing."""
        # Remove sensitive fields from display
        display_character = {k: v for k, v in current_character.items() 
                           if k not in ['id', 'novel_id', 'created_at', 'updated_at', 'ai_generated', 'user_prompt']}
        
        prompt = f"""Current Character:
{json.dumps(display_character, indent=2)}

Edit Request: {edit_request}

Please apply the requested changes to this character. Keep everything else the same unless the change requires updating other fields for consistency. Return the complete updated character as JSON."""

        return prompt
    
    def _parse_edit_response(self, ai_response: str, original_character: Dict[str, Any], edit_request: str) -> Dict[str, Any]:
        """Parse the AI response and extract updated character data."""
        try:
            # Clean up the response
            ai_response = ai_response.strip()
            
            # Remove any markdown code blocks
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.startswith('```'):
                ai_response = ai_response[3:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            
            # Parse JSON
            updated_character = json.loads(ai_response.strip())
            
            # Validate required fields and preserve original if missing
            required_fields = ['name', 'description', 'age', 'occupation', 'personality', 'backstory', 'tags', 'role']
            for field in required_fields:
                if field not in updated_character or not updated_character[field]:
                    updated_character[field] = original_character.get(field, '')
            
            # Ensure tags is a list
            if not isinstance(updated_character.get('tags'), list):
                updated_character['tags'] = original_character.get('tags', [])

            # Add occupation as a tag if it exists and isn't already in tags
            if updated_character.get('occupation'):
                if updated_character['occupation'] not in updated_character['tags']:
                    updated_character['tags'].append(updated_character['occupation'])

            # Preserve important metadata
            updated_character['novel_id'] = original_character.get('novel_id')
            updated_character['created_at'] = original_character.get('created_at')
            updated_character['ai_generated'] = True
            updated_character['ai_edited'] = True
            updated_character['last_edit_request'] = edit_request

            return updated_character
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI edit response as JSON: {e}")
            logger.error(f"AI Response: {ai_response}")
            return original_character
        except Exception as e:
            logger.error(f"Error parsing AI edit response: {e}")
            return original_character
    
    def suggest_improvements(self, character_data: Dict[str, Any]) -> List[str]:
        """
        Suggest potential improvements for a character.
        
        Args:
            character_data: Current character data
            
        Returns:
            List of improvement suggestions
        """
        if not self.openrouter_client:
            return []
        
        try:
            system_prompt = """You are a character development expert. Analyze the given character and suggest 3-5 specific improvements that would make them more interesting, well-rounded, or better integrated into their story.

Focus on:
- Adding depth to personality or backstory
- Creating interesting conflicts or motivations
- Improving their role in the story
- Adding unique traits or quirks
- Strengthening their relationships with others

Respond with a simple JSON array of suggestion strings. Each suggestion should be a clear, actionable edit request.

Example: ["Add a secret fear that conflicts with their brave exterior", "Give them a mentor who betrayed them in the past"]

Respond ONLY with the JSON array."""

            user_message = f"""Character to analyze:
{json.dumps({k: v for k, v in character_data.items() if k not in ['id', 'novel_id', 'created_at', 'updated_at']}, indent=2)}

Please suggest improvements for this character."""

            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=800
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse suggestions
            if ai_response.startswith('```json'):
                ai_response = ai_response[7:]
            if ai_response.startswith('```'):
                ai_response = ai_response[3:]
            if ai_response.endswith('```'):
                ai_response = ai_response[:-3]
            
            suggestions = json.loads(ai_response.strip())
            
            if isinstance(suggestions, list):
                return suggestions[:5]  # Limit to 5 suggestions
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to generate character suggestions: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if the character editor is available."""
        return self.openrouter_client is not None
