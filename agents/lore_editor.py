"""
AI Lore Editor Agent

This agent uses DeepSeek LLM to intelligently edit and update existing lore entries
based on natural language requests. It can modify specific fields or enhance
the entire lore entry while maintaining consistency.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import openai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class LoreEditorAgent:
    """
    AI agent that edits and updates existing lore entries using natural language requests.
    """
    
    def __init__(self):
        """Initialize the Lore Editor Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Lore editor will not work.")

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
            logger.info("Lore Editor Agent initialized with DeepSeek")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter for lore editing: {e}")
            self.openrouter_client = None
    
    def edit_lore(self, 
                  current_lore: Dict[str, Any],
                  edit_request: str,
                  novel_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Edit an existing lore entry based on a natural language request.
        
        Args:
            current_lore: Current lore entry data
            edit_request: Natural language description of what to change
            novel_context: Context about the novel
            
        Returns:
            Updated lore entry data
        """
        if not self.openrouter_client:
            logger.error("OpenRouter client not available")
            return current_lore

        # Check cache first for speed
        cache_key = f"{current_lore.get('id', 'unknown')}:{edit_request[:50]}"
        if cache_key in self._response_cache:
            logger.info("Using cached lore edit response")
            return self._response_cache[cache_key]

        try:
            # Build the AI prompt
            system_prompt = self._build_edit_system_prompt(novel_context)
            user_message = self._build_edit_user_prompt(current_lore, edit_request)
            
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
            updated_lore = self._parse_edit_response(ai_response, current_lore, edit_request)

            # Cache the successful response for speed
            if len(self._response_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            self._response_cache[cache_key] = updated_lore

            logger.info(f"Successfully edited lore '{current_lore.get('title', 'Unknown')}' using AI")
            return updated_lore
            
        except Exception as e:
            logger.error(f"AI lore editing failed: {e}")
            # Return original lore with failure flag
            failed_lore = current_lore.copy()
            failed_lore['ai_edit_failed'] = True
            failed_lore['ai_edit_error'] = str(e)
            return failed_lore
    
    def _build_edit_system_prompt(self, novel_context: Dict[str, Any] = None) -> str:
        """Build the system prompt for lore editing."""
        base_prompt = """You are a worldbuilding editor for fiction writing. Your job is to take an existing lore entry and modify it based on the user's request.

IMPORTANT WRITING STYLE:
- Use simple, clear English
- Avoid flowery or overly dramatic language
- Don't use AI-typical words like "tapestry", "weave", "intricate", "delve", "unveil", "realm"
- Write naturally and practically, not poetically
- Be specific and concrete, not vague or mystical
- Focus on story-relevant details

EDITING RULES:
- Keep the lore's core concept unless specifically asked to change it
- Only modify what the user requests - don't change unrelated fields
- Maintain consistency across all lore fields
- If changing details, ensure significance and connections still make sense
- If changing category, update details to match the new category
- Keep the same writing style as the original lore entry
- Ensure the lore serves the story and characters

You must respond with a COMPLETE JSON object containing ALL these fields:
- "title": Lore entry title (keep unless asked to change)
- "category": One of: History, Culture, Magic, Technology, Geography, Politics, Religion, Economics, Legends, Other
- "description": Concise 1-2 sentence summary
- "details": Main detailed explanation (2-4 paragraphs, clear and practical)
- "significance": Why this lore matters to the world/story
- "connections": How this relates to other world elements
- "tags": Array of 3-6 relevant, searchable keywords

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
    
    def _build_edit_user_prompt(self, current_lore: Dict[str, Any], edit_request: str) -> str:
        """Build the user prompt for lore editing."""
        # Remove sensitive fields from display
        display_lore = {k: v for k, v in current_lore.items() 
                       if k not in ['id', 'novel_id', 'created_at', 'updated_at', 'ai_generated', 'user_prompt']}
        
        prompt = f"""Current Lore Entry:
{json.dumps(display_lore, indent=2)}

Edit Request: {edit_request}

Please apply the requested changes to this lore entry. Keep everything else the same unless the change requires updating other fields for consistency. Return the complete updated lore entry as JSON."""

        return prompt
    
    def _parse_edit_response(self, ai_response: str, original_lore: Dict[str, Any], edit_request: str) -> Dict[str, Any]:
        """Parse the AI response and extract updated lore data."""
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
            updated_lore = json.loads(ai_response.strip())
            
            # Validate required fields and preserve original if missing
            required_fields = ['title', 'category', 'description', 'details', 'significance', 'connections', 'tags']
            for field in required_fields:
                if field not in updated_lore or not updated_lore[field]:
                    updated_lore[field] = original_lore.get(field, '')
            
            # Ensure tags is a list
            if not isinstance(updated_lore.get('tags'), list):
                updated_lore['tags'] = original_lore.get('tags', [])
            
            # Validate category
            valid_categories = ['History', 'Culture', 'Magic', 'Technology', 'Geography', 
                             'Politics', 'Religion', 'Economics', 'Legends', 'Other']
            if updated_lore.get('category') not in valid_categories:
                updated_lore['category'] = original_lore.get('category', 'Other')

            # Preserve important metadata
            updated_lore['novel_id'] = original_lore.get('novel_id')
            updated_lore['created_at'] = original_lore.get('created_at')
            updated_lore['ai_generated'] = True
            updated_lore['ai_edited'] = True
            updated_lore['last_edit_request'] = edit_request

            return updated_lore
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI edit response as JSON: {e}")
            logger.error(f"AI Response: {ai_response}")
            return original_lore
        except Exception as e:
            logger.error(f"Error parsing AI edit response: {e}")
            return original_lore
    
    def suggest_improvements(self, lore_data: Dict[str, Any]) -> List[str]:
        """
        Suggest potential improvements for a lore entry.
        
        Args:
            lore_data: Current lore entry data
            
        Returns:
            List of improvement suggestions
        """
        if not self.openrouter_client:
            return []
        
        try:
            system_prompt = """You are a worldbuilding expert. Analyze the given lore entry and suggest 3-5 specific improvements that would make it more interesting, detailed, or better integrated into the story world.

Focus on:
- Adding depth to the historical or cultural context
- Creating interesting conflicts or consequences
- Improving connections to characters or plot
- Adding unique details or mechanics
- Strengthening the lore's impact on the world
- Making it more story-relevant

Respond with a simple JSON array of suggestion strings. Each suggestion should be a clear, actionable edit request.

Example: ["Add details about how this affects daily life for common people", "Explain the political consequences of this event"]

Respond ONLY with the JSON array."""

            user_message = f"""Lore entry to analyze:
{json.dumps({k: v for k, v in lore_data.items() if k not in ['id', 'novel_id', 'created_at', 'updated_at']}, indent=2)}

Please suggest improvements for this lore entry."""

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
            logger.error(f"Failed to generate lore suggestions: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if the lore editor is available."""
        return self.openrouter_client is not None
