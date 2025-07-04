"""
AI Location Editor Agent

This agent uses DeepSeek LLM to intelligently edit and update existing locations
based on natural language requests. It can modify specific fields or enhance
the entire location while maintaining consistency.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import openai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class LocationEditorAgent:
    """
    AI agent that edits and updates existing locations using natural language requests.
    """
    
    def __init__(self):
        """Initialize the Location Editor Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Location editor will not work.")

        # Simple response cache for speed optimization
        self._response_cache = {}
        self._cache_max_size = 50
    
    def _init_openrouter(self):
        """Initialize OpenRouter client for DeepSeek LLM."""
        try:
            self.openrouter_client = openai.OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized for location editing")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None

    def is_available(self) -> bool:
        """Check if the location editor is available (has working OpenRouter client)."""
        return self.openrouter_client is not None
    
    def edit_location(self, 
                     current_location: Dict[str, Any],
                     edit_request: str,
                     novel_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Edit an existing location based on a natural language request.
        
        Args:
            current_location: Current location data
            edit_request: Natural language description of what to change
            novel_context: Context about the novel
            
        Returns:
            Updated location data
        """
        if not self.openrouter_client:
            logger.error("OpenRouter client not available")
            return current_location

        # Check cache first for speed
        cache_key = f"{current_location.get('id', 'unknown')}:{edit_request[:50]}"
        if cache_key in self._response_cache:
            logger.info("Using cached location edit response")
            return self._response_cache[cache_key]

        try:
            # Build the AI prompt
            system_prompt = self._build_edit_system_prompt(novel_context)
            user_message = self._build_edit_user_prompt(current_location, edit_request)
            
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
            updated_location = self._parse_edit_response(ai_response, current_location, edit_request)

            # Cache the successful response for speed
            if len(self._response_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]
            self._response_cache[cache_key] = updated_location

            logger.info(f"Successfully edited location '{current_location.get('name', 'Unknown')}' using AI")
            return updated_location
            
        except Exception as e:
            logger.error(f"AI location editing failed: {e}")
            # Return original location with failure flag
            failed_location = current_location.copy()
            failed_location['ai_edit_failed'] = True
            failed_location['ai_edit_error'] = str(e)
            return failed_location
    
    def _build_edit_system_prompt(self, novel_context: Dict[str, Any] = None) -> str:
        """Build the system prompt for location editing."""
        base_prompt = """You are an expert worldbuilding assistant specializing in editing and improving locations for novels. Your job is to take an existing location and modify it based on the user's specific requests while maintaining consistency and quality.

Key principles:
- Make only the requested changes while keeping everything else consistent
- Use simple, clear language - avoid purple prose and overly flowery descriptions
- Ensure the location feels lived-in and realistic within its context
- Consider practical aspects: who lives/works there, what happens there, how it functions
- Avoid AI-prone words like "bustling", "vibrant", "sprawling", "majestic", "ancient"
- Focus on concrete, sensory details rather than abstract concepts
- Maintain the location's role in the story and world

You must respond with a COMPLETE JSON object containing ALL these fields:
- "name": Location name (keep unless asked to change)
- "description": Clear, vivid description of the location (2-3 sentences, simple language)
- "type": Location category (City, Town, Village, Building, Landmark, Natural Feature, Region, Other)
- "climate": Weather and environmental conditions
- "culture": Social aspects, customs, and way of life of inhabitants
- "history": Background and important past events (2-3 sentences, straightforward)
- "geography": Physical features, layout, and surroundings
- "economy": How the location sustains itself, trade, resources
- "notable_features": Unique or interesting aspects that make this location special
- "tags": Array of 3-5 simple, relevant keywords for searching
- "ai_generated": true
- "user_prompt": The original edit request

Apply the requested changes while keeping everything else consistent and natural."""
        
        if novel_context:
            context_info = f"""

Novel Context:
- Title: {novel_context.get('title', 'Unknown')}
- Genre: {novel_context.get('genre', 'Unknown')}
- Description: {novel_context.get('description', 'No description available')}

Consider this context when editing the location to ensure it fits the world and story."""
            base_prompt += context_info
        
        return base_prompt
    
    def _build_edit_user_prompt(self, current_location: Dict[str, Any], edit_request: str) -> str:
        """Build the user prompt for location editing."""
        # Clean the location data for display
        clean_location = {k: v for k, v in current_location.items() 
                         if k not in ['id', 'novel_id', 'created_at', 'updated_at']}
        
        prompt = f"""Current Location:
{json.dumps(clean_location, indent=2)}

Edit Request: {edit_request}

Please apply the requested changes to this location while keeping all other aspects consistent and well-integrated."""
        
        return prompt
    
    def _parse_edit_response(self, ai_response: str, original_location: Dict[str, Any], edit_request: str) -> Dict[str, Any]:
        """Parse the AI edit response into structured location data."""
        try:
            # Try to extract JSON from the response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = ai_response[json_start:json_end]
            
            # Parse JSON
            updated_location = json.loads(json_str)
            
            # Validate required fields and preserve original values if missing
            required_fields = ['name', 'description', 'type', 'climate', 'culture', 'history', 
                             'geography', 'economy', 'notable_features', 'tags']
            for field in required_fields:
                if field not in updated_location:
                    updated_location[field] = original_location.get(field, "")
            
            # Ensure tags is a list
            if not isinstance(updated_location.get('tags'), list):
                updated_location['tags'] = original_location.get('tags', [])
            
            # Preserve important metadata
            updated_location['ai_generated'] = True
            updated_location['user_prompt'] = edit_request
            
            # Preserve original metadata if it exists
            for key in ['id', 'novel_id', 'created_at']:
                if key in original_location:
                    updated_location[key] = original_location[key]
            
            return updated_location
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI edit response as JSON: {e}")
            logger.error(f"AI Response: {ai_response}")
            return original_location
        except Exception as e:
            logger.error(f"Error parsing AI edit response: {e}")
            return original_location
    
    def suggest_improvements(self, location_data: Dict[str, Any]) -> List[str]:
        """
        Suggest potential improvements for a location.
        
        Args:
            location_data: Current location data
            
        Returns:
            List of improvement suggestions
        """
        if not self.openrouter_client:
            return []
        
        try:
            system_prompt = """You are a worldbuilding expert. Analyze the given location and suggest 3-5 specific improvements that would make it more interesting, detailed, or better integrated into its story world.

Focus on:
- Adding depth to history or culture
- Creating interesting conflicts or tensions
- Improving the location's role in the story
- Adding unique features or characteristics
- Strengthening connections to other world elements
- Making it more immersive and believable

Respond with a simple JSON array of suggestion strings. Each suggestion should be a clear, actionable edit request.

Example: ["Add a hidden underground network beneath the city", "Describe the local festival that happens once a year"]

Respond ONLY with the JSON array."""

            user_message = f"""Location to analyze:
{json.dumps({k: v for k, v in location_data.items() if k not in ['id', 'novel_id', 'created_at', 'updated_at']}, indent=2)}

Please suggest improvements for this location."""

            response = self.openrouter_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            
            # Try to parse as JSON array
            try:
                suggestions = json.loads(ai_response.strip())
                if isinstance(suggestions, list):
                    return suggestions[:5]  # Limit to 5 suggestions
            except:
                pass
            
            # Fallback: split by lines and clean up
            lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
            suggestions = []
            for line in lines:
                # Remove common prefixes
                line = line.lstrip('- ').lstrip('• ').lstrip('* ')
                if line and len(line) > 10:  # Reasonable length
                    suggestions.append(line)
            
            return suggestions[:5]
            
        except Exception as e:
            logger.error(f"Failed to generate location improvement suggestions: {e}")
            return []
