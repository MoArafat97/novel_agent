"""
AI Location Creator Agent

This agent uses DeepSeek LLM to intelligently create detailed locations
from simple user prompts. It analyzes the input and populates all location
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

class LocationCreatorAgent:
    """
    AI agent that creates detailed locations from simple prompts using DeepSeek LLM.
    """
    
    def __init__(self):
        """Initialize the Location Creator Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Location creator will not work.")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client for DeepSeek LLM."""
        try:
            self.openrouter_client = openai.OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized for location creation")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None

    def is_available(self) -> bool:
        """Check if the location creator is available (has working OpenRouter client)."""
        return self.openrouter_client is not None
    
    def create_location(self, 
                       name: str, 
                       user_prompt: str, 
                       novel_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a detailed location from a simple prompt.
        
        Args:
            name: Location name
            user_prompt: User's description/prompt about the location
            novel_context: Context about the novel (title, genre, etc.)
            
        Returns:
            Dictionary with detailed location information
        """
        if not self.openrouter_client:
            logger.error("OpenRouter client not available")
            return self._create_fallback_location(name, user_prompt)
        
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
            location_data = self._parse_ai_response(ai_response, name, user_prompt)
            
            logger.info(f"Successfully created location '{name}' using AI")
            return location_data
            
        except Exception as e:
            logger.error(f"AI location creation failed: {e}")
            return self._create_fallback_location(name, user_prompt)
    
    def _build_system_prompt(self, novel_context: Dict[str, Any] = None) -> str:
        """Build the system prompt for location creation."""
        base_prompt = """You are an expert worldbuilding assistant specializing in creating detailed, immersive locations for novels. Your job is to take a simple location name and description and expand it into a rich, detailed location that feels authentic and engaging.

Key principles:
- Create vivid, specific details that bring the location to life
- Use simple, clear language - avoid purple prose and overly flowery descriptions
- Make locations feel lived-in and realistic within their context
- Consider the practical aspects: who lives/works there, what happens there, how it functions
- Avoid AI-prone words like "bustling", "vibrant", "sprawling", "majestic", "ancient"
- Focus on concrete, sensory details rather than abstract concepts
- Make the location serve the story - think about potential scenes and conflicts

You must respond with a JSON object containing these exact fields:
- "name": Location name (keep the provided name unless asked to change)
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
- "user_prompt": The original user input"""
        
        if novel_context:
            context_info = f"""

Novel Context:
- Title: {novel_context.get('title', 'Unknown')}
- Genre: {novel_context.get('genre', 'Unknown')}
- Description: {novel_context.get('description', 'No description available')}

Consider this context when creating the location to ensure it fits the world and story."""
            base_prompt += context_info
        
        return base_prompt
    
    def _build_user_prompt(self, name: str, user_prompt: str, novel_context: Dict[str, Any] = None) -> str:
        """Build the user prompt for location creation."""
        prompt = f"""Create a detailed location with the following information:

Location Name: {name}
User Description: {user_prompt}

Please create a complete location profile that expands on this basic information. Make it feel real and interesting while staying true to the user's vision."""
        
        return prompt
    
    def _parse_ai_response(self, ai_response: str, name: str, user_prompt: str) -> Dict[str, Any]:
        """Parse the AI response into structured location data."""
        try:
            # Try to extract JSON from the response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = ai_response[json_start:json_end]
            
            # Parse JSON
            location_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['name', 'description', 'type', 'climate', 'culture', 'history', 
                             'geography', 'economy', 'notable_features', 'tags']
            for field in required_fields:
                if field not in location_data:
                    if field == 'tags':
                        location_data[field] = []
                    elif field == 'type':
                        location_data[field] = 'Other'
                    else:
                        location_data[field] = ""
            
            # Ensure tags is a list
            if not isinstance(location_data.get('tags'), list):
                location_data['tags'] = []
            
            # Set metadata
            location_data['ai_generated'] = True
            location_data['user_prompt'] = user_prompt
            
            return location_data
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return self._create_fallback_location(name, user_prompt)
    
    def _create_fallback_location(self, name: str, user_prompt: str) -> Dict[str, Any]:
        """Create a basic location when AI fails."""
        return {
            'name': name,
            'description': f"A location called {name}. {user_prompt}",
            'type': 'Other',
            'climate': 'Temperate',
            'culture': 'Local customs and traditions',
            'history': f"The history of {name} is yet to be fully explored.",
            'geography': 'The physical features of this location are distinctive.',
            'economy': 'The local economy supports the community.',
            'notable_features': 'This location has unique characteristics.',
            'tags': [name.lower().replace(' ', '_')],
            'ai_generated': False,
            'user_prompt': user_prompt,
            'fallback_creation': True
        }
