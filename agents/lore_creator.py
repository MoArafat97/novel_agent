"""
AI Lore Creator Agent

This agent uses DeepSeek LLM to intelligently create detailed lore entries
from simple user prompts. It analyzes the input and populates all lore
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

class LoreCreatorAgent:
    """
    AI agent that creates detailed lore entries from simple prompts using DeepSeek LLM.
    """
    
    def __init__(self):
        """Initialize the Lore Creator Agent."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.chat_model = os.getenv('CHAT_MODEL', 'deepseek/deepseek-chat:free')
        self.openrouter_client = None
        
        if self.openrouter_api_key:
            self._init_openrouter()
        else:
            logger.warning("No OpenRouter API key found. Lore creator will not work.")
    
    def _init_openrouter(self):
        """Initialize OpenRouter client."""
        try:
            self.openrouter_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
            logger.info("OpenRouter client initialized for lore creation")
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {e}")
            self.openrouter_client = None
    
    def create_lore(self, 
                   title: str, 
                   user_prompt: str, 
                   novel_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a detailed lore entry from a simple prompt.
        
        Args:
            title: Lore entry title
            user_prompt: User's description/prompt about the lore
            novel_context: Context about the novel (title, genre, etc.)
            
        Returns:
            Dictionary with detailed lore information
        """
        if not self.openrouter_client:
            logger.error("OpenRouter client not available")
            return self._create_fallback_lore(title, user_prompt)
        
        try:
            # Build the AI prompt
            system_prompt = self._build_system_prompt(novel_context)
            user_message = self._build_user_prompt(title, user_prompt, novel_context)
            
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
            lore_data = self._parse_ai_response(ai_response, title, user_prompt)
            
            logger.info(f"Successfully created lore '{title}' using AI")
            return lore_data
            
        except Exception as e:
            logger.error(f"AI lore creation failed: {e}")
            return self._create_fallback_lore(title, user_prompt)
    
    def _build_system_prompt(self, novel_context: Dict[str, Any] = None) -> str:
        """Build the system prompt for lore creation."""
        base_prompt = """You are an expert worldbuilding assistant specializing in creating rich, detailed lore entries for fictional worlds. Your task is to take a simple lore concept and expand it into a comprehensive, engaging entry.

IMPORTANT GUIDELINES:
- Write in simple, clear English - avoid purple prose and overly flowery language
- Avoid AI-prone words like "realm", "tapestry", "weave", "intricate", "delve", "unveil"
- Focus on practical, concrete details rather than vague mysticism
- Make the lore feel lived-in and realistic within the fictional context
- Ensure consistency with the novel's genre and tone
- Create lore that serves the story and characters

LORE CATEGORIES TO CONSIDER:
- History: Past events, wars, discoveries, cultural shifts
- Culture: Traditions, customs, social structures, beliefs
- Magic/Technology: How supernatural or advanced systems work
- Geography: Important places, natural features, settlements
- Politics: Governments, factions, power structures
- Religion: Belief systems, deities, spiritual practices
- Economics: Trade, currency, resources, commerce
- Legends: Myths, folklore, famous figures, stories

You must respond with a JSON object containing these fields:
{
    "title": "The lore entry title",
    "category": "One of: History, Culture, Magic, Technology, Geography, Politics, Religion, Economics, Legends, Other",
    "description": "A concise 1-2 sentence summary",
    "details": "The main detailed explanation (2-4 paragraphs)",
    "significance": "Why this lore matters to the world/story",
    "connections": "How this relates to other world elements",
    "tags": ["relevant", "searchable", "keywords"],
    "ai_generated": true,
    "user_prompt": "original user input"
}"""

        if novel_context:
            context_info = f"""
NOVEL CONTEXT:
- Title: {novel_context.get('title', 'Unknown')}
- Genre: {novel_context.get('genre', 'Unknown')}
- Description: {novel_context.get('description', 'No description available')}

Ensure the lore fits naturally within this novel's world and genre."""
            base_prompt += context_info
        
        return base_prompt
    
    def _build_user_prompt(self, title: str, user_prompt: str, novel_context: Dict[str, Any] = None) -> str:
        """Build the user prompt for lore creation."""
        prompt = f"""Create a detailed lore entry with the title "{title}".

User's concept/description: {user_prompt}

Please expand this into a rich, detailed lore entry that would fit well in the novel's world. Focus on making it practical and story-relevant rather than just decorative worldbuilding."""
        
        return prompt
    
    def _parse_ai_response(self, ai_response: str, title: str, user_prompt: str) -> Dict[str, Any]:
        """Parse the AI response into structured lore data."""
        try:
            # Try to extract JSON from the response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                lore_data = json.loads(json_str)
                
                # Ensure required fields exist
                lore_data['title'] = lore_data.get('title', title)
                lore_data['ai_generated'] = True
                lore_data['user_prompt'] = user_prompt
                
                # Validate category
                valid_categories = ['History', 'Culture', 'Magic', 'Technology', 'Geography', 
                                 'Politics', 'Religion', 'Economics', 'Legends', 'Other']
                if lore_data.get('category') not in valid_categories:
                    lore_data['category'] = 'Other'
                
                # Ensure tags is a list
                if not isinstance(lore_data.get('tags'), list):
                    lore_data['tags'] = []
                
                return lore_data
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Fallback parsing - extract key information manually
            return self._manual_parse_fallback(ai_response, title, user_prompt)
    
    def _manual_parse_fallback(self, ai_response: str, title: str, user_prompt: str) -> Dict[str, Any]:
        """Manually parse AI response when JSON parsing fails."""
        # Basic fallback - use the AI response as details
        return {
            'title': title,
            'category': 'Other',
            'description': f"Lore entry about {title}",
            'details': ai_response[:1000],  # Limit length
            'significance': 'Important worldbuilding element',
            'connections': 'Connected to the broader world',
            'tags': [title.lower().replace(' ', '_')],
            'ai_generated': True,
            'user_prompt': user_prompt,
            'parsing_fallback': True
        }
    
    def _create_fallback_lore(self, title: str, user_prompt: str) -> Dict[str, Any]:
        """Create a basic lore entry when AI fails."""
        return {
            'title': title,
            'category': 'Other',
            'description': user_prompt or f"Lore entry about {title}.",
            'details': user_prompt or f"This is a lore entry about {title}. More details to be added.",
            'significance': 'Part of the world\'s background',
            'connections': 'Connected to other world elements',
            'tags': [],
            'ai_generated': False,
            'user_prompt': user_prompt,
            'fallback_created': True
        }
    
    def enhance_lore(self, 
                    lore_data: Dict[str, Any], 
                    enhancement_request: str,
                    novel_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance an existing lore entry based on user request.
        
        Args:
            lore_data: Current lore data
            enhancement_request: What the user wants to change/improve
            novel_context: Context about the novel
            
        Returns:
            Enhanced lore data
        """
        if not self.openrouter_client:
            logger.error("OpenRouter client not available for enhancement")
            return lore_data
        
        try:
            system_prompt = self._build_enhancement_system_prompt(novel_context)
            user_message = self._build_enhancement_user_prompt(lore_data, enhancement_request)
            
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
            enhanced_data = self._parse_ai_response(ai_response, lore_data['title'], lore_data.get('user_prompt', ''))
            
            # Preserve original metadata
            enhanced_data['ai_generated'] = True
            enhanced_data['enhanced'] = True
            enhanced_data['original_prompt'] = lore_data.get('user_prompt', '')
            enhanced_data['enhancement_request'] = enhancement_request
            
            logger.info(f"Successfully enhanced lore '{lore_data['title']}'")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Lore enhancement failed: {e}")
            return lore_data
    
    def _build_enhancement_system_prompt(self, novel_context: Dict[str, Any] = None) -> str:
        """Build system prompt for lore enhancement."""
        prompt = """You are enhancing an existing lore entry based on user feedback. Maintain the core concept while improving or modifying specific aspects as requested.

Follow the same guidelines as lore creation:
- Simple, clear English
- Avoid purple prose and AI-prone words
- Focus on practical, story-relevant details
- Maintain consistency with the novel's world

Return the enhanced lore as a complete JSON object with all required fields."""
        
        if novel_context:
            prompt += f"\n\nNOVEL CONTEXT: {novel_context.get('title')} ({novel_context.get('genre')})"
        
        return prompt
    
    def _build_enhancement_user_prompt(self, lore_data: Dict[str, Any], enhancement_request: str) -> str:
        """Build user prompt for lore enhancement."""
        return f"""Current lore entry:
Title: {lore_data.get('title')}
Category: {lore_data.get('category')}
Description: {lore_data.get('description')}
Details: {lore_data.get('details')}

Enhancement request: {enhancement_request}

Please provide the enhanced version as a complete JSON object."""
    
    def is_available(self) -> bool:
        """Check if the lore creator is available."""
        return self.openrouter_client is not None
