"""
Genre-Specific Prompts for Entity Classification

This module provides domain-adapted prompts for different genres to improve
classification accuracy by using genre-specific examples and terminology.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class Genre(Enum):
    """Supported genres for prompt customization."""
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    HISTORICAL = "historical"
    CONTEMPORARY = "contemporary"
    HORROR = "horror"
    THRILLER = "thriller"
    ADVENTURE = "adventure"
    GENERAL = "general"  # Default fallback


class GenreSpecificPrompts:
    """
    Provides genre-adapted prompts for better entity classification accuracy.
    """
    
    def __init__(self):
        """Initialize genre-specific prompt templates."""
        self.genre_prompts = self._initialize_genre_prompts()
        self.genre_examples = self._initialize_genre_examples()
        self.genre_keywords = self._initialize_genre_keywords()
    
    def get_system_prompt(self, genre: Genre = Genre.GENERAL) -> str:
        """Get genre-specific system prompt for entity classification."""
        base_prompt = """Classify entities as: characters (people/beings), locations (places), lore (concepts/events), or null (reject).

TYPES:"""
        
        # Add genre-specific type definitions
        type_definitions = self.genre_prompts[genre]["type_definitions"]
        
        prompt_parts = [
            base_prompt,
            type_definitions,
            "",
            "RULES:",
            "1. Person/being → characters",
            "2. Place → locations", 
            "3. Concept/event/organization → lore",
            "4. Unclear/invalid → null",
            "",
            "JSON format:",
            '{"entity_type": "characters|locations|lore|null", "confidence": 0.0-1.0, "reasoning": "brief explanation", "recommendation": "accept|review|reject"}',
            "",
            "Be conservative - reject unclear entities."
        ]
        
        return "\n".join(prompt_parts)
    
    def get_batch_system_prompt(self, genre: Genre = Genre.GENERAL) -> str:
        """Get genre-specific batch system prompt."""
        type_definitions = self.genre_prompts[genre]["type_definitions"]
        
        return f"""Classify entities as: characters (people), locations (places), lore (concepts), or null (reject).

{type_definitions}

Return JSON array in same order:
[{{"entity_type": "characters|locations|lore|null", "confidence": 0.0-1.0, "reasoning": "brief", "recommendation": "accept|review|reject"}}]

Use null for invalid entities."""
    
    def detect_genre(self, content: str, novel_metadata: Dict[str, Any] = None) -> Genre:
        """
        Detect the most likely genre based on content and metadata.
        
        Args:
            content: Text content to analyze
            novel_metadata: Optional metadata about the novel
            
        Returns:
            Detected genre
        """
        # Check explicit genre from metadata first
        if novel_metadata and 'genre' in novel_metadata:
            genre_str = novel_metadata['genre'].lower()
            for genre in Genre:
                if genre.value in genre_str or genre.name.lower() in genre_str:
                    return genre
        
        # Analyze content for genre indicators
        content_lower = content.lower()
        genre_scores = {}
        
        for genre, keywords in self.genre_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                genre_scores[genre] = score
        
        # Return genre with highest score, or GENERAL if no clear match
        if genre_scores:
            best_genre = max(genre_scores, key=genre_scores.get)
            logger.debug(f"Detected genre: {best_genre.value} (score: {genre_scores[best_genre]})")
            return best_genre
        
        return Genre.GENERAL
    
    def _initialize_genre_prompts(self) -> Dict[Genre, Dict[str, str]]:
        """Initialize genre-specific prompt templates."""
        return {
            Genre.FANTASY: {
                "type_definitions": """- characters: People, magical beings, dragons, elves, wizards, gods, spirits
- locations: Kingdoms, realms, castles, magical places, dungeons, forests, temples
- lore: Magic systems, spells, prophecies, ancient artifacts, guilds, religions, legends"""
            },
            
            Genre.SCIENCE_FICTION: {
                "type_definitions": """- characters: Humans, aliens, androids, AI entities, cyborgs, space travelers
- locations: Planets, space stations, starships, colonies, cities, laboratories
- lore: Technologies, corporations, governments, alien species, scientific concepts, protocols"""
            },
            
            Genre.MYSTERY: {
                "type_definitions": """- characters: Detectives, suspects, witnesses, victims, investigators, criminals
- locations: Crime scenes, police stations, courtrooms, cities, neighborhoods, buildings
- lore: Cases, evidence, methods, organizations, legal concepts, investigative procedures"""
            },
            
            Genre.ROMANCE: {
                "type_definitions": """- characters: Lovers, partners, family members, friends, rivals in love
- locations: Cities, homes, romantic venues, workplaces, social settings, travel destinations
- lore: Relationships, social events, cultural traditions, family histories, social circles"""
            },
            
            Genre.HISTORICAL: {
                "type_definitions": """- characters: Historical figures, nobles, commoners, soldiers, clergy, merchants
- locations: Historical places, cities, countries, battlefields, palaces, churches
- lore: Historical events, wars, political movements, cultural practices, social institutions"""
            },
            
            Genre.HORROR: {
                "type_definitions": """- characters: Humans, monsters, ghosts, demons, supernatural entities, victims
- locations: Haunted places, dark settings, isolated locations, supernatural realms
- lore: Curses, supernatural phenomena, occult practices, horror concepts, dark rituals"""
            },
            
            Genre.THRILLER: {
                "type_definitions": """- characters: Protagonists, antagonists, agents, criminals, law enforcement, civilians
- locations: Cities, safe houses, chase locations, government facilities, international settings
- lore: Conspiracies, operations, organizations, technologies, political situations, threats"""
            },
            
            Genre.GENERAL: {
                "type_definitions": """- characters: People, sentient beings (e.g., John, Mary, Captain Smith)
- locations: Places, geographical features (e.g., New York, Central Park, Main Street)
- lore: Concepts, events, organizations, artifacts (e.g., The Revolution, Tech Corp, The Agreement)"""
            }
        }
    
    def _initialize_genre_examples(self) -> Dict[Genre, Dict[str, List[str]]]:
        """Initialize genre-specific examples."""
        return {
            Genre.FANTASY: {
                "characters": ["Gandalf", "Aragorn", "Dragon Lord", "Elven Queen", "Dark Wizard"],
                "locations": ["Middle-earth", "Rivendell", "Mordor", "The Shire", "Crystal Cave"],
                "lore": ["The One Ring", "Order of Mages", "Ancient Prophecy", "Fire Magic", "The Great War"]
            },
            
            Genre.SCIENCE_FICTION: {
                "characters": ["Captain Kirk", "Data", "Alien Ambassador", "AI-7", "Space Marine"],
                "locations": ["Enterprise", "Mars Colony", "Space Station Alpha", "Cybertron", "The Bridge"],
                "lore": ["Warp Drive", "Federation", "Prime Directive", "Quantum Physics", "Time Travel"]
            },
            
            Genre.MYSTERY: {
                "characters": ["Detective Holmes", "Inspector Watson", "The Suspect", "Witness Smith", "Victim Jones"],
                "locations": ["Crime Scene", "Police Station", "Courthouse", "Baker Street", "The Library"],
                "lore": ["The Case", "Evidence", "Fingerprints", "Scotland Yard", "The Investigation"]
            }
        }
    
    def _initialize_genre_keywords(self) -> Dict[Genre, List[str]]:
        """Initialize genre detection keywords."""
        return {
            Genre.FANTASY: [
                "magic", "wizard", "dragon", "elf", "dwarf", "orc", "spell", "enchanted", "kingdom", 
                "quest", "prophecy", "sword", "castle", "realm", "potion", "fairy", "troll", "goblin"
            ],
            
            Genre.SCIENCE_FICTION: [
                "space", "alien", "robot", "android", "laser", "starship", "planet", "galaxy", 
                "technology", "future", "cybernetic", "quantum", "warp", "colony", "federation", "ai"
            ],
            
            Genre.MYSTERY: [
                "detective", "murder", "crime", "investigation", "suspect", "evidence", "clue", 
                "police", "victim", "witness", "case", "solve", "mystery", "criminal", "forensic"
            ],
            
            Genre.ROMANCE: [
                "love", "romance", "heart", "kiss", "wedding", "relationship", "boyfriend", 
                "girlfriend", "husband", "wife", "date", "passion", "attraction", "couple"
            ],
            
            Genre.HISTORICAL: [
                "century", "war", "king", "queen", "empire", "revolution", "battle", "ancient", 
                "medieval", "victorian", "colonial", "dynasty", "throne", "nobility", "peasant"
            ],
            
            Genre.HORROR: [
                "ghost", "demon", "monster", "haunted", "curse", "evil", "dark", "blood", 
                "death", "nightmare", "terror", "supernatural", "occult", "zombie", "vampire"
            ],
            
            Genre.THRILLER: [
                "chase", "escape", "danger", "threat", "conspiracy", "agent", "spy", "mission", 
                "operation", "government", "secret", "assassination", "terrorist", "pursuit"
            ]
        }


# Global instance
_genre_prompts = None

def get_genre_prompts() -> GenreSpecificPrompts:
    """Get the global genre-specific prompts instance."""
    global _genre_prompts
    if _genre_prompts is None:
        _genre_prompts = GenreSpecificPrompts()
    return _genre_prompts
