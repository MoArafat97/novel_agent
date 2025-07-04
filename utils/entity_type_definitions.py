"""
Entity Type Definitions for Cross-Reference System

This module provides comprehensive definitions of what constitutes valid entities
in the worldbuilding system, with semantic understanding and validation rules.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class EntityTypeDefinition:
    """Definition of an entity type with validation rules."""
    name: str
    description: str
    semantic_definition: str
    positive_indicators: List[str]
    negative_indicators: List[str]
    context_patterns: List[str]
    validation_rules: List[str]
    examples: List[str]
    counter_examples: List[str]


class EntityTypeDefinitions:
    """
    Comprehensive entity type definitions with semantic understanding.
    """
    
    def __init__(self):
        self.definitions = self._create_definitions()
    
    def _create_definitions(self) -> Dict[str, EntityTypeDefinition]:
        """Create comprehensive entity type definitions."""
        
        return {
            'characters': EntityTypeDefinition(
                name='characters',
                description='People, sentient beings, and named individuals with agency',
                semantic_definition="""
                A CHARACTER is a sentient being with agency - someone who can think, act, make decisions, 
                and interact with the world. Characters have personalities, motivations, and can be the 
                subject of actions. They are individuals, not groups or concepts.
                
                Key characteristics:
                - Has consciousness and agency
                - Can perform actions and make decisions  
                - Has a distinct identity and personality
                - Can be referenced with personal pronouns (he/she/they)
                - Exists as an individual entity, not a collective
                """,
                positive_indicators=[
                    # Titles and roles
                    'king', 'queen', 'prince', 'princess', 'lord', 'lady', 'duke', 'duchess',
                    'count', 'countess', 'baron', 'baroness', 'knight', 'sir', 'dame',
                    'captain', 'general', 'admiral', 'commander', 'lieutenant',
                    
                    # Professions and occupations
                    'wizard', 'mage', 'sorcerer', 'witch', 'warlock', 'priest', 'cleric',
                    'paladin', 'monk', 'druid', 'ranger', 'bard', 'rogue', 'thief',
                    'assassin', 'warrior', 'fighter', 'archer', 'guardian',
                    'merchant', 'trader', 'blacksmith', 'innkeeper', 'shopkeeper',
                    'scholar', 'sage', 'scribe', 'healer', 'doctor', 'alchemist',
                    'farmer', 'hunter', 'fisherman', 'sailor', 'soldier', 'guard',
                    
                    # Character descriptors
                    'hero', 'villain', 'protagonist', 'antagonist', 'ally', 'enemy',
                    'friend', 'companion', 'mentor', 'student', 'master', 'apprentice',
                    'leader', 'follower', 'rebel', 'loyalist',
                    
                    # Fantasy races and beings
                    'elf', 'dwarf', 'human', 'halfling', 'orc', 'goblin', 'troll',
                    'dragon', 'demon', 'angel', 'spirit', 'ghost', 'vampire',
                    'werewolf', 'giant', 'fairy', 'pixie'
                ],
                negative_indicators=[
                    # Time and temporal concepts
                    'annual', 'yearly', 'monthly', 'daily', 'century', 'decade',
                    'season', 'winter', 'summer', 'spring', 'autumn', 'fall',
                    'morning', 'evening', 'night', 'day', 'week', 'moment',
                    
                    # Abstract concepts
                    'founded', 'established', 'created', 'built', 'destroyed',
                    'beginning', 'end', 'start', 'finish', 'origin', 'conclusion',
                    'concept', 'idea', 'theory', 'principle', 'rule', 'law',
                    
                    # Adjectives and descriptors
                    'cold', 'hot', 'warm', 'cool', 'big', 'small', 'large', 'tiny',
                    'old', 'new', 'ancient', 'modern', 'recent', 'past', 'future',
                    'good', 'bad', 'evil', 'holy', 'dark', 'light', 'bright',
                    
                    # Common words
                    'the', 'and', 'or', 'but', 'with', 'from', 'to', 'of', 'in',
                    'on', 'at', 'by', 'for', 'as', 'is', 'was', 'are', 'were'
                ],
                context_patterns=[
                    r'\b(?:said|spoke|told|asked|replied|answered|whispered|shouted)\b',
                    r'\b(?:he|she|they|him|her|them|his|hers|their)\b',
                    r'\b(?:walked|ran|fought|killed|saved|helped|loved|hated)\b',
                    r'\b(?:born|died|lived|grew up|trained|studied|learned)\b',
                    r'\b(?:thinks|believes|knows|remembers|feels|wants|needs)\b'
                ],
                validation_rules=[
                    'Must be a proper noun (capitalized)',
                    'Must be at least 2 characters long',
                    'Cannot be a common English word',
                    'Should have context indicating personhood or agency',
                    'Should not be a time period, location, or abstract concept'
                ],
                examples=[
                    'King Arthur', 'Gandalf the Grey', 'Princess Leia', 'Captain Hook',
                    'Sherlock Holmes', 'Harry Potter', 'Frodo Baggins', 'Aragorn',
                    'The Wizard of Oz', 'Lady Macbeth', 'Sir Lancelot', 'Robin Hood'
                ],
                counter_examples=[
                    'Annual', 'Founded', 'Cold', 'Military', 'The', 'Kingdom',
                    'Winter', 'Battle', 'Magic', 'Ancient', 'Royal', 'Noble'
                ]
            ),
            
            'locations': EntityTypeDefinition(
                name='locations',
                description='Places, geographical features, and spatial entities',
                semantic_definition="""
                A LOCATION is a place that exists in physical or conceptual space - somewhere that 
                can be visited, referenced spatially, or serves as a setting for events. Locations 
                have spatial properties and can contain other entities.
                
                Key characteristics:
                - Has spatial properties (can be visited, traveled to)
                - Serves as a setting or backdrop for events
                - Can contain other entities (characters, objects)
                - Has geographical or architectural properties
                - Can be referenced with spatial prepositions (in, at, near, from)
                """,
                positive_indicators=[
                    # Settlements
                    'city', 'town', 'village', 'hamlet', 'settlement', 'colony',
                    'capital', 'metropolis', 'port', 'harbor', 'outpost',
                    
                    # Political entities
                    'kingdom', 'empire', 'realm', 'nation', 'country', 'land',
                    'territory', 'province', 'region', 'district', 'county',
                    'duchy', 'principality', 'republic', 'federation',
                    
                    # Buildings and structures
                    'castle', 'fortress', 'tower', 'palace', 'manor', 'hall',
                    'temple', 'shrine', 'cathedral', 'church', 'monastery',
                    'tavern', 'inn', 'shop', 'market', 'arena', 'colosseum',
                    'library', 'academy', 'school', 'university', 'hospital',
                    'prison', 'dungeon', 'vault', 'tomb', 'crypt',
                    'hogwarts', 'winterfell', 'camelot', 'rivendell',  # Famous fictional locations
                    
                    # Natural features
                    'forest', 'woods', 'jungle', 'mountain', 'hill', 'peak',
                    'valley', 'canyon', 'gorge', 'cliff', 'cave', 'cavern',
                    'river', 'stream', 'lake', 'pond', 'sea', 'ocean',
                    'desert', 'plains', 'meadow', 'field', 'swamp', 'marsh',
                    'island', 'peninsula', 'continent', 'coast', 'shore',
                    
                    # Urban features
                    'street', 'road', 'avenue', 'boulevard', 'square', 'plaza',
                    'quarter', 'ward', 'district', 'neighborhood', 'suburb',
                    'bridge', 'gate', 'wall', 'tower', 'fountain', 'garden'
                ],
                negative_indicators=[
                    # People and characters
                    'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers',
                    'said', 'spoke', 'thought', 'felt', 'believed', 'knew',
                    
                    # Abstract concepts
                    'idea', 'concept', 'theory', 'belief', 'tradition', 'custom',
                    'magic', 'spell', 'curse', 'blessing', 'prophecy', 'legend',
                    
                    # Time periods
                    'year', 'month', 'day', 'century', 'age', 'era', 'period',
                    'season', 'time', 'moment', 'instant', 'duration'
                ],
                context_patterns=[
                    r'\b(?:in|at|near|from|to|towards|through|across|over|under)\b',
                    r'\b(?:located|situated|found|built|constructed|established)\b',
                    r'\b(?:traveled|journeyed|went|visited|arrived|departed)\b',
                    r'\b(?:north|south|east|west|northern|southern|eastern|western)\b',
                    r'\b(?:inside|outside|within|beyond|beside|behind|before)\b'
                ],
                validation_rules=[
                    'Must be a proper noun (capitalized)',
                    'Must be at least 2 characters long',
                    'Should have spatial context or geographical indicators',
                    'Should not refer to people, time periods, or abstract concepts',
                    'Can be referenced with spatial prepositions'
                ],
                examples=[
                    'Middle-earth', 'Hogwarts', 'King\'s Landing', 'Rivendell',
                    'The Shire', 'Mordor', 'Winterfell', 'Camelot',
                    'The Forbidden Forest', 'Dragon Mountain', 'Crystal Lake'
                ],
                counter_examples=[
                    'Tom', 'Annual', 'Magic', 'Battle', 'Founded', 'Ancient',
                    'Cold', 'Military', 'Royal', 'Noble', 'Sacred', 'Holy'
                ]
            ),
            
            'lore': EntityTypeDefinition(
                name='lore',
                description='Concepts, events, organizations, artifacts, and knowledge',
                semantic_definition="""
                LORE encompasses abstract concepts, historical events, organizations, artifacts, 
                knowledge systems, and cultural elements that exist in the world but are not 
                physical places or individual people. Lore represents the intangible aspects 
                of worldbuilding.
                
                Key characteristics:
                - Abstract or conceptual in nature
                - Represents knowledge, culture, or history
                - Can be learned, taught, or passed down
                - Influences the world and its inhabitants
                - Not a specific person or physical location
                """,
                positive_indicators=[
                    # Events and conflicts
                    'war', 'battle', 'conflict', 'siege', 'campaign', 'crusade',
                    'revolution', 'rebellion', 'uprising', 'invasion', 'conquest',
                    'treaty', 'alliance', 'pact', 'agreement', 'accord',
                    
                    # Organizations and groups
                    'guild', 'order', 'brotherhood', 'sisterhood', 'society',
                    'organization', 'faction', 'clan', 'tribe', 'house',
                    'council', 'assembly', 'parliament', 'court', 'senate',
                    'army', 'legion', 'regiment', 'company', 'squad',
                    
                    # Magic and supernatural
                    'magic', 'spell', 'enchantment', 'curse', 'blessing',
                    'ritual', 'ceremony', 'incantation', 'potion', 'elixir',
                    'artifact', 'relic', 'talisman', 'amulet', 'charm',
                    
                    # Knowledge and culture
                    'legend', 'myth', 'prophecy', 'tale', 'story', 'saga',
                    'tradition', 'custom', 'law', 'rule', 'decree', 'edict',
                    'religion', 'faith', 'belief', 'doctrine', 'creed',
                    'philosophy', 'ideology', 'principle', 'code', 'oath',
                    
                    # Titles and concepts
                    'crown', 'throne', 'scepter', 'sword', 'shield', 'banner',
                    'title', 'rank', 'position', 'office', 'role', 'duty'
                ],
                negative_indicators=[
                    # Specific people
                    'he', 'she', 'they', 'him', 'her', 'them', 'his', 'hers',
                    'said', 'spoke', 'thought', 'walked', 'ran', 'fought',
                    
                    # Specific places
                    'city', 'town', 'castle', 'forest', 'mountain', 'river',
                    'in', 'at', 'near', 'located', 'situated', 'built',
                    
                    # Common adjectives
                    'big', 'small', 'old', 'new', 'good', 'bad', 'cold', 'hot'
                ],
                context_patterns=[
                    r'\b(?:legend|myth|story|tale|prophecy|tradition)\b',
                    r'\b(?:magic|magical|enchanted|cursed|blessed|sacred)\b',
                    r'\b(?:ancient|old|forgotten|lost|hidden|secret)\b',
                    r'\b(?:power|knowledge|wisdom|truth|mystery|secret)\b',
                    r'\b(?:founded|established|created|formed|began|started)\b'
                ],
                validation_rules=[
                    'Must represent an abstract concept, event, or organization',
                    'Should not be a specific person or physical location',
                    'Must be at least 3 characters long',
                    'Should have context indicating cultural or historical significance',
                    'Can be proper nouns or common nouns with specific meaning'
                ],
                examples=[
                    'The War of the Ring', 'The Order of the Phoenix', 'The Force',
                    'The One Ring', 'Excalibur', 'The Holy Grail', 'The Prophecy',
                    'The Code of Chivalry', 'The Dark Arts', 'The Elder Wand'
                ],
                counter_examples=[
                    'Tom', 'Ironhold', 'Annual', 'Founded', 'Cold', 'Military',
                    'Big', 'Small', 'Good', 'Bad', 'The', 'And', 'Or'
                ]
            )
        }
    
    def get_definition(self, entity_type: str) -> Optional[EntityTypeDefinition]:
        """Get definition for a specific entity type."""
        return self.definitions.get(entity_type)
    
    def get_all_definitions(self) -> Dict[str, EntityTypeDefinition]:
        """Get all entity type definitions."""
        return self.definitions
    
    def validate_entity_name(self, name: str, entity_type: str) -> Dict[str, Any]:
        """
        Validate if a name is appropriate for the given entity type.
        
        Returns:
            Dict with validation results including score, reasons, and recommendation
        """
        definition = self.get_definition(entity_type)
        if not definition:
            return {'valid': False, 'score': 0.0, 'reasons': ['Unknown entity type']}
        
        name_lower = name.lower()
        validation_result = {
            'valid': True,
            'score': 0.5,  # Base score
            'reasons': [],
            'recommendation': 'accept'
        }
        
        # Check basic validation rules
        if len(name) < 2:
            validation_result['valid'] = False
            validation_result['score'] = 0.0
            validation_result['reasons'].append('Name too short')
            validation_result['recommendation'] = 'reject'
            return validation_result
        
        # Check against negative indicators (use word boundaries to avoid partial matches)
        negative_score = 0
        for indicator in definition.negative_indicators:
            # Use word boundaries to avoid partial matches like "in" in "King"
            import re
            pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(pattern, name_lower):
                negative_score += 1
                validation_result['reasons'].append(f'Contains negative indicator: {indicator}')

        # Check against positive indicators (use word boundaries)
        positive_score = 0
        for indicator in definition.positive_indicators:
            pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(pattern, name_lower):
                positive_score += 1
                validation_result['reasons'].append(f'Contains positive indicator: {indicator}')
        
        # Calculate final score
        if negative_score > 0:
            validation_result['score'] = max(0.0, validation_result['score'] - (negative_score * 0.3))
        
        if positive_score > 0:
            validation_result['score'] = min(1.0, validation_result['score'] + (positive_score * 0.2))
        
        # Check if it's a proper noun (for characters and locations)
        if entity_type in ['characters', 'locations']:
            if not re.match(r'^[A-Z][a-zA-Z\s\-\']*$', name):
                validation_result['score'] *= 0.7
                validation_result['reasons'].append('Not a proper noun format')
        
        # Final recommendation
        if validation_result['score'] < 0.3:
            validation_result['valid'] = False
            validation_result['recommendation'] = 'reject'
        elif validation_result['score'] < 0.6:
            validation_result['recommendation'] = 'review'
        else:
            validation_result['recommendation'] = 'accept'
        
        return validation_result
