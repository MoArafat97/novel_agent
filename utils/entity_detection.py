"""
Entity Detection Utilities

This module provides lightweight text analysis utilities for detecting entity mentions,
named entity recognition patterns, and integration helpers for cross-reference analysis.
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class EntityDetectionUtils:
    """
    Utility class for lightweight entity detection and text analysis.
    """
    
    def __init__(self):
        # Common patterns for entity detection
        self.name_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper names
            r'\b[A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)+\b',  # Names with titles
        ]
        
        self.location_indicators = [
            'city', 'town', 'village', 'kingdom', 'empire', 'land', 'realm',
            'castle', 'fortress', 'tower', 'palace', 'temple', 'shrine',
            'forest', 'mountain', 'river', 'lake', 'sea', 'ocean',
            'desert', 'plains', 'valley', 'hill', 'peak', 'cave',
            'tavern', 'inn', 'shop', 'market', 'square', 'street',
            'district', 'quarter', 'ward', 'province', 'region'
        ]
        
        self.character_indicators = [
            'king', 'queen', 'prince', 'princess', 'lord', 'lady',
            'duke', 'duchess', 'count', 'countess', 'baron', 'baroness',
            'knight', 'sir', 'dame', 'captain', 'general', 'admiral',
            'wizard', 'mage', 'sorcerer', 'witch', 'priest', 'cleric',
            'merchant', 'trader', 'blacksmith', 'innkeeper', 'guard',
            'soldier', 'warrior', 'archer', 'assassin', 'thief',
            'scholar', 'sage', 'bard', 'healer', 'farmer', 'hunter'
        ]
        
        self.lore_indicators = [
            'legend', 'myth', 'prophecy', 'curse', 'blessing', 'ritual',
            'ceremony', 'tradition', 'custom', 'law', 'rule', 'decree',
            'magic', 'spell', 'enchantment', 'artifact', 'relic',
            'organization', 'guild', 'order', 'brotherhood', 'sisterhood',
            'religion', 'faith', 'god', 'goddess', 'deity', 'spirit',
            'war', 'battle', 'conflict', 'treaty', 'alliance', 'pact'
        ]
    
    def detect_potential_entities(self, text: str, existing_entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect potential entity mentions in text using pattern matching.
        
        Args:
            text: Text to analyze
            existing_entities: Dictionary of existing entities by type
            
        Returns:
            Dictionary of detected entities by type
        """
        detected = {
            'characters': [],
            'locations': [],
            'lore': []
        }
        
        try:
            # Clean and prepare text
            text = self._clean_text(text)
            
            # Extract potential names using patterns
            potential_names = self._extract_names(text)
            
            # Categorize names based on context
            for name in potential_names:
                context = self._get_name_context(text, name)
                entity_type = self._classify_entity_type(name, context)
                
                if entity_type and not self._is_existing_entity(name, entity_type, existing_entities):
                    detected[entity_type].append({
                        'name': name,
                        'context': context,
                        'confidence': self._calculate_confidence(name, context, entity_type),
                        'detection_method': 'pattern_matching'
                    })
            
            # Remove duplicates and low-confidence detections
            for entity_type in detected:
                detected[entity_type] = self._filter_detections(detected[entity_type])
            
            return detected
            
        except Exception as e:
            logger.error(f"Entity detection failed: {e}")
            return detected
    
    def find_entity_mentions(self, text: str, entity_name: str, entity_type: str) -> List[Dict[str, Any]]:
        """
        Find specific mentions of an entity in text.
        
        Args:
            text: Text to search
            entity_name: Name of entity to find
            entity_type: Type of entity
            
        Returns:
            List of mention details
        """
        mentions = []
        
        try:
            # Create search patterns
            patterns = self._create_mention_patterns(entity_name)
            
            for pattern in patterns:
                matches = pattern.finditer(text)  # Pattern already has flags compiled
                for match in matches:
                    context = self._extract_mention_context(text, match.start(), match.end())
                    mentions.append({
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'context': context,
                        'pattern_used': pattern.pattern
                    })
            
            return mentions
            
        except Exception as e:
            logger.error(f"Mention detection failed for {entity_name}: {e}")
            return []
    
    def analyze_text_relationships(self, text: str, entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Analyze text for relationships between entities.
        
        Args:
            text: Text to analyze
            entities: Dictionary of entities by type
            
        Returns:
            List of potential relationships
        """
        relationships = []
        
        try:
            # Find co-occurrences of entities
            entity_mentions = {}
            
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_name = entity.get('name') or entity.get('title', '')
                    if entity_name:
                        mentions = self.find_entity_mentions(text, entity_name, entity_type)
                        if mentions:
                            entity_mentions[entity['id']] = {
                                'entity': entity,
                                'mentions': mentions,
                                'type': entity_type
                            }
            
            # Analyze proximity and context for relationships
            for entity_id1, data1 in entity_mentions.items():
                for entity_id2, data2 in entity_mentions.items():
                    if entity_id1 != entity_id2:
                        relationship = self._analyze_entity_proximity(data1, data2, text)
                        if relationship:
                            relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship analysis failed: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,;:!?\-\'"]', ' ', text)
        return text.strip()
    
    def _extract_names(self, text: str) -> Set[str]:
        """Extract potential names using regex patterns."""
        names = set()
        
        for pattern in self.name_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Filter out common words and short names
                if len(match) > 2 and not self._is_common_word(match):
                    names.add(match.strip())
        
        return names
    
    def _get_name_context(self, text: str, name: str) -> str:
        """Get context around a name mention."""
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        match = pattern.search(text)
        
        if match:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            return text[start:end].strip()
        
        return ""
    
    def _classify_entity_type(self, name: str, context: str) -> str:
        """Classify entity type based on name and context."""
        context_lower = context.lower()
        name_lower = name.lower()
        
        # Check for character indicators
        character_score = sum(1 for indicator in self.character_indicators 
                            if indicator in context_lower or indicator in name_lower)
        
        # Check for location indicators
        location_score = sum(1 for indicator in self.location_indicators 
                           if indicator in context_lower or indicator in name_lower)
        
        # Check for lore indicators
        lore_score = sum(1 for indicator in self.lore_indicators 
                        if indicator in context_lower)
        
        # Determine type based on highest score
        scores = {
            'characters': character_score,
            'locations': location_score,
            'lore': lore_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        # Default classification based on capitalization patterns
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', name):
            return 'characters'  # Proper names default to characters
        
        return None
    
    def _is_existing_entity(self, name: str, entity_type: str, existing_entities: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Check if entity already exists."""
        entities = existing_entities.get(entity_type, [])
        
        for entity in entities:
            existing_name = entity.get('name') or entity.get('title', '')
            if existing_name.lower() == name.lower():
                return True
        
        return False
    
    def _calculate_confidence(self, name: str, context: str, entity_type: str) -> str:
        """Calculate confidence level for entity detection."""
        score = 0
        
        # Length bonus
        if len(name) > 5:
            score += 1
        
        # Context indicators
        context_lower = context.lower()
        if entity_type == 'characters':
            indicators = self.character_indicators
        elif entity_type == 'locations':
            indicators = self.location_indicators
        else:
            indicators = self.lore_indicators
        
        indicator_count = sum(1 for indicator in indicators if indicator in context_lower)
        score += min(indicator_count, 3)
        
        # Capitalization pattern
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', name):
            score += 1
        
        if score >= 4:
            return 'high'
        elif score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out low-quality detections."""
        # Remove low confidence detections
        filtered = [d for d in detections if d['confidence'] != 'low']
        
        # Remove duplicates
        seen_names = set()
        unique_detections = []
        
        for detection in filtered:
            name_lower = detection['name'].lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_detections.append(detection)
        
        return unique_detections
    
    def _is_common_word(self, word: str) -> bool:
        """Check if word is a common word that shouldn't be considered a name."""
        common_words = {
            'the', 'and', 'but', 'for', 'nor', 'yet', 'so', 'or',
            'this', 'that', 'these', 'those', 'here', 'there',
            'when', 'where', 'why', 'how', 'what', 'who', 'which',
            'some', 'many', 'few', 'all', 'any', 'each', 'every',
            'first', 'last', 'next', 'previous', 'new', 'old',
            'good', 'bad', 'big', 'small', 'long', 'short'
        }
        
        return word.lower() in common_words
    
    def _create_mention_patterns(self, entity_name: str) -> List[re.Pattern]:
        """Create regex patterns for finding entity mentions."""
        patterns = []
        
        # Exact match
        patterns.append(re.compile(re.escape(entity_name), re.IGNORECASE))
        
        # Partial matches for multi-word names
        words = entity_name.split()
        if len(words) > 1:
            # First name only
            patterns.append(re.compile(r'\b' + re.escape(words[0]) + r'\b', re.IGNORECASE))
            # Last name only
            patterns.append(re.compile(r'\b' + re.escape(words[-1]) + r'\b', re.IGNORECASE))
        
        return patterns
    
    def _extract_mention_context(self, text: str, start: int, end: int, window: int = 30) -> str:
        """Extract context around a mention."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _analyze_entity_proximity(self, data1: Dict[str, Any], data2: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Analyze proximity between two entities to determine relationships."""
        entity1 = data1['entity']
        entity2 = data2['entity']
        mentions1 = data1['mentions']
        mentions2 = data2['mentions']
        
        # Find closest mentions
        min_distance = float('inf')
        closest_context = ""
        
        for mention1 in mentions1:
            for mention2 in mentions2:
                distance = abs(mention1['start'] - mention2['start'])
                if distance < min_distance:
                    min_distance = distance
                    # Get context that includes both mentions
                    start = min(mention1['start'], mention2['start'])
                    end = max(mention1['end'], mention2['end'])
                    closest_context = self._extract_mention_context(text, start, end, 50)
        
        # Only consider relationships if entities are mentioned close together
        if min_distance < 200:  # Within 200 characters
            return {
                'entity1_id': entity1['id'],
                'entity1_name': entity1.get('name') or entity1.get('title'),
                'entity1_type': data1['type'],
                'entity2_id': entity2['id'],
                'entity2_name': entity2.get('name') or entity2.get('title'),
                'entity2_type': data2['type'],
                'distance': min_distance,
                'context': closest_context,
                'strength': 'strong' if min_distance < 50 else 'moderate' if min_distance < 100 else 'weak'
            }
        
        return None
