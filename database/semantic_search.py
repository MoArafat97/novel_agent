"""
Enhanced Semantic Search for Worldbuilding.

This module provides advanced semantic search capabilities with filtering,
ranking, and result enhancement for better worldbuilding workflows.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """
    Enhanced semantic search engine for worldbuilding content.
    """
    
    def __init__(self, world_state):
        """
        Initialize the semantic search engine.
        
        Args:
            world_state: WorldState instance for data access
        """
        self.world_state = world_state
        
    def search(self, 
               query: str,
               entity_types: List[str] = None,
               novel_id: str = None,
               n_results: int = 10,
               min_similarity: float = 0.0,
               include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Enhanced semantic search with filtering and ranking.
        
        Args:
            query: Search query text
            entity_types: List of entity types to search (None for all)
            novel_id: Filter by specific novel ID
            n_results: Maximum number of results
            min_similarity: Minimum similarity threshold
            include_metadata: Include additional metadata in results
            
        Returns:
            List of enhanced search results
        """
        try:
            # Perform base semantic search
            if entity_types:
                all_results = []
                for entity_type in entity_types:
                    results = self.world_state.semantic_query(query, entity_type, n_results)
                    all_results.extend(results)
            else:
                all_results = self.world_state.semantic_query(query, None, n_results)
            
            # Filter by novel if specified
            if novel_id:
                all_results = [r for r in all_results if r.get('data', {}).get('novel_id') == novel_id]
            
            # Filter by similarity threshold
            if min_similarity > 0.0:
                all_results = [r for r in all_results 
                             if r.get('similarity_score', 1.0) <= min_similarity]
            
            # Enhance results with metadata
            if include_metadata:
                all_results = self._enhance_results(all_results)
            
            # Sort by relevance
            all_results = self._rank_results(all_results, query)
            
            return all_results[:n_results]
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return []
    
    def search_by_category(self, query: str, novel_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search and organize results by entity type.
        
        Args:
            query: Search query
            novel_id: Optional novel filter
            
        Returns:
            Dictionary with entity types as keys and results as values
        """
        try:
            results = self.search(query, novel_id=novel_id, n_results=20)
            
            categorized = {
                'characters': [],
                'locations': [],
                'lore': [],
                'novels': []
            }
            
            for result in results:
                entity_type = result.get('entity_type')
                if entity_type in categorized:
                    categorized[entity_type].append(result)
            
            return categorized
            
        except Exception as e:
            logger.error(f"Categorized search failed: {e}")
            return {k: [] for k in ['characters', 'locations', 'lore', 'novels']}
    
    def find_related(self, entity_type: str, entity_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find entities related to a specific entity.
        
        Args:
            entity_type: Type of the source entity
            entity_id: ID of the source entity
            n_results: Number of related entities to find
            
        Returns:
            List of related entities
        """
        try:
            # Get the source entity
            source_entity = self.world_state.get(entity_type, entity_id)
            if not source_entity:
                return []
            
            # Create search query from entity description
            search_text = self._create_search_text(entity_type, source_entity)
            
            # Search for similar entities (excluding the source)
            results = self.search(search_text, n_results=n_results + 1)
            
            # Filter out the source entity
            related = [r for r in results if r.get('entity_id') != entity_id]
            
            return related[:n_results]
            
        except Exception as e:
            logger.error(f"Related search failed: {e}")
            return []
    
    def suggest_connections(self, novel_id: str) -> List[Dict[str, Any]]:
        """
        Suggest potential connections between entities in a novel.
        
        Args:
            novel_id: Novel ID to analyze
            
        Returns:
            List of suggested connections
        """
        try:
            # Get all entities for the novel
            novel_entities = self.world_state.get_entities_by_novel(novel_id)
            
            suggestions = []
            
            # Find character-location connections
            for character in novel_entities.get('characters', []):
                char_text = self._create_search_text('characters', character)
                location_results = self.search(char_text, entity_types=['locations'], 
                                             novel_id=novel_id, n_results=3)
                
                for loc_result in location_results:
                    suggestions.append({
                        'type': 'character_location',
                        'character': character,
                        'location': loc_result['data'],
                        'reason': f"Character traits match location themes",
                        'similarity': loc_result.get('similarity_score', 0)
                    })
            
            # Find character-character connections
            characters = novel_entities.get('characters', [])
            for i, char1 in enumerate(characters):
                char1_text = self._create_search_text('characters', char1)
                char_results = self.search(char1_text, entity_types=['characters'], 
                                         novel_id=novel_id, n_results=3)
                
                for char_result in char_results:
                    if char_result['entity_id'] != char1['id']:
                        suggestions.append({
                            'type': 'character_character',
                            'character1': char1,
                            'character2': char_result['data'],
                            'reason': f"Similar backgrounds or complementary skills",
                            'similarity': char_result.get('similarity_score', 0)
                        })
            
            # Sort by similarity and limit results
            suggestions.sort(key=lambda x: x.get('similarity', 1.0))
            return suggestions[:10]
            
        except Exception as e:
            logger.error(f"Connection suggestions failed: {e}")
            return []
    
    def _enhance_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add metadata and context to search results."""
        enhanced = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add entity metadata
            entity_data = result.get('data', {})
            enhanced_result['metadata'] = {
                'created_at': entity_data.get('created_at'),
                'updated_at': entity_data.get('updated_at'),
                'tags': entity_data.get('tags', []),
                'origin': entity_data.get('origin', 'manual')
            }
            
            # Add novel context if applicable
            novel_id = entity_data.get('novel_id')
            if novel_id:
                novel = self.world_state.get('novels', novel_id)
                if novel:
                    enhanced_result['novel_context'] = {
                        'title': novel.get('title'),
                        'genre': novel.get('genre')
                    }
            
            # Add relevance indicators
            enhanced_result['relevance'] = self._calculate_relevance(result)
            
            enhanced.append(enhanced_result)
        
        return enhanced
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank results by relevance and quality."""
        def rank_score(result):
            # Base similarity score (lower is better for distance)
            similarity = result.get('similarity_score', 1.0)
            base_score = 1.0 - similarity if similarity <= 1.0 else 0.0
            
            # Boost recent content
            updated_at = result.get('data', {}).get('updated_at')
            recency_boost = 0.0
            if updated_at:
                try:
                    update_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    days_old = (datetime.now() - update_time.replace(tzinfo=None)).days
                    recency_boost = max(0, (30 - days_old) / 30 * 0.1)  # 10% boost for recent content
                except:
                    pass
            
            # Boost content with tags
            tags_boost = len(result.get('data', {}).get('tags', [])) * 0.02
            
            return base_score + recency_boost + tags_boost
        
        return sorted(results, key=rank_score, reverse=True)
    
    def _calculate_relevance(self, result: Dict[str, Any]) -> str:
        """Calculate relevance indicator for a result."""
        similarity = result.get('similarity_score', 1.0)
        
        if similarity <= 0.3:
            return 'high'
        elif similarity <= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _create_search_text(self, entity_type: str, entity_data: Dict[str, Any]) -> str:
        """Create search text from entity data."""
        if entity_type == 'characters':
            parts = [
                entity_data.get('description', ''),
                entity_data.get('personality', ''),
                entity_data.get('occupation', '')
            ]
        elif entity_type == 'locations':
            parts = [
                entity_data.get('description', ''),
                entity_data.get('culture', ''),
                entity_data.get('type', '')
            ]
        elif entity_type == 'lore':
            parts = [
                entity_data.get('description', ''),
                entity_data.get('category', ''),
                entity_data.get('details', '')[:200]  # Limit details length
            ]
        else:
            parts = [str(entity_data)]
        
        return ' '.join(part for part in parts if part)
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on existing content."""
        try:
            # Get all entities
            all_entities = []
            for entity_type in ['characters', 'locations', 'lore', 'novels']:
                entities = self.world_state.get_all(entity_type)
                all_entities.extend(entities)
            
            # Extract common terms and phrases
            suggestions = set()
            
            for entity in all_entities:
                # Add names/titles
                name = entity.get('name') or entity.get('title', '')
                if name and partial_query.lower() in name.lower():
                    suggestions.add(name)
                
                # Add tags
                for tag in entity.get('tags', []):
                    if partial_query.lower() in tag.lower():
                        suggestions.add(tag)
                
                # Add occupation/type/category
                for field in ['occupation', 'type', 'category', 'genre']:
                    value = entity.get(field, '')
                    if value and partial_query.lower() in value.lower():
                        suggestions.add(value)
            
            return sorted(list(suggestions))[:10]
            
        except Exception as e:
            logger.error(f"Search suggestions failed: {e}")
            return []
