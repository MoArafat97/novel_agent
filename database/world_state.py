"""
WorldState: A unified interface for managing structured (TinyDB) and semantic (ChromaDB) data.

This class provides a clean abstraction layer that handles synchronization between
TinyDB for structured worldbuilding data and ChromaDB for semantic memory.
Optimized for high performance with advanced caching, indexing, and memory management.
"""

import os
import uuid
import logging
import time
import hashlib
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from collections import defaultdict, OrderedDict
from functools import lru_cache
import json
import gc

import chromadb
from chromadb.config import Settings
from tinydb import TinyDB, Query
from tinydb.table import Table
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorldState:
    """
    Unified interface for managing worldbuilding data across TinyDB and ChromaDB.

    Responsibilities:
    - TinyDB: Structured data storage (characters, locations, lore, novels)
    - ChromaDB: Semantic memory with embeddings for AI queries
    - Synchronization: Automatic sync between both databases
    - Performance: Advanced caching, indexing, and memory optimization
    """

    def __init__(self,
                 tinydb_path: str = None,
                 chromadb_path: str = None,
                 openrouter_api_key: str = None,
                 enable_caching: bool = True,
                 cache_size: int = 1000,
                 enable_indexing: bool = True):
        """
        Initialize WorldState with database connections and performance optimizations.

        Args:
            tinydb_path: Path to TinyDB storage directory
            chromadb_path: Path to ChromaDB storage directory
            openrouter_api_key: OpenRouter API key for embeddings
            enable_caching: Enable query result caching
            cache_size: Maximum number of cached items
            enable_indexing: Enable in-memory indexing
        """
        # Configuration
        self.tinydb_path = tinydb_path or os.getenv('TINYDB_PATH', './data/tinydb')
        self.chromadb_path = chromadb_path or os.getenv('CHROMADB_PATH', './data/chromadb')
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')

        # Performance configuration
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.enable_indexing = enable_indexing

        # Entity types - define before initialization
        self.entity_types = ['novels', 'characters', 'locations', 'lore']

        # Performance optimization structures
        self._init_performance_structures()

        # Ensure directories exist
        Path(self.tinydb_path).mkdir(parents=True, exist_ok=True)
        Path(self.chromadb_path).mkdir(parents=True, exist_ok=True)

        # Initialize databases
        self._init_tinydb()
        self._init_chromadb()
        self._init_openrouter()

        # Initialize performance features
        if self.enable_indexing:
            self._build_indexes()

        logger.info("WorldState initialized successfully with performance optimizations")

    def _init_performance_structures(self):
        """Initialize performance optimization structures."""
        # Query result cache
        self.query_cache = OrderedDict() if self.enable_caching else None
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}

        # Embedding cache
        self.embedding_cache = {}
        self.embedding_cache_stats = {'hits': 0, 'misses': 0}

        # In-memory indexes
        self.indexes = {
            'by_novel_id': defaultdict(lambda: defaultdict(list)),
            'by_name': defaultdict(dict),
            'by_tags': defaultdict(lambda: defaultdict(list)),
            'by_type': defaultdict(list)
        }

        # Connection pools and resource management
        self.connection_pool = {}
        self.resource_lock = threading.RLock()

        # Performance metrics
        self.performance_metrics = {
            'query_times': [],
            'embedding_times': [],
            'cache_hit_rate': 0.0,
            'memory_usage': 0
        }

        logger.info("Performance structures initialized")

    def _init_tinydb(self):
        """Initialize TinyDB connections for each entity type with optimization."""
        self.tinydb_connections = {}

        for entity_type in ['novels', 'characters', 'locations', 'lore']:
            db_path = os.path.join(self.tinydb_path, f'{entity_type}.json')
            self.tinydb_connections[entity_type] = TinyDB(db_path)

        logger.info(f"TinyDB initialized at {self.tinydb_path}")

    def _build_indexes(self):
        """Build in-memory indexes for faster queries."""
        if not self.enable_indexing:
            return

        start_time = time.time()

        for entity_type in self.entity_types:
            db = self.tinydb_connections[entity_type]
            all_entities = db.all()

            for entity in all_entities:
                entity_id = entity.get('id')
                novel_id = entity.get('novel_id')
                name = entity.get('name') or entity.get('title', '')
                tags = entity.get('tags', [])

                if entity_id:
                    # Index by novel_id
                    if novel_id:
                        self.indexes['by_novel_id'][novel_id][entity_type].append(entity_id)

                    # Index by name
                    if name:
                        self.indexes['by_name'][entity_type][name.lower()] = entity_id

                    # Index by tags
                    for tag in tags:
                        if tag:
                            self.indexes['by_tags'][tag.lower()][entity_type].append(entity_id)

                    # Index by type
                    self.indexes['by_type'][entity_type].append(entity_id)

        build_time = time.time() - start_time
        logger.info(f"Indexes built in {build_time:.3f}s")

    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for operation."""
        key_parts = [operation]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            key_parts.append(f"{k}:{v}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def _cache_get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enable_caching or not self.query_cache:
            return None

        if cache_key in self.query_cache:
            # Move to end (LRU)
            value = self.query_cache.pop(cache_key)
            self.query_cache[cache_key] = value
            self.cache_stats['hits'] += 1
            return value

        self.cache_stats['misses'] += 1
        return None

    def _cache_set(self, cache_key: str, value: Any):
        """Set value in cache with LRU eviction."""
        if not self.enable_caching or not self.query_cache:
            return

        # Evict oldest if at capacity
        if len(self.query_cache) >= self.cache_size:
            self.query_cache.popitem(last=False)
            self.cache_stats['evictions'] += 1

        self.query_cache[cache_key] = value

    def _init_chromadb(self):
        """Initialize ChromaDB with DuckDB+Parquet persistence."""
        try:
            # Configure ChromaDB with DuckDB+Parquet persistence
            settings = Settings(
                persist_directory=self.chromadb_path,
                is_persistent=True
            )
            
            self.chroma_client = chromadb.PersistentClient(
                path=self.chromadb_path,
                settings=settings
            )
            
            # Create collections for each entity type
            self.chroma_collections = {}
            for entity_type in self.entity_types:
                try:
                    collection = self.chroma_client.get_collection(name=entity_type)
                except:
                    collection = self.chroma_client.create_collection(
                        name=entity_type,
                        metadata={"description": f"Semantic embeddings for {entity_type}"}
                    )
                self.chroma_collections[entity_type] = collection
                
            logger.info(f"ChromaDB initialized at {self.chromadb_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _init_openrouter(self):
        """Initialize OpenRouter client for embeddings."""
        # For now, disable OpenRouter embeddings and use ChromaDB default
        use_openrouter = os.getenv('USE_OPENROUTER_EMBEDDINGS', 'false').lower() == 'true'

        if not use_openrouter or not self.openrouter_api_key:
            logger.info("Using ChromaDB default embeddings (OpenRouter disabled)")
            self.openrouter_client = None
            return

        try:
            self.openrouter_client = openai.OpenAI(
                base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
                api_key=self.openrouter_api_key
            )
            logger.info("OpenRouter client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter: {e}")
            self.openrouter_client = None
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenRouter/DeepSeek with caching.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        # Check embedding cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            self.embedding_cache_stats['hits'] += 1
            return self.embedding_cache[text_hash]

        self.embedding_cache_stats['misses'] += 1

        if not self.openrouter_client:
            logger.warning("OpenRouter client not available, returning dummy embedding")
            # Return a dummy embedding for testing
            embedding = [0.0] * 1536
        else:
            try:
                start_time = time.time()
                response = self.openrouter_client.embeddings.create(
                    model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                    input=text
                )
                embedding = response.data[0].embedding

                # Track performance
                embedding_time = time.time() - start_time
                self.performance_metrics['embedding_times'].append(embedding_time)

            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                # Return dummy embedding as fallback
                embedding = [0.0] * 1536

        # Cache the embedding
        self.embedding_cache[text_hash] = embedding

        # Limit cache size
        if len(self.embedding_cache) > 1000:
            # Remove oldest 20% of entries
            items_to_remove = len(self.embedding_cache) // 5
            for _ in range(items_to_remove):
                self.embedding_cache.pop(next(iter(self.embedding_cache)))

        return embedding
    
    def _prepare_embedding_text(self, entity_type: str, data: Dict[str, Any]) -> str:
        """
        Prepare optimized text for embedding based on entity type.

        Args:
            entity_type: Type of entity (novels, characters, locations, lore)
            data: Entity data

        Returns:
            Formatted text for embedding (truncated to optimal length)
        """
        # Prepare base text
        if entity_type == 'novels':
            text = f"Novel: {data.get('title', '')}. Genre: {data.get('genre', '')}. Description: {data.get('description', '')}"

        elif entity_type == 'characters':
            text = f"Character: {data.get('name', '')}. Occupation: {data.get('occupation', '')}. Description: {data.get('description', '')}. Personality: {data.get('personality', '')}"

        elif entity_type == 'locations':
            text = f"Location: {data.get('name', '')}. Type: {data.get('type', '')}. Description: {data.get('description', '')}. Culture: {data.get('culture', '')}"

        elif entity_type == 'lore':
            text = f"Lore: {data.get('title', '')}. Category: {data.get('category', '')}. Description: {data.get('description', '')}. Details: {data.get('details', '')}"

        else:
            text = str(data)

        # Optimize text length for embedding (most models work best with 512-1024 tokens)
        # Roughly 4 characters per token, so aim for ~2000-4000 characters
        if len(text) > 3000:
            text = text[:2900] + "..."

        return text.strip()
    
    def add_or_update(self, entity_type: str, entity_id: str, data: Dict[str, Any],
                     embedding_text: str = None, skip_embeddings: bool = False) -> bool:
        """
        Add or update an entity in both TinyDB and ChromaDB.
        
        Args:
            entity_type: Type of entity (novels, characters, locations, lore)
            entity_id: Unique identifier for the entity
            data: Entity data to store
            embedding_text: Optional custom text for embedding (auto-generated if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate entity type
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")
            
            # Ensure ID and timestamps
            data['id'] = entity_id
            current_time = datetime.now().isoformat()
            
            # Check if this is an update or new entry
            EntityQuery = Query()
            existing = self.tinydb_connections[entity_type].search(EntityQuery.id == entity_id)
            
            if existing:
                data['updated_at'] = current_time
                # Convert TinyDB Document to dict to access attributes safely
                existing_dict = dict(existing[0])
                data['created_at'] = existing_dict.get('created_at', current_time)
            else:
                data['created_at'] = current_time
                data['updated_at'] = current_time
            
            # Update TinyDB
            self.tinydb_connections[entity_type].upsert(data, EntityQuery.id == entity_id)
            
            # Prepare embedding text
            if embedding_text is None:
                embedding_text = self._prepare_embedding_text(entity_type, data)
            
            # Update ChromaDB - let ChromaDB handle embeddings automatically
            collection = self.chroma_collections[entity_type]

            # Check if document exists in ChromaDB
            try:
                existing_docs = collection.get(ids=[entity_id])
                if existing_docs['ids']:
                    # Update existing document (let ChromaDB generate embeddings)
                    collection.update(
                        ids=[entity_id],
                        documents=[embedding_text],
                        metadatas=[{
                            'entity_type': entity_type,
                            'entity_id': entity_id,
                            'updated_at': current_time
                        }]
                    )
                else:
                    # Add new document (let ChromaDB generate embeddings)
                    collection.add(
                        ids=[entity_id],
                        documents=[embedding_text],
                        metadatas=[{
                            'entity_type': entity_type,
                            'entity_id': entity_id,
                            'created_at': current_time
                        }]
                    )
            except:
                # Add new document (fallback)
                collection.add(
                    ids=[entity_id],
                    documents=[embedding_text],
                    metadatas=[{
                        'entity_type': entity_type,
                        'entity_id': entity_id,
                        'created_at': current_time
                    }]
                )
            
            logger.info(f"Successfully added/updated {entity_type} entity: {entity_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add/update {entity_type} entity {entity_id}: {e}")
            return False

    def get(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID from TinyDB.

        Args:
            entity_type: Type of entity
            entity_id: Unique identifier

        Returns:
            Entity data or None if not found
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")

            EntityQuery = Query()
            results = self.tinydb_connections[entity_type].search(EntityQuery.id == entity_id)
            # Convert TinyDB Document to regular dictionary
            return dict(results[0]) if results else None

        except Exception as e:
            logger.error(f"Failed to get {entity_type} entity {entity_id}: {e}")
            return None

    def get_all(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type from TinyDB.

        Args:
            entity_type: Type of entity

        Returns:
            List of entity data
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")

            # Convert TinyDB Document objects to regular dictionaries
            documents = self.tinydb_connections[entity_type].all()
            return [dict(doc) for doc in documents]

        except Exception as e:
            logger.error(f"Failed to get all {entity_type} entities: {e}")
            return []

    def delete(self, entity_type: str, entity_id: str) -> bool:
        """
        Delete an entity from both TinyDB and ChromaDB.

        Args:
            entity_type: Type of entity
            entity_id: Unique identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")

            # Delete from TinyDB
            EntityQuery = Query()
            removed_count = len(self.tinydb_connections[entity_type].remove(EntityQuery.id == entity_id))

            # Delete from ChromaDB
            try:
                collection = self.chroma_collections[entity_type]
                collection.delete(ids=[entity_id])
            except Exception as e:
                logger.warning(f"Failed to delete from ChromaDB (may not exist): {e}")

            if removed_count > 0:
                logger.info(f"Successfully deleted {entity_type} entity: {entity_id}")
                return True
            else:
                logger.warning(f"Entity not found for deletion: {entity_type} {entity_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete {entity_type} entity {entity_id}: {e}")
            return False

    def semantic_query(self, query_text: str, entity_type: str = None,
                      n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search across ChromaDB collections.

        Args:
            query_text: Text to search for
            entity_type: Optional entity type to limit search (None for all types)
            n_results: Number of results to return

        Returns:
            List of matching entities with similarity scores
        """
        try:
            results = []

            # Determine which collections to search
            collections_to_search = [entity_type] if entity_type else self.entity_types

            for collection_name in collections_to_search:
                if collection_name not in self.entity_types:
                    continue

                collection = self.chroma_collections[collection_name]

                # Perform semantic search
                search_results = collection.query(
                    query_texts=[query_text],
                    n_results=min(n_results, 10)  # Limit per collection
                )

                # Process results
                if search_results['ids'] and search_results['ids'][0]:
                    for i, entity_id in enumerate(search_results['ids'][0]):
                        # Get full entity data from TinyDB
                        entity_data = self.get(collection_name, entity_id)
                        if entity_data:
                            result = {
                                'entity_type': collection_name,
                                'entity_id': entity_id,
                                'data': entity_data,
                                'similarity_score': search_results.get('distances', [[]])[0][i] if search_results.get('distances') else None,
                                'matched_text': search_results.get('documents', [[]])[0][i] if search_results.get('documents') else None
                            }
                            results.append(result)

            # Sort by similarity score (lower is better for distance)
            if results and results[0].get('similarity_score') is not None:
                results.sort(key=lambda x: x['similarity_score'])

            return results[:n_results]

        except Exception as e:
            logger.error(f"Failed to perform semantic query: {e}")
            return []

    def get_entities_by_novel(self, novel_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all entities associated with a specific novel.

        Args:
            novel_id: Novel identifier

        Returns:
            Dictionary with entity types as keys and lists of entities as values
        """
        try:
            result = {}
            EntityQuery = Query()

            for entity_type in ['characters', 'locations', 'lore']:
                entities = self.tinydb_connections[entity_type].search(EntityQuery.novel_id == novel_id)
                # Convert TinyDB Documents to regular dictionaries
                result[entity_type] = [dict(entity) for entity in entities]

            return result

        except Exception as e:
            logger.error(f"Failed to get entities for novel {novel_id}: {e}")
            return {}

    def delete_novel_and_related(self, novel_id: str) -> bool:
        """
        Delete a novel and all its related entities.

        Args:
            novel_id: Novel identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            success = True

            # Delete the novel itself
            if not self.delete('novels', novel_id):
                success = False

            # Delete related entities
            EntityQuery = Query()
            for entity_type in ['characters', 'locations', 'lore']:
                related_entities = self.tinydb_connections[entity_type].search(EntityQuery.novel_id == novel_id)
                for entity in related_entities:
                    # Convert TinyDB Document to dict to access attributes safely
                    entity_dict = dict(entity)
                    if not self.delete(entity_type, entity_dict['id']):
                        success = False

            logger.info(f"Deleted novel {novel_id} and related entities")
            return success

        except Exception as e:
            logger.error(f"Failed to delete novel and related entities {novel_id}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the worldbuilding data.

        Returns:
            Dictionary with counts and other statistics
        """
        try:
            stats = {}

            # Count entities in TinyDB
            for entity_type in self.entity_types:
                count = len(self.tinydb_connections[entity_type].all())
                stats[f'{entity_type}_count'] = count

            # ChromaDB collection info
            chroma_stats = {}
            for entity_type in self.entity_types:
                try:
                    collection = self.chroma_collections[entity_type]
                    chroma_count = collection.count()
                    chroma_stats[f'{entity_type}_embeddings'] = chroma_count
                except:
                    chroma_stats[f'{entity_type}_embeddings'] = 0

            stats['chromadb'] = chroma_stats
            stats['total_entities'] = sum(stats[f'{et}_count'] for et in self.entity_types)

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    # ===== OPTIMIZED QUERY METHODS =====

    def get_entities_by_novel_optimized(self, novel_id: str, entity_type: str = None) -> List[Dict[str, Any]]:
        """
        Get entities by novel ID using optimized indexing and caching.

        Args:
            novel_id: Novel ID to filter by
            entity_type: Optional entity type filter

        Returns:
            List of entities
        """
        cache_key = self._get_cache_key("get_by_novel", novel_id=novel_id, entity_type=entity_type)

        # Check cache first
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result

        start_time = time.time()
        results = []

        # Use index if available
        if self.enable_indexing and novel_id in self.indexes['by_novel_id']:
            entity_types = [entity_type] if entity_type else self.entity_types

            for etype in entity_types:
                entity_ids = self.indexes['by_novel_id'][novel_id].get(etype, [])
                for entity_id in entity_ids:
                    entity = self.get_entity_optimized(etype, entity_id)
                    if entity:
                        results.append(entity)
        else:
            # Fallback to direct query
            entity_types = [entity_type] if entity_type else self.entity_types
            EntityQuery = Query()

            for etype in entity_types:
                db = self.tinydb_connections[etype]
                entities = db.search(EntityQuery.novel_id == novel_id)
                results.extend(entities)

        # Track performance
        query_time = time.time() - start_time
        self.performance_metrics['query_times'].append(query_time)

        # Cache result
        self._cache_set(cache_key, results)

        return results

    def get_entity_optimized(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get single entity with caching optimization.

        Args:
            entity_type: Type of entity
            entity_id: Entity ID

        Returns:
            Entity data or None
        """
        cache_key = self._get_cache_key("get_entity", entity_type=entity_type, entity_id=entity_id)

        # Check cache first
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result

        # Query database
        EntityQuery = Query()
        db = self.tinydb_connections[entity_type]
        results = db.search(EntityQuery.id == entity_id)

        result = results[0] if results else None

        # Cache result
        self._cache_set(cache_key, result)

        return result

    def search_entities_optimized(self, query: str, entity_types: List[str] = None,
                                novel_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Optimized semantic search across entities.

        Args:
            query: Search query
            entity_types: Optional entity type filter
            novel_id: Optional novel filter
            limit: Maximum results

        Returns:
            List of matching entities with scores
        """
        cache_key = self._get_cache_key("search", query=query, entity_types=entity_types,
                                      novel_id=novel_id, limit=limit)

        # Check cache first
        cached_result = self._cache_get(cache_key)
        if cached_result is not None:
            return cached_result

        start_time = time.time()
        results = []

        entity_types = entity_types or self.entity_types

        for entity_type in entity_types:
            try:
                collection = self.chroma_collections[entity_type]

                # Build where clause for novel filtering
                where_clause = {}
                if novel_id:
                    where_clause = {"$and": [{"entity_type": {"$eq": entity_type}}]}

                # Perform semantic search
                search_results = collection.query(
                    query_texts=[query],
                    n_results=min(limit, 50),  # Limit per entity type
                    where=where_clause if where_clause else None
                )

                # Process results
                if search_results['ids'] and search_results['ids'][0]:
                    for i, entity_id in enumerate(search_results['ids'][0]):
                        # Get full entity data from TinyDB
                        entity_data = self.get_entity_optimized(entity_type, entity_id)
                        if entity_data:
                            # Filter by novel_id if specified
                            if novel_id and entity_data.get('novel_id') != novel_id:
                                continue

                            entity_data['_score'] = search_results['distances'][0][i]
                            entity_data['_entity_type'] = entity_type
                            results.append(entity_data)

            except Exception as e:
                logger.error(f"Search error for {entity_type}: {e}")
                continue

        # Sort by score and limit
        results.sort(key=lambda x: x.get('_score', 1.0))
        results = results[:limit]

        # Track performance
        search_time = time.time() - start_time
        self.performance_metrics['query_times'].append(search_time)

        # Cache result
        self._cache_set(cache_key, results)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = 0.0
        if self.cache_stats['hits'] + self.cache_stats['misses'] > 0:
            cache_hit_rate = self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses'])

        embedding_hit_rate = 0.0
        if self.embedding_cache_stats['hits'] + self.embedding_cache_stats['misses'] > 0:
            embedding_hit_rate = self.embedding_cache_stats['hits'] / (
                self.embedding_cache_stats['hits'] + self.embedding_cache_stats['misses'])

        return {
            'cache_stats': self.cache_stats,
            'cache_hit_rate': cache_hit_rate,
            'embedding_cache_stats': self.embedding_cache_stats,
            'embedding_hit_rate': embedding_hit_rate,
            'query_times': self.performance_metrics['query_times'][-100:],  # Last 100
            'embedding_times': self.performance_metrics['embedding_times'][-100:],  # Last 100
            'memory_usage': self._get_memory_usage()
        }

    def _get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        return {
            'query_cache_size': len(self.query_cache) if self.query_cache else 0,
            'embedding_cache_size': len(self.embedding_cache),
            'index_size': sum(len(str(self.indexes))),
        }

    def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries matching pattern."""
        if not self.enable_caching or not self.query_cache:
            return

        if pattern is None:
            # Clear all cache
            self.query_cache.clear()
            self.cache_stats['invalidations'] += 1
        else:
            # Clear matching entries
            keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.query_cache[key]
                self.cache_stats['invalidations'] += 1

    def close(self):
        """Close database connections and cleanup resources."""
        try:
            # Close TinyDB connections
            for db in self.tinydb_connections.values():
                db.close()

            # Clear caches to free memory
            if self.query_cache:
                self.query_cache.clear()
            self.embedding_cache.clear()

            # Force garbage collection
            gc.collect()

            logger.info("WorldState connections closed and resources cleaned up")

        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False  # Don't suppress exceptions
