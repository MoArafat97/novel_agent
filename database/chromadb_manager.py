"""
ChromaDB Manager: Handles semantic memory operations for worldbuilding entities.

This module provides specialized operations for ChromaDB, including embedding generation,
semantic search, vector database management with DuckDB+Parquet persistence, and performance optimizations.
"""

import os
import logging
import time
import hashlib
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import OrderedDict, defaultdict
import json

import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """
    Manages ChromaDB operations for semantic memory and vector search with performance optimizations.
    """

    def __init__(self,
                 db_path: str = './data/chromadb',
                 openrouter_api_key: str = None,
                 enable_caching: bool = True,
                 cache_size: int = 500,
                 batch_size: int = 100):
        """
        Initialize ChromaDB manager with performance optimizations.

        Args:
            db_path: Path to ChromaDB storage directory
            openrouter_api_key: OpenRouter API key for embeddings
            enable_caching: Enable embedding and query caching
            cache_size: Maximum number of cached items
            batch_size: Batch size for bulk operations
        """
        self.db_path = db_path
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        self.enable_caching = enable_caching
        self.cache_size = cache_size
        self.batch_size = batch_size

        # Ensure directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        # Performance optimization structures
        self.embedding_cache = OrderedDict() if enable_caching else None
        self.query_cache = OrderedDict() if enable_caching else None
        self.cache_stats = {
            'embedding_hits': 0, 'embedding_misses': 0,
            'query_hits': 0, 'query_misses': 0,
            'evictions': 0
        }

        # Performance metrics
        self.performance_metrics = {
            'embedding_times': [],
            'search_times': [],
            'batch_times': []
        }

        # Thread safety
        self.cache_lock = threading.RLock()

        # Initialize ChromaDB
        self._init_chromadb()

        # Initialize OpenRouter client
        self._init_openrouter()

        # Entity types
        self.entity_types = ['novels', 'characters', 'locations', 'lore']

        logger.info(f"ChromaDB Manager initialized at {self.db_path} with optimizations")
    
    def _init_chromadb(self):
        """Initialize ChromaDB with DuckDB+Parquet persistence."""
        try:
            # Configure ChromaDB with persistence
            settings = Settings(
                persist_directory=self.db_path,
                is_persistent=True
            )
            
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=settings
            )
            
            # Initialize collections for each entity type
            self.collections = {}
            for entity_type in self.entity_types:
                try:
                    # Try to get existing collection
                    collection = self.client.get_collection(name=entity_type)
                except:
                    # Create new collection if it doesn't exist
                    collection = self.client.create_collection(
                        name=entity_type,
                        metadata={
                            "description": f"Semantic embeddings for {entity_type}",
                            "created_at": datetime.now().isoformat()
                        }
                    )
                
                self.collections[entity_type] = collection
                logger.info(f"ChromaDB collection '{entity_type}' ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _init_openrouter(self):
        """Initialize OpenRouter client for embeddings."""
        # Disable OpenRouter embeddings for now - use ChromaDB default
        use_openrouter = os.getenv('USE_OPENROUTER_EMBEDDINGS', 'false').lower() == 'true'

        if not use_openrouter:
            logger.info("Using ChromaDB default embeddings (OpenRouter disabled)")
            self.openrouter_client = None
            return

        if not self.openrouter_api_key:
            logger.warning("No OpenRouter API key provided. Using ChromaDB default embeddings.")
            self.openrouter_client = None
            return

        try:
            self.openrouter_client = openai.OpenAI(
                base_url=os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
                api_key=self.openrouter_api_key
            )

            # Test the connection
            self._test_embedding_generation()
            logger.info("OpenRouter client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter: {e}")
            self.openrouter_client = None
    
    def _test_embedding_generation(self):
        """Test embedding generation with a simple text."""
        try:
            if self.openrouter_client:
                test_embedding = self._generate_embedding("test")
                if test_embedding and len(test_embedding) > 0:
                    logger.info(f"Embedding test successful, dimension: {len(test_embedding)}")
                else:
                    raise Exception("Empty embedding returned")
        except Exception as e:
            logger.warning(f"Embedding test failed: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using OpenRouter/DeepSeek with caching.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()

        with self.cache_lock:
            if self.enable_caching and self.embedding_cache and text_hash in self.embedding_cache:
                # Move to end (LRU)
                embedding = self.embedding_cache.pop(text_hash)
                self.embedding_cache[text_hash] = embedding
                self.cache_stats['embedding_hits'] += 1
                return embedding

        self.cache_stats['embedding_misses'] += 1

        if not self.openrouter_client:
            # Return dummy embedding for testing
            logger.debug("Using dummy embedding (no OpenRouter client)")
            embedding = [0.1] * 1536  # Standard embedding dimension
        else:
            try:
                start_time = time.time()

                # Clean and prepare text
                text = text.strip()
                if not text:
                    text = "empty"

                # Optimize text length for embedding
                if len(text) > 3000:
                    text = text[:2900] + "..."

                # Generate embedding using OpenRouter
                response = self.openrouter_client.embeddings.create(
                    model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                    input=text
                )

                embedding = response.data[0].embedding

                # Track performance
                embedding_time = time.time() - start_time
                self.performance_metrics['embedding_times'].append(embedding_time)

                logger.debug(f"Generated embedding for text (length: {len(text)}, dim: {len(embedding)}, time: {embedding_time:.3f}s)")

            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                # Return dummy embedding as fallback
                embedding = [0.1] * 1536

        # Cache the embedding
        with self.cache_lock:
            if self.enable_caching and self.embedding_cache is not None:
                # Evict oldest if at capacity
                if len(self.embedding_cache) >= self.cache_size:
                    self.embedding_cache.popitem(last=False)
                    self.cache_stats['evictions'] += 1

                self.embedding_cache[text_hash] = embedding

        return embedding
    
    def _prepare_embedding_text(self, entity_type: str, data: Dict[str, Any]) -> str:
        """
        Prepare text for embedding based on entity type.
        
        Args:
            entity_type: Type of entity
            data: Entity data
            
        Returns:
            Formatted text for embedding
        """
        try:
            if entity_type == 'novels':
                parts = [
                    f"Novel: {data.get('title', '')}",
                    f"Genre: {data.get('genre', '')}",
                    f"Description: {data.get('description', '')}"
                ]
                
            elif entity_type == 'characters':
                parts = [
                    f"Character: {data.get('name', '')}",
                    f"Occupation: {data.get('occupation', '')}",
                    f"Description: {data.get('description', '')}",
                    f"Personality: {data.get('personality', '')}"
                ]
                
            elif entity_type == 'locations':
                parts = [
                    f"Location: {data.get('name', '')}",
                    f"Type: {data.get('type', '')}",
                    f"Description: {data.get('description', '')}",
                    f"Culture: {data.get('culture', '')}"
                ]
                
            elif entity_type == 'lore':
                parts = [
                    f"Lore: {data.get('title', '')}",
                    f"Category: {data.get('category', '')}",
                    f"Description: {data.get('description', '')}",
                    f"Details: {data.get('details', '')}"
                ]
            else:
                parts = [str(data)]
            
            # Filter out empty parts and join
            text = ". ".join(part for part in parts if part.split(": ", 1)[-1].strip())
            return text if text else "No description available"
            
        except Exception as e:
            logger.error(f"Failed to prepare embedding text: {e}")
            return str(data)
    
    def add_or_update(self, entity_type: str, entity_id: str, 
                     data: Dict[str, Any], embedding_text: str = None) -> bool:
        """
        Add or update an entity's semantic representation.
        
        Args:
            entity_type: Type of entity
            entity_id: Unique identifier
            data: Entity data
            embedding_text: Optional custom text for embedding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")
            
            collection = self.collections[entity_type]
            
            # Prepare embedding text
            if embedding_text is None:
                embedding_text = self._prepare_embedding_text(entity_type, data)
            
            # Generate embedding
            embedding = self._generate_embedding(embedding_text)
            
            # Prepare metadata
            metadata = {
                'entity_type': entity_type,
                'entity_id': entity_id,
                'updated_at': datetime.now().isoformat(),
                'name': data.get('name') or data.get('title', ''),
                'tags': ','.join(data.get('tags', [])) if data.get('tags') else ''
            }
            
            # Check if document already exists
            try:
                existing = collection.get(ids=[entity_id])
                if existing['ids']:
                    # Update existing document
                    collection.update(
                        ids=[entity_id],
                        documents=[embedding_text],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                    logger.debug(f"Updated ChromaDB entry for {entity_type}: {entity_id}")
                else:
                    # Add new document
                    collection.add(
                        ids=[entity_id],
                        documents=[embedding_text],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                    logger.debug(f"Added ChromaDB entry for {entity_type}: {entity_id}")
            except:
                # Fallback: add as new document
                collection.add(
                    ids=[entity_id],
                    documents=[embedding_text],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
                logger.debug(f"Added ChromaDB entry (fallback) for {entity_type}: {entity_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add/update ChromaDB entry for {entity_type} {entity_id}: {e}")
            return False
    
    def delete(self, entity_type: str, entity_id: str) -> bool:
        """
        Delete an entity's semantic representation.
        
        Args:
            entity_type: Type of entity
            entity_id: Unique identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")
            
            collection = self.collections[entity_type]
            
            # Check if document exists before deleting
            try:
                existing = collection.get(ids=[entity_id])
                if existing['ids']:
                    collection.delete(ids=[entity_id])
                    logger.debug(f"Deleted ChromaDB entry for {entity_type}: {entity_id}")
                    return True
                else:
                    logger.warning(f"ChromaDB entry not found for deletion: {entity_type} {entity_id}")
                    return False
            except Exception as e:
                logger.warning(f"Error checking/deleting ChromaDB entry: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB entry for {entity_type} {entity_id}: {e}")
            return False
    
    def semantic_search(self, query_text: str, entity_type: str = None,
                       n_results: int = 5, novel_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform optimized semantic search across collections with caching.

        Args:
            query_text: Text to search for
            entity_type: Optional entity type to limit search
            n_results: Number of results to return
            novel_id: Optional novel filter

        Returns:
            List of search results with metadata
        """
        # Check cache first
        cache_key = self._get_cache_key("search", query=query_text, entity_type=entity_type,
                                      n_results=n_results, novel_id=novel_id)

        with self.cache_lock:
            if self.enable_caching and self.query_cache and cache_key in self.query_cache:
                # Move to end (LRU)
                result = self.query_cache.pop(cache_key)
                self.query_cache[cache_key] = result
                self.cache_stats['query_hits'] += 1
                return result

        self.cache_stats['query_misses'] += 1
        start_time = time.time()

        try:
            results = []

            # Determine collections to search
            collections_to_search = [entity_type] if entity_type else self.entity_types

            for collection_name in collections_to_search:
                if collection_name not in self.entity_types:
                    continue

                collection = self.collections[collection_name]

                try:
                    # Build where clause for novel filtering
                    where_clause = None
                    if novel_id:
                        where_clause = {"$and": [{"entity_type": {"$eq": collection_name}}]}

                    # Perform semantic search
                    search_results = collection.query(
                        query_texts=[query_text],
                        n_results=min(n_results * 2, 50),  # Get more results for filtering
                        where=where_clause
                    )

                    # Process results
                    if search_results['ids'] and search_results['ids'][0]:
                        for i, entity_id in enumerate(search_results['ids'][0]):
                            metadata = search_results.get('metadatas', [[]])[0][i] if search_results.get('metadatas') else {}

                            # Filter by novel_id if specified
                            if novel_id and metadata.get('novel_id') != novel_id:
                                continue

                            result = {
                                'entity_type': collection_name,
                                'entity_id': entity_id,
                                'distance': search_results.get('distances', [[]])[0][i] if search_results.get('distances') else None,
                                'document': search_results.get('documents', [[]])[0][i] if search_results.get('documents') else None,
                                'metadata': metadata
                            }
                            results.append(result)

                except Exception as e:
                    logger.warning(f"Search failed for collection {collection_name}: {e}")
                    continue

            # Sort by distance (lower is better)
            if results and results[0].get('distance') is not None:
                results.sort(key=lambda x: x['distance'] or float('inf'))

            # Limit results
            final_results = results[:n_results]

            # Track performance
            search_time = time.time() - start_time
            self.performance_metrics['search_times'].append(search_time)

            # Cache result
            with self.cache_lock:
                if self.enable_caching and self.query_cache is not None:
                    # Evict oldest if at capacity
                    if len(self.query_cache) >= self.cache_size:
                        self.query_cache.popitem(last=False)
                        self.cache_stats['evictions'] += 1

                    self.query_cache[cache_key] = final_results

            return final_results

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []

    # ===== UTILITY METHODS =====

    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for operation."""
        key_parts = [operation]
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            key_parts.append(f"{k}:{v}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def batch_add_or_update(self, entity_type: str, entities: List[Tuple[str, Dict[str, Any], str]]) -> bool:
        """
        Batch add or update multiple entities for better performance.

        Args:
            entity_type: Type of entities
            entities: List of (entity_id, data, embedding_text) tuples

        Returns:
            True if successful, False otherwise
        """
        if not entities:
            return True

        start_time = time.time()

        try:
            if entity_type not in self.entity_types:
                raise ValueError(f"Invalid entity type: {entity_type}")

            collection = self.collections[entity_type]

            # Prepare batch data
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for entity_id, data, embedding_text in entities:
                # Prepare embedding text
                if embedding_text is None:
                    embedding_text = self._prepare_embedding_text(entity_type, data)

                # Generate embedding
                embedding = self._generate_embedding(embedding_text)

                # Prepare metadata
                metadata = {
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'updated_at': datetime.now().isoformat(),
                    'name': data.get('name') or data.get('title', ''),
                    'tags': ','.join(data.get('tags', [])) if data.get('tags') else '',
                    'novel_id': data.get('novel_id', '')
                }

                ids.append(entity_id)
                documents.append(embedding_text)
                embeddings.append(embedding)
                metadatas.append(metadata)

            # Check which entities already exist
            try:
                existing = collection.get(ids=ids)
                existing_ids = set(existing['ids']) if existing['ids'] else set()

                # Separate into updates and new additions
                update_indices = [i for i, entity_id in enumerate(ids) if entity_id in existing_ids]
                add_indices = [i for i, entity_id in enumerate(ids) if entity_id not in existing_ids]

                # Batch update existing entities
                if update_indices:
                    update_ids = [ids[i] for i in update_indices]
                    update_documents = [documents[i] for i in update_indices]
                    update_embeddings = [embeddings[i] for i in update_indices]
                    update_metadatas = [metadatas[i] for i in update_indices]

                    collection.update(
                        ids=update_ids,
                        documents=update_documents,
                        embeddings=update_embeddings,
                        metadatas=update_metadatas
                    )

                # Batch add new entities
                if add_indices:
                    add_ids = [ids[i] for i in add_indices]
                    add_documents = [documents[i] for i in add_indices]
                    add_embeddings = [embeddings[i] for i in add_indices]
                    add_metadatas = [metadatas[i] for i in add_indices]

                    collection.add(
                        ids=add_ids,
                        documents=add_documents,
                        embeddings=add_embeddings,
                        metadatas=add_metadatas
                    )

            except Exception:
                # Fallback: try to add all as new
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

            # Track performance
            batch_time = time.time() - start_time
            self.performance_metrics['batch_times'].append(batch_time)

            logger.info(f"Batch processed {len(entities)} {entity_type} entities in {batch_time:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to batch process {entity_type} entities: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ChromaDB collections.

        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {}

            for entity_type in self.entity_types:
                try:
                    collection = self.collections[entity_type]
                    count = collection.count()
                    stats[f'{entity_type}_embeddings'] = count
                except Exception as e:
                    logger.warning(f"Failed to get stats for {entity_type}: {e}")
                    stats[f'{entity_type}_embeddings'] = 0

            stats['total_embeddings'] = sum(stats.values())
            return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        embedding_hit_rate = 0.0
        if self.cache_stats['embedding_hits'] + self.cache_stats['embedding_misses'] > 0:
            embedding_hit_rate = self.cache_stats['embedding_hits'] / (
                self.cache_stats['embedding_hits'] + self.cache_stats['embedding_misses'])

        query_hit_rate = 0.0
        if self.cache_stats['query_hits'] + self.cache_stats['query_misses'] > 0:
            query_hit_rate = self.cache_stats['query_hits'] / (
                self.cache_stats['query_hits'] + self.cache_stats['query_misses'])

        return {
            'cache_stats': self.cache_stats,
            'embedding_hit_rate': embedding_hit_rate,
            'query_hit_rate': query_hit_rate,
            'embedding_times': self.performance_metrics['embedding_times'][-100:],  # Last 100
            'search_times': self.performance_metrics['search_times'][-100:],
            'batch_times': self.performance_metrics['batch_times'][-100:],
            'cache_sizes': {
                'embedding_cache': len(self.embedding_cache) if self.embedding_cache else 0,
                'query_cache': len(self.query_cache) if self.query_cache else 0
            }
        }

    def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries matching pattern."""
        with self.cache_lock:
            if pattern is None:
                # Clear all caches
                if self.embedding_cache:
                    self.embedding_cache.clear()
                if self.query_cache:
                    self.query_cache.clear()
            else:
                # Clear matching query cache entries
                if self.query_cache:
                    keys_to_remove = [k for k in self.query_cache.keys() if pattern in k]
                    for key in keys_to_remove:
                        del self.query_cache[key]

    def close(self):
        """Close ChromaDB connections and cleanup resources."""
        try:
            # Clear caches to free memory
            with self.cache_lock:
                if self.embedding_cache:
                    self.embedding_cache.clear()
                if self.query_cache:
                    self.query_cache.clear()

            # ChromaDB handles persistence automatically
            logger.info("ChromaDB connections closed and resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing ChromaDB connections: {e}")
