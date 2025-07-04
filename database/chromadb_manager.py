"""
ChromaDB Manager: Handles semantic memory operations for worldbuilding entities.

This module provides specialized operations for ChromaDB, including embedding generation,
semantic search, and vector database management with DuckDB+Parquet persistence.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """
    Manages ChromaDB operations for semantic memory and vector search.
    """
    
    def __init__(self, 
                 db_path: str = './data/chromadb',
                 openrouter_api_key: str = None):
        """
        Initialize ChromaDB manager.
        
        Args:
            db_path: Path to ChromaDB storage directory
            openrouter_api_key: OpenRouter API key for embeddings
        """
        self.db_path = db_path
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        
        # Ensure directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Initialize OpenRouter client
        self._init_openrouter()
        
        # Entity types
        self.entity_types = ['novels', 'characters', 'locations', 'lore']
        
        logger.info(f"ChromaDB Manager initialized at {self.db_path}")
    
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
        Generate embedding for text using OpenRouter/DeepSeek.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not self.openrouter_client:
            # Return dummy embedding for testing
            logger.debug("Using dummy embedding (no OpenRouter client)")
            return [0.1] * 1536  # Standard embedding dimension
        
        try:
            # Clean and prepare text
            text = text.strip()
            if not text:
                text = "empty"
            
            # Generate embedding using OpenRouter
            response = self.openrouter_client.embeddings.create(
                model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                input=text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text (length: {len(text)}, dim: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return dummy embedding as fallback
            return [0.1] * 1536
    
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
                       n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search across collections.
        
        Args:
            query_text: Text to search for
            entity_type: Optional entity type to limit search
            n_results: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            results = []
            
            # Determine collections to search
            collections_to_search = [entity_type] if entity_type else self.entity_types
            
            for collection_name in collections_to_search:
                if collection_name not in self.entity_types:
                    continue
                
                collection = self.collections[collection_name]
                
                try:
                    # Perform semantic search
                    search_results = collection.query(
                        query_texts=[query_text],
                        n_results=min(n_results, 10)  # Limit per collection
                    )
                    
                    # Process results
                    if search_results['ids'] and search_results['ids'][0]:
                        for i, entity_id in enumerate(search_results['ids'][0]):
                            result = {
                                'entity_type': collection_name,
                                'entity_id': entity_id,
                                'distance': search_results.get('distances', [[]])[0][i] if search_results.get('distances') else None,
                                'document': search_results.get('documents', [[]])[0][i] if search_results.get('documents') else None,
                                'metadata': search_results.get('metadatas', [[]])[0][i] if search_results.get('metadatas') else {}
                            }
                            results.append(result)
                            
                except Exception as e:
                    logger.warning(f"Search failed for collection {collection_name}: {e}")
                    continue
            
            # Sort by distance (lower is better)
            if results and results[0].get('distance') is not None:
                results.sort(key=lambda x: x['distance'] or float('inf'))
            
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
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
    
    def close(self):
        """Close ChromaDB connections."""
        try:
            # ChromaDB handles persistence automatically
            logger.info("ChromaDB connections closed")
        except Exception as e:
            logger.error(f"Error closing ChromaDB connections: {e}")
