"""
WorldState: A unified interface for managing structured (TinyDB) and semantic (ChromaDB) data.

This class provides a clean abstraction layer that handles synchronization between
TinyDB for structured worldbuilding data and ChromaDB for semantic memory.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

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
    """
    
    def __init__(self, 
                 tinydb_path: str = None, 
                 chromadb_path: str = None,
                 openrouter_api_key: str = None):
        """
        Initialize WorldState with database connections.
        
        Args:
            tinydb_path: Path to TinyDB storage directory
            chromadb_path: Path to ChromaDB storage directory  
            openrouter_api_key: OpenRouter API key for embeddings
        """
        # Configuration
        self.tinydb_path = tinydb_path or os.getenv('TINYDB_PATH', './data/tinydb')
        self.chromadb_path = chromadb_path or os.getenv('CHROMADB_PATH', './data/chromadb')
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')

        # Entity types - define before initialization
        self.entity_types = ['novels', 'characters', 'locations', 'lore']

        # Ensure directories exist
        Path(self.tinydb_path).mkdir(parents=True, exist_ok=True)
        Path(self.chromadb_path).mkdir(parents=True, exist_ok=True)

        # Initialize databases
        self._init_tinydb()
        self._init_chromadb()
        self._init_openrouter()
        
        logger.info("WorldState initialized successfully")
    
    def _init_tinydb(self):
        """Initialize TinyDB connections for each entity type."""
        self.tinydb_connections = {}
        
        for entity_type in ['novels', 'characters', 'locations', 'lore']:
            db_path = os.path.join(self.tinydb_path, f'{entity_type}.json')
            self.tinydb_connections[entity_type] = TinyDB(db_path)
            
        logger.info(f"TinyDB initialized at {self.tinydb_path}")
    
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
        Generate embedding for text using OpenRouter/DeepSeek.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not self.openrouter_client:
            logger.warning("OpenRouter client not available, returning dummy embedding")
            # Return a dummy embedding for testing
            return [0.0] * 1536
            
        try:
            response = self.openrouter_client.embeddings.create(
                model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return dummy embedding as fallback
            return [0.0] * 1536
    
    def _prepare_embedding_text(self, entity_type: str, data: Dict[str, Any]) -> str:
        """
        Prepare text for embedding based on entity type.
        
        Args:
            entity_type: Type of entity (novels, characters, locations, lore)
            data: Entity data
            
        Returns:
            Formatted text for embedding
        """
        if entity_type == 'novels':
            return f"Novel: {data.get('title', '')}. Genre: {data.get('genre', '')}. Description: {data.get('description', '')}"
            
        elif entity_type == 'characters':
            return f"Character: {data.get('name', '')}. Occupation: {data.get('occupation', '')}. Description: {data.get('description', '')}. Personality: {data.get('personality', '')}"
            
        elif entity_type == 'locations':
            return f"Location: {data.get('name', '')}. Type: {data.get('type', '')}. Description: {data.get('description', '')}. Culture: {data.get('culture', '')}"
            
        elif entity_type == 'lore':
            return f"Lore: {data.get('title', '')}. Category: {data.get('category', '')}. Description: {data.get('description', '')}. Details: {data.get('details', '')}"
            
        else:
            return str(data)
    
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

    def close(self):
        """Close database connections."""
        try:
            # Close TinyDB connections
            for db in self.tinydb_connections.values():
                db.close()

            logger.info("WorldState connections closed")

        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False  # Don't suppress exceptions
