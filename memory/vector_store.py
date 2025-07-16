"""Vector store implementation using ChromaDB for semantic memory."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel

from config import settings
from utils.exceptions import MemoryError
from utils.logging import get_agent_logger


class VectorMemoryEntry(BaseModel):
    """Structured representation of a vector memory entry."""
    
    id: str
    content: str
    tags: List[str]
    source_agent: str
    timestamp: datetime
    project_id: str
    metadata: Dict[str, Any]


class VectorStore:
    """Vector store for semantic memory using ChromaDB."""
    
    def __init__(self, collection_name: str = "agent_memory"):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.logger = get_agent_logger("VectorStore")
        self.collection_name = collection_name
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(settings.chromadb_persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Semantic memory for Data Science Agent"},
            )
            
            self.logger.info(f"Initialized vector store with collection: {collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise MemoryError(f"Vector store initialization failed: {e}")
    
    def store_memory(
        self,
        content: str,
        tags: List[str],
        source_agent: str,
        project_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Store a memory entry in the vector store.
        
        Args:
            content: The text content to store
            tags: List of tags for categorization
            source_agent: Agent that created this memory
            project_id: Project identifier for filtering
            metadata: Additional metadata
            memory_id: Optional custom ID (generated if not provided)
            
        Returns:
            The ID of the stored memory entry
            
        Raises:
            MemoryError: If storage fails
        """
        try:
            if memory_id is None:
                memory_id = f"{source_agent}_{datetime.utcnow().isoformat()}_{hash(content) % 10000}"
            
            if metadata is None:
                metadata = {}
            
            # Prepare metadata for ChromaDB
            chroma_metadata = {
                "source_agent": source_agent,
                "project_id": project_id,
                "timestamp": datetime.utcnow().isoformat(),
                "tags": json.dumps(tags),
                **metadata,
            }
            
            # Store in ChromaDB
            self.collection.add(
                documents=[content],
                metadatas=[chroma_metadata],
                ids=[memory_id],
            )
            
            self.logger.info(
                f"Stored memory entry: {memory_id}",
                source_agent=source_agent,
                tags=tags,
                project_id=project_id,
            )
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise MemoryError(f"Memory storage failed: {e}")
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        project_id: Optional[str] = None,
        source_agent: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[VectorMemoryEntry]:
        """Search for similar memories using semantic similarity.
        
        Args:
            query: Query text for similarity search
            top_k: Number of results to return
            project_id: Optional project filter
            source_agent: Optional agent filter
            tags: Optional tag filters
            
        Returns:
            List of similar memory entries
            
        Raises:
            MemoryError: If search fails
        """
        try:
            # Build where clause for filtering
            where_clause = {}
            if project_id:
                where_clause["project_id"] = project_id
            if source_agent:
                where_clause["source_agent"] = source_agent
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_clause if where_clause else None,
            )
            
            # Convert results to VectorMemoryEntry objects
            entries = []
            if results["ids"] and results["ids"][0]:
                for i, memory_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    content = results["documents"][0][i]
                    
                    # Parse tags from JSON
                    tags_list = json.loads(metadata.get("tags", "[]"))
                    
                    # Filter by tags if specified
                    if tags and not any(tag in tags_list for tag in tags):
                        continue
                    
                    entry = VectorMemoryEntry(
                        id=memory_id,
                        content=content,
                        tags=tags_list,
                        source_agent=metadata["source_agent"],
                        timestamp=datetime.fromisoformat(metadata["timestamp"]),
                        project_id=metadata["project_id"],
                        metadata={k: v for k, v in metadata.items() 
                                if k not in ["source_agent", "project_id", "timestamp", "tags"]},
                    )
                    entries.append(entry)
            
            self.logger.info(
                f"Found {len(entries)} similar memories for query: {query[:50]}...",
                top_k=top_k,
                filters={"project_id": project_id, "source_agent": source_agent, "tags": tags},
            )
            
            return entries
            
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            raise MemoryError(f"Memory search failed: {e}")
    
    def get_memory(self, memory_id: str) -> Optional[VectorMemoryEntry]:
        """Retrieve a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory entry if found, None otherwise
            
        Raises:
            MemoryError: If retrieval fails
        """
        try:
            results = self.collection.get(ids=[memory_id])
            
            if not results["ids"] or not results["ids"][0]:
                return None
            
            metadata = results["metadatas"][0]
            content = results["documents"][0]
            tags = json.loads(metadata.get("tags", "[]"))
            
            return VectorMemoryEntry(
                id=memory_id,
                content=content,
                tags=tags,
                source_agent=metadata["source_agent"],
                timestamp=datetime.fromisoformat(metadata["timestamp"]),
                project_id=metadata["project_id"],
                metadata={k: v for k, v in metadata.items() 
                        if k not in ["source_agent", "project_id", "timestamp", "tags"]},
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get memory {memory_id}: {e}")
            raise MemoryError(f"Memory retrieval failed: {e}")
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            MemoryError: If deletion fails
        """
        try:
            self.collection.delete(ids=[memory_id])
            self.logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise MemoryError(f"Memory deletion failed: {e}")
    
    def clear_collection(self) -> None:
        """Clear all memories from the collection.
        
        Raises:
            MemoryError: If clearing fails
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Semantic memory for Data Science Agent"},
            )
            self.logger.info("Cleared all memories from vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            raise MemoryError(f"Collection clearing failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_entries": count,
                "last_updated": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}