"""Memory client that provides a unified interface to both vector and symbolic stores."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from memory.symbolic_store import SymbolicMemoryEntry, SymbolicStore
from memory.vector_store import VectorMemoryEntry, VectorStore
from utils.exceptions import MemoryError
from utils.logging import get_agent_logger


class MemoryClient:
    """Unified interface for accessing both vector and symbolic memory stores."""
    
    def __init__(self):
        """Initialize the memory client with both stores."""
        self.logger = get_agent_logger("MemoryClient")
        
        try:
            self.vector_store = VectorStore()
            self.symbolic_store = SymbolicStore()
            self.logger.info("Memory client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory client: {e}")
            raise MemoryError(f"Memory client initialization failed: {e}")
    
    # Vector store methods
    def store_vector(
        self,
        content: str,
        tags: List[str],
        source_agent: str,
        project_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store content in the vector store for semantic search.
        
        Args:
            content: Text content to store
            tags: Tags for categorization
            source_agent: Agent that created this memory
            project_id: Project identifier
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        return self.vector_store.store_memory(
            content=content,
            tags=tags,
            source_agent=source_agent,
            project_id=project_id,
            metadata=metadata,
        )
    
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
            query: Query text
            top_k: Number of results
            project_id: Optional project filter
            source_agent: Optional agent filter
            tags: Optional tag filters
            
        Returns:
            List of similar memory entries
        """
        return self.vector_store.search_similar(
            query=query,
            top_k=top_k,
            project_id=project_id,
            source_agent=source_agent,
            tags=tags,
        )
    
    # Symbolic store methods
    def store_symbolic(
        self,
        table_name: str,
        data: Dict[str, Any],
        source_agent: str,
        project_id: str = "default",
    ) -> str:
        """Store structured data in the symbolic store.
        
        Args:
            table_name: Table/category name
            data: Structured data to store
            source_agent: Agent that created this memory
            project_id: Project identifier
            
        Returns:
            Memory ID
        """
        return self.symbolic_store.store_memory(
            table_name=table_name,
            data=data,
            source_agent=source_agent,
            project_id=project_id,
        )
    
    def query_symbolic(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        source_agent: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SymbolicMemoryEntry]:
        """Query structured data from the symbolic store.
        
        Args:
            table_name: Table name to query
            filters: Optional data filters
            project_id: Optional project filter
            source_agent: Optional agent filter
            limit: Optional result limit
            
        Returns:
            List of matching memory entries
        """
        return self.symbolic_store.query_memory(
            table_name=table_name,
            filters=filters,
            project_id=project_id,
            source_agent=source_agent,
            limit=limit,
        )
    
    # DataFrame operations
    def store_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        source_agent: str,
        project_id: str = "default",
    ) -> str:
        """Store a DataFrame in the symbolic store.
        
        Args:
            df: DataFrame to store
            table_name: Table name
            source_agent: Agent that created this data
            project_id: Project identifier
            
        Returns:
            File path of stored DataFrame
        """
        return self.symbolic_store.store_dataframe(
            df=df,
            table_name=table_name,
            source_agent=source_agent,
            project_id=project_id,
        )
    
    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        """Load a DataFrame from storage.
        
        Args:
            file_path: Path to the DataFrame file
            
        Returns:
            Loaded DataFrame
        """
        return self.symbolic_store.load_dataframe(file_path)
    
    # High-level convenience methods
    def store_dataset_profile(
        self,
        dataset_metadata: Dict[str, Any],
        source_agent: str,
        project_id: str = "default",
    ) -> str:
        """Store dataset profiling information.
        
        Args:
            dataset_metadata: Dataset metadata and profile
            source_agent: Agent that created this profile
            project_id: Project identifier
            
        Returns:
            Memory ID
        """
        # Store in symbolic store
        symbolic_id = self.store_symbolic(
            table_name="dataset_profiles",
            data=dataset_metadata,
            source_agent=source_agent,
            project_id=project_id,
        )
        
        # Create semantic description for vector store
        description = f"""
        Dataset profile for project {project_id}:
        - Shape: {dataset_metadata.get('shape', 'unknown')}
        - Columns: {len(dataset_metadata.get('columns', []))}
        - Target: {dataset_metadata.get('target_column', 'unknown')}
        - Missing values: {dataset_metadata.get('missing_percentage', 'unknown')}%
        - Data types: {', '.join(dataset_metadata.get('data_types', {}).keys())}
        """
        
        # Store in vector store for semantic search
        vector_id = self.store_vector(
            content=description.strip(),
            tags=["dataset_profile", "data_summary"],
            source_agent=source_agent,
            project_id=project_id,
            metadata={"symbolic_id": symbolic_id},
        )
        
        return symbolic_id
    
    def store_feature_success(
        self,
        feature_name: str,
        performance_metrics: Dict[str, Any],
        dataset_context: Dict[str, Any],
        source_agent: str,
        project_id: str = "default",
    ) -> str:
        """Store feature engineering success/failure information.
        
        Args:
            feature_name: Name of the feature
            performance_metrics: Performance impact metrics
            dataset_context: Context about the dataset
            source_agent: Agent that created this feature
            project_id: Project identifier
            
        Returns:
            Memory ID
        """
        feature_data = {
            "feature_name": feature_name,
            "performance_metrics": performance_metrics,
            "dataset_context": dataset_context,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Store in symbolic store
        symbolic_id = self.store_symbolic(
            table_name="feature_history",
            data=feature_data,
            source_agent=source_agent,
            project_id=project_id,
        )
        
        # Create semantic description
        impact = performance_metrics.get("improvement", 0)
        metric_name = performance_metrics.get("metric", "performance")
        
        description = f"""
        Feature '{feature_name}' engineering result:
        - {metric_name} impact: {impact:+.3f}
        - Dataset type: {dataset_context.get('task_type', 'unknown')}
        - Feature type: {dataset_context.get('feature_type', 'unknown')}
        - Successful: {'Yes' if impact > 0 else 'No'}
        """
        
        tags = ["feature_engineering", "performance_impact"]
        if impact > 0:
            tags.append("successful_feature")
        else:
            tags.append("failed_feature")
        
        # Store in vector store
        vector_id = self.store_vector(
            content=description.strip(),
            tags=tags,
            source_agent=source_agent,
            project_id=project_id,
            metadata={"symbolic_id": symbolic_id, "feature_name": feature_name},
        )
        
        return symbolic_id
    
    def store_model_performance(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_profile: Dict[str, Any],
        source_agent: str,
        project_id: str = "default",
    ) -> str:
        """Store model performance information.
        
        Args:
            model_name: Name of the model
            hyperparameters: Model hyperparameters
            metrics: Performance metrics
            dataset_profile: Dataset characteristics
            source_agent: Agent that trained this model
            project_id: Project identifier
            
        Returns:
            Memory ID
        """
        model_data = {
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "dataset_profile": dataset_profile,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Store in symbolic store
        symbolic_id = self.store_symbolic(
            table_name="model_performance",
            data=model_data,
            source_agent=source_agent,
            project_id=project_id,
        )
        
        # Create semantic description
        primary_metric = list(metrics.keys())[0] if metrics else "unknown"
        primary_score = metrics.get(primary_metric, 0)
        
        description = f"""
        Model '{model_name}' performance:
        - {primary_metric}: {primary_score:.3f}
        - Dataset shape: {dataset_profile.get('shape', 'unknown')}
        - Task type: {dataset_profile.get('task_type', 'unknown')}
        - Feature count: {dataset_profile.get('feature_count', 'unknown')}
        - All metrics: {', '.join(f'{k}={v:.3f}' for k, v in metrics.items())}
        """
        
        # Store in vector store
        vector_id = self.store_vector(
            content=description.strip(),
            tags=["model_performance", "evaluation", model_name.lower()],
            source_agent=source_agent,
            project_id=project_id,
            metadata={"symbolic_id": symbolic_id, "model_name": model_name},
        )
        
        return symbolic_id
    
    def get_similar_datasets(
        self,
        current_profile: Dict[str, Any],
        top_k: int = 3,
    ) -> List[VectorMemoryEntry]:
        """Find datasets similar to the current one.
        
        Args:
            current_profile: Current dataset profile
            top_k: Number of similar datasets to return
            
        Returns:
            List of similar dataset memories
        """
        query = f"""
        Dataset with {current_profile.get('shape', 'unknown')} shape,
        {len(current_profile.get('columns', []))} columns,
        target column type: {current_profile.get('target_type', 'unknown')},
        missing values: {current_profile.get('missing_percentage', 0)}%
        """
        
        return self.search_similar(
            query=query.strip(),
            top_k=top_k,
            tags=["dataset_profile"],
        )
    
    def get_successful_features_for_task(
        self,
        task_type: str,
        dataset_characteristics: Dict[str, Any],
        top_k: int = 5,
    ) -> List[VectorMemoryEntry]:
        """Get successful features for similar tasks and datasets.
        
        Args:
            task_type: Type of ML task (classification, regression, etc.)
            dataset_characteristics: Characteristics of the current dataset
            top_k: Number of features to return
            
        Returns:
            List of successful feature memories
        """
        query = f"""
        Successful features for {task_type} task with similar dataset characteristics:
        feature engineering, successful, {task_type}
        """
        
        return self.search_similar(
            query=query,
            top_k=top_k,
            tags=["feature_engineering", "successful_feature"],
        )
    
    def get_best_models_for_dataset(
        self,
        dataset_profile: Dict[str, Any],
        top_k: int = 3,
    ) -> List[VectorMemoryEntry]:
        """Get best performing models for similar datasets.
        
        Args:
            dataset_profile: Current dataset profile
            top_k: Number of models to return
            
        Returns:
            List of model performance memories
        """
        task_type = dataset_profile.get("task_type", "unknown")
        feature_count = dataset_profile.get("feature_count", 0)
        
        query = f"""
        Best performing models for {task_type} with approximately {feature_count} features,
        dataset shape similar to {dataset_profile.get('shape', 'unknown')}
        """
        
        return self.search_similar(
            query=query,
            top_k=top_k,
            tags=["model_performance"],
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about both memory stores.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            # Get symbolic store stats for main tables
            symbolic_stats = {}
            main_tables = [
                "dataset_profiles",
                "feature_history", 
                "model_performance",
                "user_preferences",
            ]
            
            for table in main_tables:
                symbolic_stats[table] = self.symbolic_store.get_table_stats(table)
            
            return {
                "vector_store": vector_stats,
                "symbolic_store": symbolic_stats,
                "last_updated": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}


# Global memory client instance
memory_client = MemoryClient()