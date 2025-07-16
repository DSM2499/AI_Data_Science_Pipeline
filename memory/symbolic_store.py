"""Symbolic store implementation for structured memory using SQLite and JSON."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

from config import settings
from utils.exceptions import MemoryError
from utils.logging import get_agent_logger


class SymbolicMemoryEntry(BaseModel):
    """Structured representation of a symbolic memory entry."""
    
    id: str
    table_name: str
    data: Dict[str, Any]
    timestamp: datetime
    project_id: str
    source_agent: str


class SymbolicStore:
    """Symbolic store for structured memory using SQLite and file storage."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the symbolic store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.logger = get_agent_logger("SymbolicStore")
        
        if db_path is None:
            db_path = settings.memory_symbolic_path / "symbolic_memory.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._init_database()
            self.logger.info(f"Initialized symbolic store: {db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize symbolic store: {e}")
            raise MemoryError(f"Symbolic store initialization failed: {e}")
    
    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create main memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    source_agent TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_table_name ON memory_entries(table_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_id ON memory_entries(project_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_agent ON memory_entries(source_agent)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)")
            
            # Create specific tables for structured data
            self._create_structured_tables(cursor)
            
            conn.commit()
    
    def _create_structured_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create structured tables for specific data types."""
        
        # Past projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS past_projects (
                project_id TEXT PRIMARY KEY,
                dataset_shape TEXT,
                target_column TEXT,
                user_decisions TEXT,
                model_success TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                preference_id TEXT PRIMARY KEY,
                user_id TEXT,
                category TEXT,
                preference_key TEXT,
                preference_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feature history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_history (
                feature_id TEXT PRIMARY KEY,
                feature_name TEXT,
                feature_type TEXT,
                data_type TEXT,
                correlation_with_target REAL,
                importance_score REAL,
                always_dropped BOOLEAN,
                leakage_risk BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                performance_id TEXT PRIMARY KEY,
                project_id TEXT,
                model_name TEXT,
                hyperparameters TEXT,
                metrics TEXT,
                dataset_profile TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def store_memory(
        self,
        table_name: str,
        data: Dict[str, Any],
        source_agent: str,
        project_id: str = "default",
        memory_id: Optional[str] = None,
    ) -> str:
        """Store a memory entry in the symbolic store.
        
        Args:
            table_name: Table/category name for the data
            data: Dictionary of data to store
            source_agent: Agent that created this memory
            project_id: Project identifier
            memory_id: Optional custom ID
            
        Returns:
            The ID of the stored memory entry
            
        Raises:
            MemoryError: If storage fails
        """
        try:
            if memory_id is None:
                memory_id = f"{table_name}_{source_agent}_{datetime.utcnow().isoformat()}_{hash(str(data)) % 10000}"
            
            timestamp = datetime.utcnow().isoformat()
            data_json = json.dumps(data, default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_entries 
                    (id, table_name, project_id, source_agent, timestamp, data_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (memory_id, table_name, project_id, source_agent, timestamp, data_json))
                
                conn.commit()
            
            self.logger.info(
                f"Stored symbolic memory: {memory_id}",
                table_name=table_name,
                source_agent=source_agent,
                project_id=project_id,
            )
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store symbolic memory: {e}")
            raise MemoryError(f"Symbolic memory storage failed: {e}")
    
    def query_memory(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        source_agent: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[SymbolicMemoryEntry]:
        """Query memory entries from the symbolic store.
        
        Args:
            table_name: Table name to query
            filters: Optional filters to apply to the data
            project_id: Optional project filter
            source_agent: Optional agent filter
            limit: Optional limit on number of results
            
        Returns:
            List of matching memory entries
            
        Raises:
            MemoryError: If query fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = "SELECT id, table_name, project_id, source_agent, timestamp, data_json FROM memory_entries WHERE table_name = ?"
                params = [table_name]
                
                if project_id:
                    query += " AND project_id = ?"
                    params.append(project_id)
                
                if source_agent:
                    query += " AND source_agent = ?"
                    params.append(source_agent)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
            
            # Convert to SymbolicMemoryEntry objects
            entries = []
            for row in rows:
                memory_id, table_name, project_id, source_agent, timestamp, data_json = row
                data = json.loads(data_json)
                
                # Apply data filters if specified
                if filters and not self._matches_filters(data, filters):
                    continue
                
                entry = SymbolicMemoryEntry(
                    id=memory_id,
                    table_name=table_name,
                    data=data,
                    timestamp=datetime.fromisoformat(timestamp),
                    project_id=project_id,
                    source_agent=source_agent,
                )
                entries.append(entry)
            
            self.logger.info(
                f"Queried {len(entries)} entries from table: {table_name}",
                filters=filters,
                project_id=project_id,
                source_agent=source_agent,
            )
            
            return entries
            
        except Exception as e:
            self.logger.error(f"Failed to query symbolic memory: {e}")
            raise MemoryError(f"Symbolic memory query failed: {e}")
    
    def _matches_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if data matches the given filters.
        
        Args:
            data: Data dictionary to check
            filters: Filters to apply
            
        Returns:
            True if data matches all filters
        """
        for key, value in filters.items():
            if key not in data:
                return False
            
            if isinstance(value, list):
                if data[key] not in value:
                    return False
            elif data[key] != value:
                return False
        
        return True
    
    def get_memory(self, memory_id: str) -> Optional[SymbolicMemoryEntry]:
        """Retrieve a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory entry if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, table_name, project_id, source_agent, timestamp, data_json 
                    FROM memory_entries WHERE id = ?
                """, (memory_id,))
                row = cursor.fetchone()
            
            if not row:
                return None
            
            memory_id, table_name, project_id, source_agent, timestamp, data_json = row
            data = json.loads(data_json)
            
            return SymbolicMemoryEntry(
                id=memory_id,
                table_name=table_name,
                data=data,
                timestamp=datetime.fromisoformat(timestamp),
                project_id=project_id,
                source_agent=source_agent,
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get memory {memory_id}: {e}")
            raise MemoryError(f"Memory retrieval failed: {e}")
    
    def store_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        source_agent: str,
        project_id: str = "default",
    ) -> str:
        """Store a pandas DataFrame as a parquet file.
        
        Args:
            df: DataFrame to store
            table_name: Table name for the data
            source_agent: Agent that created this data
            project_id: Project identifier
            
        Returns:
            File path of the stored DataFrame
        """
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{table_name}_{source_agent}_{timestamp}.parquet"
            file_path = settings.memory_symbolic_path / "dataframes" / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_parquet(file_path)
            
            # Store metadata in the main database
            metadata = {
                "file_path": str(file_path),
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            }
            
            self.store_memory(
                table_name=f"dataframe_{table_name}",
                data=metadata,
                source_agent=source_agent,
                project_id=project_id,
            )
            
            self.logger.info(
                f"Stored DataFrame: {filename}",
                shape=df.shape,
                source_agent=source_agent,
            )
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to store DataFrame: {e}")
            raise MemoryError(f"DataFrame storage failed: {e}")
    
    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        """Load a DataFrame from a parquet file.
        
        Args:
            file_path: Path to the parquet file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_parquet(file_path)
            self.logger.info(f"Loaded DataFrame: {file_path}", shape=df.shape)
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load DataFrame {file_path}: {e}")
            raise MemoryError(f"DataFrame loading failed: {e}")
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM memory_entries WHERE id = ?", (memory_id,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
            
            if deleted:
                self.logger.info(f"Deleted symbolic memory: {memory_id}")
            else:
                self.logger.warning(f"Memory not found for deletion: {memory_id}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise MemoryError(f"Memory deletion failed: {e}")
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
                    FROM memory_entries WHERE table_name = ?
                """, (table_name,))
                count, min_time, max_time = cursor.fetchone()
            
            return {
                "table_name": table_name,
                "total_entries": count or 0,
                "earliest_entry": min_time,
                "latest_entry": max_time,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get table stats for {table_name}: {e}")
            return {"error": str(e)}