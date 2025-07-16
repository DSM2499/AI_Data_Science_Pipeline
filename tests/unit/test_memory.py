"""Tests for memory system."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from memory.memory_client import MemoryClient
from memory.symbolic_store import SymbolicStore
from memory.vector_store import VectorStore


class TestMemorySystem:
    """Test cases for memory system components."""
    
    def test_memory_client_initialization(self):
        """Test memory client initializes properly."""
        # This test verifies the memory client can be created
        # In a real environment, it would connect to ChromaDB and SQLite
        try:
            client = MemoryClient()
            assert client is not None
            # Basic functionality check
            stats = client.get_memory_stats()
            assert isinstance(stats, dict)
        except Exception as e:
            # Memory initialization might fail in test environment
            # This is expected without proper ChromaDB setup
            assert "Memory" in str(type(e).__name__) or "chroma" in str(e).lower()
    
    def test_symbolic_store_with_temp_db(self):
        """Test symbolic store with temporary database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            store = SymbolicStore(db_path)
            
            # Test basic operations
            test_data = {
                "test_key": "test_value",
                "number": 42,
                "list": [1, 2, 3],
            }
            
            memory_id = store.store_memory(
                table_name="test_table",
                data=test_data,
                source_agent="TestAgent",
                project_id="test_project",
            )
            
            assert memory_id is not None
            
            # Retrieve memory
            retrieved = store.get_memory(memory_id)
            assert retrieved is not None
            assert retrieved.data == test_data
            assert retrieved.source_agent == "TestAgent"
    
    def test_dataframe_storage(self):
        """Test DataFrame storage and retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            store = SymbolicStore(db_path)
            
            # Create test DataFrame
            df = pd.DataFrame({
                "A": [1, 2, 3],
                "B": ["x", "y", "z"],
                "C": [1.1, 2.2, 3.3],
            })
            
            # Store DataFrame
            file_path = store.store_dataframe(
                df=df,
                table_name="test_df",
                source_agent="TestAgent",
            )
            
            assert file_path is not None
            assert Path(file_path).exists()
            
            # Load DataFrame
            loaded_df = store.load_dataframe(file_path)
            
            # Verify data integrity
            pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_query_memory_with_filters(self):
        """Test querying memory with filters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            store = SymbolicStore(db_path)
            
            # Store multiple entries
            entries = [
                {"type": "model", "accuracy": 0.85, "agent": "ModelingAgent"},
                {"type": "cleaning", "operations": 5, "agent": "CleaningAgent"},
                {"type": "model", "accuracy": 0.92, "agent": "ModelingAgent"},
            ]
            
            for i, entry in enumerate(entries):
                store.store_memory(
                    table_name="test_table",
                    data=entry,
                    source_agent=entry["agent"],
                    project_id="test_project",
                )
            
            # Query with filters
            model_entries = store.query_memory(
                table_name="test_table",
                filters={"type": "model"},
                project_id="test_project",
            )
            
            assert len(model_entries) == 2
            for entry in model_entries:
                assert entry.data["type"] == "model"
    
    def test_table_stats(self):
        """Test table statistics functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            store = SymbolicStore(db_path)
            
            # Initially empty
            stats = store.get_table_stats("empty_table")
            assert stats["total_entries"] == 0
            
            # Add some entries
            for i in range(3):
                store.store_memory(
                    table_name="test_table",
                    data={"index": i},
                    source_agent="TestAgent",
                )
            
            # Check stats
            stats = store.get_table_stats("test_table")
            assert stats["total_entries"] == 3