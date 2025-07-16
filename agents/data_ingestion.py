"""Data Ingestion Agent - Validates and ingests structured data files."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from agents.base import AgentContext, AgentResult, BaseAgent, UserFeedback
from memory.memory_client import memory_client
from utils.exceptions import DataIngestionError, ValidationError
from utils.logging import get_agent_logger


class DataIngestionResult(BaseModel):
    """Result of data ingestion process."""
    
    file_path: str
    file_hash: str
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    target_column: Optional[str]
    null_counts: Dict[str, int]
    memory_usage_mb: float


class DataIngestionAgent(BaseAgent):
    """Agent responsible for validating and ingesting structured data files."""
    
    def __init__(self):
        """Initialize the data ingestion agent."""
        super().__init__("DataIngestionAgent")
        
        # Supported file formats
        self.supported_formats = {".csv", ".xlsx", ".xls", ".parquet"}
        
        # Validation thresholds
        self.min_rows = 10
        self.min_columns = 2
        self.max_null_percentage = 95.0
    
    def run(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> AgentResult:
        """Execute data ingestion process.
        
        Args:
            context: Pipeline context containing file path and metadata
            feedback: Optional user feedback for refinement
            
        Returns:
            AgentResult with ingested data and metadata
        """
        try:
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            # Handle user feedback
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            # Extract file path and target from metadata
            file_path = context.metadata.get("uploaded_file")
            target_variable = context.metadata.get("target_variable")
            
            if not file_path:
                raise DataIngestionError("No file path provided in context metadata")
            
            self.logger.info(f"Starting data ingestion for file: {file_path}")
            
            # Load the data
            data = self._load_data(file_path)
            
            # Validate the loaded data
            self._validate_data(data, target_variable)
            
            # Generate file hash for tracking
            file_hash = self._calculate_file_hash(file_path)
            
            # Extract metadata
            ingestion_result = self._extract_metadata(data, file_path, file_hash, target_variable)
            
            # Update context
            context.data = data
            context.target = target_variable
            context.metadata.update({
                "ingestion_result": ingestion_result.dict(),
                "file_hash": file_hash,
                "original_shape": data.shape,
                "original_columns": list(data.columns),
            })
            
            # Store ingestion information in memory
            self._store_ingestion_memory(ingestion_result, context)
            
            # Check if similar datasets exist
            similar_datasets = self._find_similar_datasets(ingestion_result)
            
            suggestions = []
            if similar_datasets:
                suggestions.append(
                    f"Found {len(similar_datasets)} similar datasets in memory. "
                    f"Consider reviewing past successful approaches."
                )
            
            result = AgentResult(
                success=True,
                agent_name=self.name,
                context=context,
                logs=[
                    f"Successfully loaded data from {file_path}",
                    f"Data shape: {data.shape}",
                    f"Target column: {target_variable}",
                    f"File hash: {file_hash}",
                ],
                metrics={
                    "rows": data.shape[0],
                    "columns": data.shape[1],
                    "memory_usage_mb": ingestion_result.memory_usage_mb,
                    "null_percentage": (data.isnull().sum().sum() / data.size) * 100,
                },
                artifacts={"ingestion_result": ingestion_result.dict()},
                suggestions=suggestions,
                requires_approval=False,  # Auto-proceed unless issues found
            )
            
            # Add warnings if data quality issues detected
            warnings = self._detect_data_quality_issues(data, target_variable)
            if warnings:
                result.warnings = warnings
                result.requires_approval = True
                result.user_message = (
                    f"Data ingested successfully, but {len(warnings)} potential issues detected. "
                    f"Please review the warnings and approve to continue."
                )
            
            self._log_execution(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Data ingestion failed: {e}")
            
            result = AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=str(e),
                logs=[f"Data ingestion failed: {e}"],
                user_message="Data ingestion failed. Please check the file format and try again.",
            )
            
            self._log_execution(result)
            return result
    
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate that required inputs are provided.
        
        Args:
            context: Pipeline context to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValidationError: If validation fails
        """
        if not context.metadata.get("uploaded_file"):
            raise ValidationError("No uploaded file path provided")
        
        file_path = Path(context.metadata["uploaded_file"])
        
        if not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValidationError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        return True
    
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file based on file extension.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataIngestionError: If loading fails
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == ".csv":
                # Try different encodings and separators
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        self.logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise DataIngestionError("Could not read CSV file with any supported encoding")
            
            elif extension in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                
            elif extension == ".parquet":
                df = pd.read_parquet(file_path)
                
            else:
                raise DataIngestionError(f"Unsupported file format: {extension}")
            
            self.logger.info(f"Loaded data with shape: {df.shape}")
            return df
            
        except Exception as e:
            raise DataIngestionError(f"Failed to load file {file_path}: {e}")
    
    def _validate_data(self, data: pd.DataFrame, target_variable: Optional[str] = None) -> None:
        """Validate the loaded data meets minimum requirements.
        
        Args:
            data: Loaded DataFrame
            target_variable: Optional target column name
            
        Raises:
            ValidationError: If validation fails
        """
        # Check minimum size requirements
        if data.shape[0] < self.min_rows:
            raise ValidationError(
                f"Dataset has only {data.shape[0]} rows. Minimum required: {self.min_rows}"
            )
        
        if data.shape[1] < self.min_columns:
            raise ValidationError(
                f"Dataset has only {data.shape[1]} columns. Minimum required: {self.min_columns}"
            )
        
        # Validate target column if specified
        if target_variable:
            if target_variable not in data.columns:
                available_columns = list(data.columns)
                raise ValidationError(
                    f"Target column '{target_variable}' not found. "
                    f"Available columns: {available_columns}"
                )
            
            # Check if target column is all null
            if data[target_variable].isnull().all():
                raise ValidationError(f"Target column '{target_variable}' contains only null values")
            
            # Check if target has sufficient non-null values
            null_percentage = (data[target_variable].isnull().sum() / len(data)) * 100
            if null_percentage > self.max_null_percentage:
                raise ValidationError(
                    f"Target column '{target_variable}' has {null_percentage:.1f}% null values. "
                    f"Maximum allowed: {self.max_null_percentage}%"
                )
        
        # Check for completely empty columns
        empty_columns = data.columns[data.isnull().all()].tolist()
        if empty_columns:
            self.logger.warning(f"Found completely empty columns: {empty_columns}")
        
        # Check overall null percentage
        overall_null_percentage = (data.isnull().sum().sum() / data.size) * 100
        if overall_null_percentage > 80:
            self.logger.warning(f"Dataset has high null percentage: {overall_null_percentage:.1f}%")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of the file for tracking.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _extract_metadata(
        self,
        data: pd.DataFrame,
        file_path: str,
        file_hash: str,
        target_variable: Optional[str],
    ) -> DataIngestionResult:
        """Extract comprehensive metadata from the loaded data.
        
        Args:
            data: Loaded DataFrame
            file_path: Original file path
            file_hash: File hash
            target_variable: Target column name
            
        Returns:
            DataIngestionResult with extracted metadata
        """
        # Calculate memory usage
        memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Get null counts
        null_counts = data.isnull().sum().to_dict()
        
        # Convert dtypes to strings for serialization
        dtypes = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        return DataIngestionResult(
            file_path=file_path,
            file_hash=file_hash,
            shape=data.shape,
            columns=list(data.columns),
            dtypes=dtypes,
            target_column=target_variable,
            null_counts=null_counts,
            memory_usage_mb=memory_usage_mb,
        )
    
    def _store_ingestion_memory(
        self,
        ingestion_result: DataIngestionResult,
        context: AgentContext,
    ) -> None:
        """Store ingestion information in memory for future reference.
        
        Args:
            ingestion_result: Ingestion result to store
            context: Current pipeline context
        """
        try:
            # Store in symbolic store
            memory_client.store_symbolic(
                table_name="data_ingestion_log",
                data=ingestion_result.dict(),
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
            # Create semantic description for vector store
            description = f"""
            Data ingestion completed:
            - File: {Path(ingestion_result.file_path).name}
            - Shape: {ingestion_result.shape[0]} rows, {ingestion_result.shape[1]} columns
            - Target: {ingestion_result.target_column or 'Not specified'}
            - Memory usage: {ingestion_result.memory_usage_mb:.1f} MB
            - Null values: {sum(ingestion_result.null_counts.values())} total
            - Data types: {', '.join(set(ingestion_result.dtypes.values()))}
            """
            
            tags = ["data_ingestion", "dataset_loaded"]
            if ingestion_result.target_column:
                tags.append("supervised_learning")
            
            memory_client.store_vector(
                content=description.strip(),
                tags=tags,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
                metadata={"file_hash": ingestion_result.file_hash},
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store ingestion memory: {e}")
    
    def _find_similar_datasets(self, ingestion_result: DataIngestionResult) -> List[Any]:
        """Find similar datasets in memory based on characteristics.
        
        Args:
            ingestion_result: Current ingestion result
            
        Returns:
            List of similar dataset memories
        """
        try:
            query = f"""
            Dataset with {ingestion_result.shape[0]} rows and {ingestion_result.shape[1]} columns,
            target column type similar to {ingestion_result.target_column},
            data types: {', '.join(set(ingestion_result.dtypes.values()))}
            """
            
            similar_datasets = memory_client.search_similar(
                query=query,
                top_k=3,
                tags=["data_ingestion"],
            )
            
            if similar_datasets:
                self.logger.info(f"Found {len(similar_datasets)} similar datasets")
            
            return similar_datasets
            
        except Exception as e:
            self.logger.warning(f"Failed to search for similar datasets: {e}")
            return []
    
    def _detect_data_quality_issues(
        self,
        data: pd.DataFrame,
        target_variable: Optional[str],
    ) -> List[str]:
        """Detect potential data quality issues.
        
        Args:
            data: Loaded DataFrame
            target_variable: Target column name
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Check for high null percentage in any column
        null_percentages = (data.isnull().sum() / len(data)) * 100
        high_null_cols = null_percentages[null_percentages > 50].index.tolist()
        
        if high_null_cols:
            warnings.append(
                f"Columns with >50% null values: {', '.join(high_null_cols)}"
            )
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(data)) * 100
            warnings.append(
                f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)"
            )
        
        # Check for constant columns
        constant_cols = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            warnings.append(f"Constant columns (may need removal): {', '.join(constant_cols)}")
        
        # Check target variable distribution if specified
        if target_variable and target_variable in data.columns:
            target_null_pct = (data[target_variable].isnull().sum() / len(data)) * 100
            if target_null_pct > 10:
                warnings.append(
                    f"Target column has {target_null_pct:.1f}% null values"
                )
            
            # Check for class imbalance (if categorical target)
            if data[target_variable].dtype == 'object' or data[target_variable].nunique() < 20:
                value_counts = data[target_variable].value_counts()
                if len(value_counts) > 1:
                    imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[1]
                    if imbalance_ratio > 10:
                        warnings.append(
                            f"Severe class imbalance detected in target (ratio: {imbalance_ratio:.1f}:1)"
                        )
        
        # Check for very large file size
        if data.memory_usage(deep=True).sum() > 1024**3:  # 1GB
            warnings.append("Large dataset detected - consider sampling for initial exploration")
        
        return warnings
    
    def _apply_overrides(
        self,
        context: AgentContext,
        overrides: Dict[str, Any],
    ) -> AgentContext:
        """Apply user overrides to the ingestion process.
        
        Args:
            context: Current context
            overrides: User overrides to apply
            
        Returns:
            Updated context
        """
        if "target_variable" in overrides:
            new_target = overrides["target_variable"]
            if context.data is not None and new_target in context.data.columns:
                context.target = new_target
                context.metadata["target_variable"] = new_target
                self.logger.info(f"Updated target variable to: {new_target}")
            else:
                self.logger.warning(f"Invalid target variable override: {new_target}")
        
        if "exclude_columns" in overrides:
            exclude_cols = overrides["exclude_columns"]
            if context.data is not None:
                available_cols = [col for col in exclude_cols if col in context.data.columns]
                if available_cols:
                    context.data = context.data.drop(columns=available_cols)
                    self.logger.info(f"Excluded columns: {available_cols}")
        
        return context