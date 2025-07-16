"""Data Profiling Agent - Generates statistical summary and data insights."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from agents.base import AgentContext, AgentResult, LLMCapableAgent, UserFeedback
from memory.memory_client import memory_client
from utils.exceptions import DataProfilingError, ValidationError


class ProfileSummary(BaseModel):
    """Summary of data profiling results."""
    
    dataset_shape: tuple
    column_count: int
    row_count: int
    memory_usage_mb: float
    overall_null_percentage: float
    duplicate_rows: int
    
    # Column-level statistics
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    boolean_columns: List[str]
    
    # Statistical summaries
    numeric_stats: Dict[str, Dict[str, float]]
    categorical_stats: Dict[str, Dict[str, Any]]
    null_statistics: Dict[str, float]
    
    # Data quality alerts
    high_null_columns: List[str]
    constant_columns: List[str]
    high_cardinality_columns: List[str]
    potential_identifiers: List[str]
    
    # Feature insights
    correlated_features: List[tuple]
    target_correlations: Optional[Dict[str, float]]
    feature_importances: Optional[Dict[str, float]]


class DataProfilingAgent(LLMCapableAgent):
    """Agent that generates comprehensive data profiling and statistical analysis."""
    
    def __init__(self, use_llm: bool = True):
        """Initialize the data profiling agent.
        
        Args:
            use_llm: Whether to use LLM for generating insights
        """
        super().__init__("DataProfilingAgent", use_llm)
        
        # Profiling thresholds
        self.high_null_threshold = 50.0  # Percentage
        self.high_cardinality_threshold = 100  # Unique values
        self.correlation_threshold = 0.8  # Correlation coefficient
        self.identifier_uniqueness_threshold = 0.95  # Uniqueness ratio
    
    def run(
        self,
        context: AgentContext,
        feedback: Optional[UserFeedback] = None,
    ) -> AgentResult:
        """Execute data profiling process.
        
        Args:
            context: Pipeline context containing the data
            feedback: Optional user feedback for refinement
            
        Returns:
            AgentResult with profiling summary and insights
        """
        try:
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            # Handle user feedback
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            data = context.data
            target = context.target
            
            self.logger.info(f"Starting data profiling for dataset with shape: {data.shape}")
            
            # Generate comprehensive profile
            profile_summary = self._generate_profile_summary(data, target)
            
            # Generate LLM insights if enabled
            llm_insights = ""
            if self.use_llm:
                llm_insights = self._generate_data_insights(profile_summary, context)
            
            # Update context
            context.profile_summary = profile_summary.dict()
            context.llm_insights[self.name] = llm_insights
            
            # Store profiling results in memory
            self._store_profiling_memory(profile_summary, context)
            
            # Generate suggestions based on findings
            suggestions = self._generate_suggestions(profile_summary)
            
            # Check for similar dataset patterns
            similar_patterns = self._find_similar_patterns(profile_summary)
            if similar_patterns:
                suggestions.extend([
                    f"Found {len(similar_patterns)} datasets with similar patterns. "
                    f"Consider reviewing past successful approaches."
                ])
            
            result = AgentResult(
                success=True,
                agent_name=self.name,
                context=context,
                logs=[
                    f"Profiled dataset with {profile_summary.row_count} rows and {profile_summary.column_count} columns",
                    f"Found {len(profile_summary.numeric_columns)} numeric and {len(profile_summary.categorical_columns)} categorical columns",
                    f"Overall null percentage: {profile_summary.overall_null_percentage:.1f}%",
                    f"Detected {len(profile_summary.high_null_columns)} high-null columns",
                ],
                metrics={
                    "row_count": profile_summary.row_count,
                    "column_count": profile_summary.column_count,
                    "null_percentage": profile_summary.overall_null_percentage,
                    "duplicate_rows": profile_summary.duplicate_rows,
                    "numeric_columns": len(profile_summary.numeric_columns),
                    "categorical_columns": len(profile_summary.categorical_columns),
                },
                artifacts={
                    "profile_summary": profile_summary.dict(),
                    "llm_insights": llm_insights,
                },
                suggestions=suggestions,
                requires_approval=len(profile_summary.high_null_columns) > 5 or profile_summary.overall_null_percentage > 30,
            )
            
            # Add warnings for data quality issues
            warnings = self._generate_warnings(profile_summary)
            if warnings:
                result.warnings = warnings
                result.requires_approval = True
                result.user_message = (
                    f"Data profiling completed with {len(warnings)} quality concerns. "
                    f"Please review the findings before proceeding."
                )
            
            self._log_execution(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Data profiling failed: {e}")
            
            result = AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=str(e),
                logs=[f"Data profiling failed: {e}"],
                user_message="Data profiling encountered an error. Please review the data and try again.",
            )
            
            self._log_execution(result)
            return result
    
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate that the context contains required data.
        
        Args:
            context: Pipeline context to validate
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValidationError: If validation fails
        """
        if context.data is None:
            raise ValidationError("No data found in context")
        
        if not isinstance(context.data, pd.DataFrame):
            raise ValidationError("Data must be a pandas DataFrame")
        
        if context.data.empty:
            raise ValidationError("DataFrame is empty")
        
        return True
    
    def _generate_profile_summary(self, data: pd.DataFrame, target: Optional[str]) -> ProfileSummary:
        """Generate comprehensive profiling summary.
        
        Args:
            data: DataFrame to profile
            target: Target column name
            
        Returns:
            ProfileSummary with all profiling results
        """
        # Basic dataset information
        dataset_shape = data.shape
        row_count, column_count = dataset_shape
        memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        overall_null_percentage = (data.isnull().sum().sum() / data.size) * 100
        duplicate_rows = data.duplicated().sum()
        
        # Categorize columns by data type
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
        boolean_columns = data.select_dtypes(include=['bool']).columns.tolist()
        
        # Generate statistical summaries
        numeric_stats = self._generate_numeric_stats(data, numeric_columns)
        categorical_stats = self._generate_categorical_stats(data, categorical_columns)
        null_statistics = (data.isnull().sum() / len(data) * 100).to_dict()
        
        # Detect data quality issues
        high_null_columns = [
            col for col, null_pct in null_statistics.items() 
            if null_pct > self.high_null_threshold
        ]
        
        constant_columns = [
            col for col in data.columns 
            if data[col].nunique() <= 1
        ]
        
        high_cardinality_columns = [
            col for col in categorical_columns 
            if data[col].nunique() > self.high_cardinality_threshold
        ]
        
        potential_identifiers = [
            col for col in data.columns 
            if (data[col].nunique() / len(data)) > self.identifier_uniqueness_threshold
        ]
        
        # Feature analysis
        correlated_features = self._find_correlated_features(data, numeric_columns)
        target_correlations = self._calculate_target_correlations(data, target) if target else None
        feature_importances = self._calculate_feature_importances(data, target) if target else None
        
        return ProfileSummary(
            dataset_shape=dataset_shape,
            column_count=column_count,
            row_count=row_count,
            memory_usage_mb=memory_usage_mb,
            overall_null_percentage=overall_null_percentage,
            duplicate_rows=duplicate_rows,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            boolean_columns=boolean_columns,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            null_statistics=null_statistics,
            high_null_columns=high_null_columns,
            constant_columns=constant_columns,
            high_cardinality_columns=high_cardinality_columns,
            potential_identifiers=potential_identifiers,
            correlated_features=correlated_features,
            target_correlations=target_correlations,
            feature_importances=feature_importances,
        )
    
    def _generate_numeric_stats(self, data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Dict[str, float]]:
        """Generate statistics for numeric columns.
        
        Args:
            data: DataFrame
            numeric_columns: List of numeric column names
            
        Returns:
            Dictionary of statistics for each numeric column
        """
        numeric_stats = {}
        
        for col in numeric_columns:
            series = data[col].dropna()
            
            if len(series) == 0:
                numeric_stats[col] = {"count": 0}
                continue
            
            # Calculate statistics
            stats = {
                "count": len(series),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "zeros": int((series == 0).sum()),
                "outliers": self._count_outliers(series),
            }
            
            numeric_stats[col] = stats
        
        return numeric_stats
    
    def _generate_categorical_stats(self, data: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate statistics for categorical columns.
        
        Args:
            data: DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            Dictionary of statistics for each categorical column
        """
        categorical_stats = {}
        
        for col in categorical_columns:
            series = data[col].dropna()
            
            if len(series) == 0:
                categorical_stats[col] = {"count": 0}
                continue
            
            value_counts = series.value_counts()
            
            stats = {
                "count": len(series),
                "unique": int(series.nunique()),
                "top": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "mode_percentage": float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0,
                "cardinality": int(series.nunique()),
                "entropy": self._calculate_entropy(value_counts),
            }
            
            # Add top values for inspection
            top_values = value_counts.head(10).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}
            
            categorical_stats[col] = stats
        
        return categorical_stats
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method.
        
        Args:
            series: Numeric series
            
        Returns:
            Number of outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        return int(outliers)
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy for categorical distribution.
        
        Args:
            value_counts: Value counts series
            
        Returns:
            Entropy value
        """
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
    
    def _find_correlated_features(self, data: pd.DataFrame, numeric_columns: List[str]) -> List[tuple]:
        """Find highly correlated feature pairs.
        
        Args:
            data: DataFrame
            numeric_columns: List of numeric columns
            
        Returns:
            List of correlated feature pairs with correlation coefficient
        """
        if len(numeric_columns) < 2:
            return []
        
        corr_matrix = data[numeric_columns].corr()
        correlated_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) > self.correlation_threshold:
                    correlated_pairs.append((col1, col2, float(correlation)))
        
        return correlated_pairs
    
    def _calculate_target_correlations(self, data: pd.DataFrame, target: str) -> Optional[Dict[str, float]]:
        """Calculate correlations with target variable.
        
        Args:
            data: DataFrame
            target: Target column name
            
        Returns:
            Dictionary of correlations with target
        """
        if target not in data.columns:
            return None
        
        target_correlations = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            if col != target:
                try:
                    correlation = data[col].corr(data[target])
                    if not pd.isna(correlation):
                        target_correlations[col] = float(correlation)
                except:
                    continue
        
        return target_correlations
    
    def _calculate_feature_importances(self, data: pd.DataFrame, target: str) -> Optional[Dict[str, float]]:
        """Calculate basic feature importances using correlation and mutual information.
        
        Args:
            data: DataFrame
            target: Target column name
            
        Returns:
            Dictionary of feature importance scores
        """
        if target not in data.columns:
            return None
        
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare features
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_features:
                numeric_features.remove(target)
            
            if len(numeric_features) == 0:
                return None
            
            X = data[numeric_features].fillna(0)
            y = data[target].fillna(0)
            
            # Determine if regression or classification task
            if data[target].dtype == 'object' or data[target].nunique() < 20:
                # Classification
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
            else:
                # Regression
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            importances = dict(zip(numeric_features, mi_scores.astype(float)))
            return importances
            
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importances: {e}")
            return None
    
    def _generate_suggestions(self, profile_summary: ProfileSummary) -> List[str]:
        """Generate actionable suggestions based on profiling results.
        
        Args:
            profile_summary: Profiling summary
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # High null columns
        if profile_summary.high_null_columns:
            suggestions.append(
                f"Consider removing or imputing {len(profile_summary.high_null_columns)} columns with >50% null values: "
                f"{', '.join(profile_summary.high_null_columns[:3])}{'...' if len(profile_summary.high_null_columns) > 3 else ''}"
            )
        
        # Constant columns
        if profile_summary.constant_columns:
            suggestions.append(
                f"Remove {len(profile_summary.constant_columns)} constant columns: "
                f"{', '.join(profile_summary.constant_columns[:3])}"
            )
        
        # High cardinality columns
        if profile_summary.high_cardinality_columns:
            suggestions.append(
                f"High cardinality categorical columns may need encoding: "
                f"{', '.join(profile_summary.high_cardinality_columns[:3])}"
            )
        
        # Potential identifiers
        if profile_summary.potential_identifiers:
            suggestions.append(
                f"Potential identifier columns (consider removing): "
                f"{', '.join(profile_summary.potential_identifiers[:3])}"
            )
        
        # Correlated features
        if profile_summary.correlated_features:
            suggestions.append(
                f"Found {len(profile_summary.correlated_features)} highly correlated feature pairs - "
                f"consider removing redundant features"
            )
        
        # Duplicate rows
        if profile_summary.duplicate_rows > 0:
            suggestions.append(
                f"Found {profile_summary.duplicate_rows} duplicate rows - consider deduplication"
            )
        
        return suggestions
    
    def _generate_warnings(self, profile_summary: ProfileSummary) -> List[str]:
        """Generate warnings for data quality issues.
        
        Args:
            profile_summary: Profiling summary
            
        Returns:
            List of warning strings
        """
        warnings = []
        
        if profile_summary.overall_null_percentage > 30:
            warnings.append(f"High overall null percentage: {profile_summary.overall_null_percentage:.1f}%")
        
        if len(profile_summary.high_null_columns) > 10:
            warnings.append(f"Many columns ({len(profile_summary.high_null_columns)}) have >50% null values")
        
        if profile_summary.duplicate_rows > profile_summary.row_count * 0.1:
            warnings.append(f"High number of duplicate rows: {profile_summary.duplicate_rows}")
        
        if len(profile_summary.constant_columns) > 5:
            warnings.append(f"Many constant columns detected: {len(profile_summary.constant_columns)}")
        
        return warnings
    
    def _store_profiling_memory(self, profile_summary: ProfileSummary, context: AgentContext) -> None:
        """Store profiling results in memory.
        
        Args:
            profile_summary: Profiling summary to store
            context: Current pipeline context
        """
        try:
            # Store dataset profile using memory client convenience method
            dataset_metadata = {
                "shape": profile_summary.dataset_shape,
                "columns": profile_summary.numeric_columns + profile_summary.categorical_columns,
                "target_column": context.target,
                "data_types": {
                    "numeric": len(profile_summary.numeric_columns),
                    "categorical": len(profile_summary.categorical_columns),
                    "datetime": len(profile_summary.datetime_columns),
                    "boolean": len(profile_summary.boolean_columns),
                },
                "missing_percentage": profile_summary.overall_null_percentage,
                "duplicate_rows": profile_summary.duplicate_rows,
                "memory_usage_mb": profile_summary.memory_usage_mb,
                "quality_issues": {
                    "high_null_columns": len(profile_summary.high_null_columns),
                    "constant_columns": len(profile_summary.constant_columns),
                    "high_cardinality_columns": len(profile_summary.high_cardinality_columns),
                },
            }
            
            memory_client.store_dataset_profile(
                dataset_metadata=dataset_metadata,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store profiling memory: {e}")
    
    def _find_similar_patterns(self, profile_summary: ProfileSummary) -> List[Any]:
        """Find datasets with similar profiling patterns.
        
        Args:
            profile_summary: Current profiling summary
            
        Returns:
            List of similar dataset memories
        """
        try:
            similar_datasets = memory_client.get_similar_datasets(
                current_profile={
                    "shape": profile_summary.dataset_shape,
                    "numeric_columns": len(profile_summary.numeric_columns),
                    "categorical_columns": len(profile_summary.categorical_columns),
                    "missing_percentage": profile_summary.overall_null_percentage,
                },
                top_k=3,
            )
            
            return similar_datasets
            
        except Exception as e:
            self.logger.warning(f"Failed to find similar patterns: {e}")
            return []
    
    def _get_llm_prompt(self, context: AgentContext, feedback: Optional[UserFeedback] = None) -> str:
        """Generate LLM prompt for data profiling insights.
        
        Args:
            context: Current pipeline context
            feedback: Optional user feedback
            
        Returns:
            Formatted prompt string
        """
        profile_summary = context.profile_summary
        target = context.target
        
        prompt = f"""
        You are a data science expert analyzing a dataset profile. Please provide insights and recommendations.

        Dataset Overview:
        - Shape: {profile_summary['dataset_shape']}
        - Target variable: {target or 'Not specified'}
        - Overall null percentage: {profile_summary['overall_null_percentage']:.1f}%
        - Duplicate rows: {profile_summary['duplicate_rows']}

        Column Types:
        - Numeric: {len(profile_summary['numeric_columns'])} columns
        - Categorical: {len(profile_summary['categorical_columns'])} columns
        - DateTime: {len(profile_summary['datetime_columns'])} columns

        Data Quality Issues:
        - High null columns: {len(profile_summary['high_null_columns'])}
        - Constant columns: {len(profile_summary['constant_columns'])}
        - High cardinality columns: {len(profile_summary['high_cardinality_columns'])}
        - Potential identifiers: {len(profile_summary['potential_identifiers'])}

        Please provide:
        1. Key insights about the data quality and structure
        2. Potential challenges for machine learning
        3. Recommendations for data cleaning and preparation
        4. Feature engineering opportunities
        5. Any red flags or concerns

        Be concise and actionable in your response.
        """
        
        return prompt.strip()
    
    def _generate_data_insights(self, profile_summary: ProfileSummary, context: AgentContext) -> str:
        """Generate LLM-based insights about the data.
        
        Args:
            profile_summary: Data profiling summary
            context: Pipeline context
            
        Returns:
            Generated insights text
        """
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "LLM service not available for insights generation"
            
            data_summary = {
                "shape": profile_summary.dataset_shape,
                "null_percentage": profile_summary.overall_null_percentage,
                "column_types": {
                    "numeric": len(profile_summary.numeric_columns),
                    "categorical": len(profile_summary.categorical_columns),
                    "datetime": len(profile_summary.datetime_columns),
                },
                "quality_issues": {
                    "high_null_columns": len(profile_summary.high_null_columns),
                    "constant_columns": len(profile_summary.constant_columns),
                    "high_cardinality": len(profile_summary.high_cardinality_columns),
                },
                "target": context.target,
            }
            
            insights = llm_service.generate_data_insights(
                data_summary=data_summary,
                context=f"Data profiling for {context.target or 'unsupervised'} task",
                max_tokens=800,
                temperature=0.7,
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate data insights: {e}")
            return f"Could not generate insights: {e}"