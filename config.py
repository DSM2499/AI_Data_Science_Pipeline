"""Configuration management for Data Science Agent."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field



class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")

    # Application Configuration
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    project_root: Path = Field(Path.cwd(), env="PROJECT_ROOT")

    # Memory Configuration
    memory_vector_path: Path = Field(Path("memory/vector"), env="MEMORY_VECTOR_PATH")
    memory_symbolic_path: Path = Field(
        Path("memory/symbolic"), env="MEMORY_SYMBOLIC_PATH"
    )
    chromadb_persist_directory: Path = Field(
        Path("memory/vector/chromadb"), env="CHROMADB_PERSIST_DIRECTORY"
    )

    # Output Configuration
    output_path: Path = Field(Path("project_output"), env="OUTPUT_PATH")
    models_path: Path = Field(Path("project_output/models"), env="MODELS_PATH")
    reports_path: Path = Field(Path("project_output/reports"), env="REPORTS_PATH")
    data_path: Path = Field(Path("project_output/data"), env="DATA_PATH")

    # Streamlit Configuration
    streamlit_server_port: int = Field(8501, env="STREAMLIT_SERVER_PORT")
    streamlit_server_address: str = Field("localhost", env="STREAMLIT_SERVER_ADDRESS")

    # Model Training Configuration
    max_models_per_run: int = Field(5, env="MAX_MODELS_PER_RUN")
    max_time_per_model: int = Field(300, env="MAX_TIME_PER_MODEL")
    max_hyperparam_configs: int = Field(25, env="MAX_HYPERPARAM_CONFIGS")
    memory_limit_gb: int = Field(4, env="MEMORY_LIMIT_GB")

    # Time Series Configuration
    default_forecast_horizon: int = Field(90, env="DEFAULT_FORECAST_HORIZON")
    default_validation_window: int = Field(12, env="DEFAULT_VALIDATION_WINDOW")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize settings and create directories."""
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.memory_vector_path,
            self.memory_symbolic_path,
            self.chromadb_persist_directory,
            self.output_path,
            self.models_path,
            self.reports_path,
            self.data_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


class ModelingConfig:
    """Configuration for modeling policies and constraints."""

    # Classification models
    CLASSIFICATION_MODELS = {
        "logistic_regression": {
            "class": "sklearn.linear_model.LogisticRegression",
            "use_case": "Baseline, interpretability",
            "params": {"max_iter": 1000, "random_state": 42},
        },
        "random_forest": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "use_case": "Tabular data, non-linear, robust to noise",
            "params": {"n_estimators": 100, "random_state": 42},
        },
        "xgboost": {
            "class": "xgboost.XGBClassifier",
            "use_case": "Benchmark model for structured classification",
            "params": {"random_state": 42, "eval_metric": "logloss"},
        },
        "knn": {
            "class": "sklearn.neighbors.KNeighborsClassifier",
            "use_case": "Low dimensional, small datasets",
            "params": {"n_neighbors": 5},
        },
        "naive_bayes": {
            "class": "sklearn.naive_bayes.GaussianNB",
            "use_case": "High-cardinality categorical, text-based inputs",
            "params": {},
        },
        "svm": {
            "class": "sklearn.svm.SVC",
            "use_case": "Margin-focused, low-sample",
            "params": {"probability": True, "random_state": 42},
        },
    }

    # Regression models
    REGRESSION_MODELS = {
        "linear_regression": {
            "class": "sklearn.linear_model.LinearRegression",
            "use_case": "Baseline, explainability",
            "params": {},
        },
        "ridge": {
            "class": "sklearn.linear_model.Ridge",
            "use_case": "Multicollinearity, feature shrinkage",
            "params": {"random_state": 42},
        },
        "lasso": {
            "class": "sklearn.linear_model.Lasso",
            "use_case": "Feature selection, sparse models",
            "params": {"random_state": 42},
        },
        "random_forest": {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "use_case": "Robust to outliers, nonlinear data",
            "params": {"n_estimators": 100, "random_state": 42},
        },
        "xgboost": {
            "class": "xgboost.XGBRegressor",
            "use_case": "Best performance, can overfit",
            "params": {"random_state": 42},
        },
        "svr": {
            "class": "sklearn.svm.SVR",
            "use_case": "Precise, slower on large datasets",
            "params": {},
        },
    }

    # Time series models
    TIME_SERIES_MODELS = {
        "arima": {
            "class": "statsmodels.tsa.arima.model.ARIMA",
            "use_case": "Univariate, linear trends",
            "params": {},
        },
        "prophet": {
            "class": "prophet.Prophet",
            "use_case": "Seasonal + holiday patterns",
            "params": {},
        },
        "xgboost_ts": {
            "class": "xgboost.XGBRegressor",
            "use_case": "Multivariate or engineered features",
            "params": {"random_state": 42},
        },
    }

    # Evaluation metrics
    CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    REGRESSION_METRICS = ["r2", "mse", "mae", "rmse"]
    TIME_SERIES_METRICS = ["mae", "rmse", "mape", "smape"]

    # Hyperparameter grids (simplified for initial implementation)
    HYPERPARAMETER_GRIDS = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "xgboost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2],
        },
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"],
        },
    }


# Global modeling configuration
modeling_config = ModelingConfig()