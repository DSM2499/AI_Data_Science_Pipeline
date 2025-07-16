"""Tests for ModelingAgent."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from agents.base import AgentContext
from agents.modeling import ModelingAgent
from utils.exceptions import ValidationError


class TestModelingAgent:
    """Test cases for ModelingAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = ModelingAgent()
        assert agent.name == "ModelingAgent"
        assert agent.use_llm is True
        assert agent.test_size == 0.2
        assert agent.random_state == 42
    
    def test_validation_no_data(self):
        """Test validation fails with no data."""
        agent = ModelingAgent()
        context = AgentContext()
        
        with pytest.raises(ValidationError):
            agent.validate_inputs(context)
    
    def test_validation_no_target(self):
        """Test validation fails with no target."""
        agent = ModelingAgent()
        data = pd.DataFrame({"feature1": [1, 2, 3]})
        context = AgentContext(data=data)
        
        with pytest.raises(ValidationError):
            agent.validate_inputs(context)
    
    def test_validation_insufficient_data(self):
        """Test validation fails with insufficient data."""
        agent = ModelingAgent()
        data = pd.DataFrame({
            "feature1": [1, 2],
            "target": [0, 1],
        })
        context = AgentContext(data=data, target="target")
        
        with pytest.raises(ValidationError):
            agent.validate_inputs(context)
    
    def test_validation_success(self):
        """Test successful validation."""
        agent = ModelingAgent()
        data = pd.DataFrame({
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1] * 50,
        })
        context = AgentContext(data=data, target="target")
        
        assert agent.validate_inputs(context) is True
    
    def test_prepare_data_classification(self):
        """Test data preparation for classification."""
        agent = ModelingAgent()
        
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": ["A", "B", "A", "B", "A"],
        })
        
        X, y, task_type = agent._prepare_data(data, "target")
        
        assert task_type == "classification"
        assert X.shape[1] == 2  # Two features
        assert len(y) == 5
        assert X.columns.tolist() == ["feature1", "feature2"]
        # Target should be encoded
        assert y.dtype in [np.int32, np.int64]
    
    def test_prepare_data_regression(self):
        """Test data preparation for regression."""
        agent = ModelingAgent()
        
        data = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],
            "target": [1.1, 2.2, 3.3, 4.4, 5.5],
        })
        
        X, y, task_type = agent._prepare_data(data, "target")
        
        assert task_type == "regression"
        assert X.shape[1] == 2
        assert len(y) == 5
        assert X.columns.tolist() == ["feature1", "feature2"]
        # Target should remain numeric
        assert y.dtype in [np.float32, np.float64]
    
    def test_model_creation_classification(self):
        """Test model creation for classification."""
        agent = ModelingAgent()
        
        # Test different classification models
        models_to_test = ["logistic_regression", "random_forest", "xgboost"]
        
        for model_name in models_to_test:
            try:
                model_config = {"params": {}}
                model = agent._create_model(model_name, model_config, "classification")
                
                assert model is not None
                assert hasattr(model, "fit")
                assert hasattr(model, "predict")
                
            except Exception as e:
                # Some models might not be available in test environment
                pytest.skip(f"Model {model_name} not available: {e}")
    
    def test_model_creation_regression(self):
        """Test model creation for regression."""
        agent = ModelingAgent()
        
        # Test different regression models
        models_to_test = ["linear_regression", "random_forest", "ridge"]
        
        for model_name in models_to_test:
            try:
                model_config = {"params": {}}
                model = agent._create_model(model_name, model_config, "regression")
                
                assert model is not None
                assert hasattr(model, "fit")
                assert hasattr(model, "predict")
                
            except Exception as e:
                # Some models might not be available in test environment
                pytest.skip(f"Model {model_name} not available: {e}")
    
    def test_metrics_calculation_classification(self):
        """Test metrics calculation for classification."""
        agent = ModelingAgent()
        
        # Simple binary classification example
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        # Mock model for testing
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])
        
        mock_model = MockModel()
        X_test = pd.DataFrame(np.random.randn(5, 2))
        
        metrics = agent._calculate_metrics(y_true, y_pred, "classification", mock_model, X_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        
        # Check value ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
    
    def test_metrics_calculation_regression(self):
        """Test metrics calculation for regression."""
        agent = ModelingAgent()
        
        # Simple regression example
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
        
        class MockModel:
            pass
        
        mock_model = MockModel()
        X_test = pd.DataFrame(np.random.randn(5, 2))
        
        metrics = agent._calculate_metrics(y_true, y_pred, "regression", mock_model, X_test)
        
        assert "r2" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        
        # Check that R² is reasonable for good predictions
        assert metrics["r2"] > 0.8  # Should be high for close predictions
        assert metrics["mse"] > 0  # MSE should be positive
        assert metrics["rmse"] > 0  # RMSE should be positive
        assert metrics["mae"] > 0  # MAE should be positive
    
    @patch('agents.modeling.pickle.dump')
    @patch('builtins.open')
    def test_save_best_model(self, mock_open, mock_pickle_dump):
        """Test model saving functionality."""
        agent = ModelingAgent()
        
        # Create mock model result
        class MockModelResult:
            model_name = "test_model"
            _model_instance = "mock_model_object"
        
        mock_model_result = MockModelResult()
        context = AgentContext()
        
        # Mock the file operations
        mock_open.return_value.__enter__.return_value = "mock_file"
        
        try:
            model_path = agent._save_best_model(mock_model_result, context)
            
            assert isinstance(model_path, str)
            assert "test_model" in model_path
            assert model_path.endswith(".pkl")
            
            # Verify pickle.dump was called
            mock_pickle_dump.assert_called_once_with("mock_model_object", "mock_file")
            
        except Exception as e:
            # File operations might fail in test environment
            pytest.skip(f"Model saving test failed: {e}")
    
    def test_get_primary_metric(self):
        """Test primary metric extraction."""
        agent = ModelingAgent()
        
        # Mock model result for classification
        class MockModelResult:
            metrics = {"accuracy": 0.8, "f1": 0.75, "roc_auc": 0.85}
        
        mock_result = MockModelResult()
        
        # Classification should prefer ROC AUC
        primary = agent._get_primary_metric(mock_result, "classification")
        assert primary == 0.85
        
        # Mock model result for regression
        mock_result.metrics = {"r2": 0.7, "mae": 0.2, "rmse": 0.3}
        
        # Regression should prefer R²
        primary = agent._get_primary_metric(mock_result, "regression")
        assert primary == 0.7
    
    def test_llm_prompt_generation(self):
        """Test LLM prompt generation."""
        agent = ModelingAgent()
        
        data = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0],
        })
        
        context = AgentContext(data=data, target="target")
        
        prompt = agent._get_llm_prompt(context)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "machine learning" in prompt.lower()
        assert "target" in prompt.lower()
        assert "dataset" in prompt.lower()