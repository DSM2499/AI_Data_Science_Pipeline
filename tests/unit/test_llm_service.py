"""Tests for LLM service."""

import pytest
from unittest.mock import Mock, patch

from utils.llm_service import LLMMessage, LLMResponse, LLMService
from utils.exceptions import APIError


class TestLLMService:
    """Test cases for LLM service."""
    
    def test_initialization_without_api_key(self):
        """Test LLM service initialization without API key."""
        with patch('utils.llm_service.settings') as mock_settings:
            mock_settings.openai_api_key = None
            
            service = LLMService()
            assert not service.is_available()
    
    def test_initialization_with_api_key(self):
        """Test LLM service initialization with API key."""
        with patch('utils.llm_service.settings') as mock_settings, \
             patch('utils.llm_service.OpenAI') as mock_openai:
            
            mock_settings.openai_api_key = "test_key"
            mock_openai.return_value = Mock()
            
            service = LLMService()
            assert service.is_available()
    
    def test_llm_message_model(self):
        """Test LLMMessage pydantic model."""
        message = LLMMessage(role="user", content="Test message")
        assert message.role == "user"
        assert message.content == "Test message"
        assert message.name is None
    
    def test_llm_response_model(self):
        """Test LLMResponse pydantic model."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            tokens_used={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
            response_time=1.5,
        )
        
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.tokens_used["total_tokens"] == 30
        assert response.response_time == 1.5
    
    @patch('utils.llm_service.settings')
    def test_generate_simple_response_no_service(self, mock_settings):
        """Test simple response generation when service is unavailable."""
        mock_settings.openai_api_key = None
        
        service = LLMService()
        
        with pytest.raises(APIError):
            service.generate_simple_response("Test prompt")
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        with patch('utils.llm_service.settings') as mock_settings, \
             patch('utils.llm_service.OpenAI'):
            
            mock_settings.openai_api_key = "test_key"
            service = LLMService()
            service.min_request_interval = 0.1
            
            import time
            start_time = time.time()
            service._rate_limit()
            service._rate_limit()
            end_time = time.time()
            
            # Should have added some delay
            assert end_time - start_time >= 0.1
    
    def test_get_usage_stats(self):
        """Test usage statistics retrieval."""
        with patch('utils.llm_service.settings') as mock_settings:
            mock_settings.openai_api_key = None
            mock_settings.openai_model = "gpt-4"
            
            service = LLMService()
            stats = service.get_usage_stats()
            
            assert isinstance(stats, dict)
            assert "service_available" in stats
            assert "configured_model" in stats
            assert stats["configured_model"] == "gpt-4"