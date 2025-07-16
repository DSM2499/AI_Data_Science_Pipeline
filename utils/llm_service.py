"""LLM service layer for interacting with OpenAI and other language models."""

import time
from typing import Any, Dict, List, Optional, Union

import openai
from openai import OpenAI
from pydantic import BaseModel

from config import settings
from utils.exceptions import APIError
from utils.logging import get_agent_logger


class LLMMessage(BaseModel):
    """Represents a message in LLM conversation."""
    
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


class LLMResponse(BaseModel):
    """Represents an LLM response."""
    
    content: str
    model: str
    tokens_used: Dict[str, int]
    finish_reason: str
    response_time: float


class LLMService:
    """Service for interacting with language models."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.logger = get_agent_logger("LLMService")
        self.client = None
        
        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Initialize OpenAI client if API key is available
        if settings.openai_api_key:
            try:
                self.client = OpenAI(api_key=settings.openai_api_key, base_url="https://api.deepseek.com")
                self.logger.info("OpenAI client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
                raise APIError(f"OpenAI initialization failed: {e}")
        else:
            self.logger.warning("No OpenAI API key provided - LLM features will be disabled")
    
    def is_available(self) -> bool:
        """Check if LLM service is available.
        
        Returns:
            True if LLM service is ready to use
        """
        return self.client is not None
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_response(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        timeout: int = 60,
    ) -> LLMResponse:
        """Generate a response using the language model.
        
        Args:
            messages: List of conversation messages
            model: Model to use (defaults to configured model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            timeout: Request timeout in seconds
            
        Returns:
            LLMResponse with generated content and metadata
            
        Raises:
            APIError: If the API call fails
        """
        if not self.is_available():
            raise APIError("LLM service is not available")
        
        if model is None:
            model = settings.openai_model
        
        # Apply rate limiting
        self._rate_limit()
        
        try:
            start_time = time.time()
            
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            
            response_time = time.time() - start_time
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            
            # Extract token usage
            usage = response.usage
            tokens_used = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            
            llm_response = LLMResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                response_time=response_time,
            )
            
            self.logger.info(
                f"LLM response generated successfully",
                model=model,
                tokens=tokens_used["total_tokens"],
                response_time=response_time,
            )
            
            return llm_response
            
        except openai.APITimeoutError:
            self.logger.error("OpenAI API timeout")
            raise APIError("LLM request timed out")
        
        except openai.RateLimitError:
            self.logger.error("OpenAI rate limit exceeded")
            raise APIError("LLM rate limit exceeded - please try again later")
        
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise APIError(f"LLM API error: {e}")
        
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM call: {e}")
            raise APIError(f"LLM service error: {e}")
    
    def generate_simple_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate a simple text response from a prompt.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            **kwargs: Additional parameters for generate_response
            
        Returns:
            Generated text content
        """
        messages = []
        
        if system_message:
            messages.append(LLMMessage(role="system", content=system_message))
        
        messages.append(LLMMessage(role="user", content=prompt))
        
        response = self.generate_response(messages, **kwargs)
        return response.content
    
    def generate_code_explanation(
        self,
        code: str,
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate an explanation for code.
        
        Args:
            code: Code to explain
            context: Optional context about the code
            **kwargs: Additional parameters
            
        Returns:
            Code explanation
        """
        system_message = (
            "You are a data science expert. Explain code clearly and concisely, "
            "focusing on what it does and why it's useful for data science workflows."
        )
        
        prompt = f"Explain this code:\n\n```python\n{code}\n```"
        if context:
            prompt += f"\n\nContext: {context}"
        
        return self.generate_simple_response(
            prompt=prompt,
            system_message=system_message,
            **kwargs
        )
    
    def generate_data_insights(
        self,
        data_summary: Dict[str, Any],
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate insights about data characteristics.
        
        Args:
            data_summary: Summary of data characteristics
            context: Optional additional context
            **kwargs: Additional parameters
            
        Returns:
            Data insights and recommendations
        """
        system_message = (
            "You are a data scientist analyzing a dataset. Provide practical insights "
            "and actionable recommendations based on the data characteristics."
        )
        
        prompt = f"Analyze this dataset summary and provide insights:\n\n{data_summary}"
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        return self.generate_simple_response(
            prompt=prompt,
            system_message=system_message,
            **kwargs
        )
    
    def generate_feature_suggestions(
        self,
        columns_info: Dict[str, Any],
        target_info: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate feature engineering suggestions.
        
        Args:
            columns_info: Information about existing columns
            target_info: Information about target variable
            task_type: Type of ML task (classification, regression, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Feature engineering suggestions
        """
        system_message = (
            "You are a feature engineering expert. Suggest practical and effective "
            "feature engineering techniques based on the data characteristics."
        )
        
        prompt = f"Suggest feature engineering techniques for this data:\n\nColumns: {columns_info}"
        
        if target_info:
            prompt += f"\nTarget variable: {target_info}"
        
        if task_type:
            prompt += f"\nTask type: {task_type}"
        
        prompt += "\n\nProvide specific, actionable feature engineering suggestions."
        
        return self.generate_simple_response(
            prompt=prompt,
            system_message=system_message,
            **kwargs
        )
    
    def generate_model_recommendations(
        self,
        dataset_characteristics: Dict[str, Any],
        task_type: str,
        performance_requirements: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Generate model selection recommendations.
        
        Args:
            dataset_characteristics: Characteristics of the dataset
            task_type: Type of ML task
            performance_requirements: Performance requirements
            **kwargs: Additional parameters
            
        Returns:
            Model recommendations and rationale
        """
        system_message = (
            "You are a machine learning expert. Recommend appropriate models based on "
            "dataset characteristics and requirements, with clear rationale."
        )
        
        prompt = f"""
        Recommend machine learning models for this scenario:
        
        Dataset characteristics: {dataset_characteristics}
        Task type: {task_type}
        """
        
        if performance_requirements:
            prompt += f"\nPerformance requirements: {performance_requirements}"
        
        prompt += "\n\nProvide model recommendations with explanations."
        
        return self.generate_simple_response(
            prompt=prompt,
            system_message=system_message,
            **kwargs
        )
    
    def interpret_model_results(
        self,
        model_info: Dict[str, Any],
        metrics: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """Generate interpretation of model results.
        
        Args:
            model_info: Information about the trained model
            metrics: Model performance metrics
            feature_importance: Feature importance scores
            **kwargs: Additional parameters
            
        Returns:
            Model results interpretation
        """
        system_message = (
            "You are a data scientist interpreting model results. Explain performance "
            "metrics and feature importance in business terms."
        )
        
        prompt = f"""
        Interpret these model results:
        
        Model: {model_info}
        Performance metrics: {metrics}
        """
        
        if feature_importance:
            prompt += f"\nFeature importance: {feature_importance}"
        
        prompt += "\n\nProvide a clear interpretation of the results and their implications."
        
        return self.generate_simple_response(
            prompt=prompt,
            system_message=system_message,
            **kwargs
        )
    
    def generate_report_section(
        self,
        section_type: str,
        data: Dict[str, Any],
        audience: str = "technical",
        **kwargs
    ) -> str:
        """Generate a section of a data science report.
        
        Args:
            section_type: Type of report section (summary, methodology, results, etc.)
            data: Data for the section
            audience: Target audience (technical, business, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Generated report section
        """
        system_message = (
            f"You are writing a {section_type} section for a data science report "
            f"targeted at a {audience} audience. Be clear, accurate, and appropriately detailed."
        )
        
        prompt = f"Write a {section_type} section based on this data:\n\n{data}"
        
        return self.generate_simple_response(
            prompt=prompt,
            system_message=system_message,
            **kwargs
        )
    
    def process_natural_language_feedback(
        self,
        feedback_text: str,
        agent_context: Dict[str, Any],
        current_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process natural language feedback and convert to structured actions.
        
        Args:
            feedback_text: User feedback in natural language
            agent_context: Context about the current agent and data
            current_result: Current agent result for context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with structured feedback and actions
        """
        system_message = (
            "You are an AI assistant that processes user feedback for data science pipelines. "
            "Convert natural language feedback into structured actions and code. "
            "Respond with a JSON object containing the following fields:\n"
            "- 'intent': The user's main intention (modify, skip, add, rollback, etc.)\n"
            "- 'actions': List of specific actions to take\n"
            "- 'code': Python code to implement the feedback (if applicable)\n"
            "- 'parameters': Any parameter adjustments needed\n"
            "- 'skip_operations': List of operations to skip\n"
            "- 'explanation': Brief explanation of the interpretation\n"
            "Focus on practical, implementable solutions."
        )
        
        agent_name = agent_context.get('agent_name', 'Unknown')
        data_info = agent_context.get('data_info', {})
        available_operations = agent_context.get('available_operations', [])
        
        prompt = f"""
        Process this user feedback for a {agent_name}:
        
        User Feedback: "{feedback_text}"
        
        Context:
        - Agent: {agent_name}
        - Data shape: {data_info.get('shape', 'Unknown')}
        - Available columns: {data_info.get('columns', [])[:10]}...
        - Available operations: {available_operations[:5]}...
        
        Current Result Summary: {current_result or 'No current result'}
        
        Convert this feedback into structured actions that can be executed by the system.
        If the user wants custom code, generate the appropriate Python code.
        If they want to skip operations, identify which ones.
        If they want to modify parameters, specify which parameters and values.
        
        Return a valid JSON object with the fields specified above.
        """
        
        try:
            response = self.generate_simple_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3,  # Lower temperature for more consistent JSON
                **kwargs
            )
            
            # Try to parse as JSON, return structured result
            import json
            try:
                parsed_response = json.loads(response)
                return parsed_response
            except json.JSONDecodeError:
                # If JSON parsing fails, extract key information
                return {
                    "intent": "modify",
                    "actions": ["process_natural_language"],
                    "code": self._extract_code_from_response(response),
                    "parameters": {},
                    "skip_operations": [],
                    "explanation": response[:200] + "..." if len(response) > 200 else response
                }
                
        except Exception as e:
            self.logger.error(f"Failed to process natural language feedback: {e}")
            return {
                "intent": "unknown",
                "actions": [],
                "code": "",
                "parameters": {},
                "skip_operations": [],
                "explanation": f"Error processing feedback: {e}"
            }
    
    def generate_code_from_description(
        self,
        description: str,
        context: Dict[str, Any],
        code_type: str = "feature_engineering",
        **kwargs
    ) -> str:
        """Generate Python code from natural language description.
        
        Args:
            description: Natural language description of what to do
            context: Context about data and available variables
            code_type: Type of code to generate (feature_engineering, cleaning, etc.)
            **kwargs: Additional parameters
            
        Returns:
            Generated Python code
        """
        system_message = (
            f"You are a data science code generator. Generate clean, efficient Python code "
            f"for {code_type} based on natural language descriptions. "
            "Use pandas and numpy conventions. Include comments explaining the logic. "
            "Make sure the code is safe and handles edge cases."
        )
        
        data_info = context.get('data_info', {})
        columns = context.get('columns', [])
        target = context.get('target', None)
        
        prompt = f"""
        Generate Python code for: "{description}"
        
        Context:
        - Data shape: {data_info.get('shape', 'Unknown')}
        - Available columns: {columns}
        - Target variable: {target}
        - Code type: {code_type}
        
        Requirements:
        1. Use 'data' as the DataFrame variable name
        2. Handle missing values and edge cases
        3. Add comments explaining the logic
        4. Return clean, executable code
        5. Don't modify the original 'data' variable, create new columns
        
        Generate only the Python code, no explanations outside of comments.
        """
        
        try:
            code = self.generate_simple_response(
                prompt=prompt,
                system_message=system_message,
                temperature=0.2,  # Low temperature for consistent code
                **kwargs
            )
            
            # Clean the code response
            code = self._clean_generated_code(code)
            return code
            
        except Exception as e:
            self.logger.error(f"Failed to generate code from description: {e}")
            return f"# Error generating code: {e}\n# Description: {description}"
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        import re
        
        # Look for code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Look for code patterns
        code_lines = []
        for line in response.split('\n'):
            if (line.strip().startswith('data[') or 
                line.strip().startswith('data.') or
                'pd.' in line or 'np.' in line or
                line.strip().startswith('#')):
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else ""
    
    def _clean_generated_code(self, code: str) -> str:
        """Clean and validate generated code."""
        # Remove markdown code blocks
        import re
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        # Ensure we have actual code
        if not code or len(code.split('\n')) < 1:
            return "# No valid code generated"
        
        return code
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for the LLM service.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "service_available": self.is_available(),
            "client_type": "OpenAI" if self.client else None,
            "configured_model": settings.openai_model,
            "rate_limit_interval": self.min_request_interval,
        }


# Global LLM service instance
llm_service = LLMService()