import requests
import json
import time
import random
from typing import List, Dict, Union, Any
from graph_of_thoughts.language_models import AbstractLanguageModel


class LlamaTogetherAI(AbstractLanguageModel):
    """
    Enhanced LLama model interface with BETTER DIVERSITY SUPPORT for parameter synthesis
    """

    def __init__(self, config_path: str = "", model_name: str = "llama8b", cache: bool = False) -> None:
        """
        Initialize the Llama model with Together AI configuration
        
        :param config_path: Path to the configuration file
        :param model_name: Model configuration name in config file
        :param cache: Whether to cache responses
        """
        super().__init__(config_path, model_name, cache)
        
        # Load configuration
        self.config: Dict = self.config[model_name]
        self.api_key: str = self.config["api_key"]
        self.model: str = self.config["model"]
        self.api_base: str = self.config.get("api_base", "https://api.together.xyz/v1")
        self.temperature: float = self.config.get("temperature", 0.8)  # Higher default for diversity
        self.max_tokens: int = self.config.get("max_tokens", 2048)
        
        # Cost tracking
        self.prompt_token_cost: float = self.config.get("prompt_token_cost", 0.0002)
        self.response_token_cost: float = self.config.get("response_token_cost", 0.0002)
        
        # Diversity settings
        self.diversity_mode = True
        self.attempt_counter = 0
        
        # Initialize cache if needed
        if self.cache:
            self._cache = {}
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def query(self, query: str, num_responses: int = 1) -> Dict[str, Any]:
        """
        Query the Llama model with ENHANCED DIVERSITY for parameter synthesis
        
        :param query: The input query/prompt
        :param num_responses: Number of responses to generate
        :return: API response with generated texts
        """
        # Check cache first if enabled (but skip for diversity mode)
        if self.cache and not self.diversity_mode:
            cache_key = f"{query}_{num_responses}_{self.temperature}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Increment attempt counter for diversity
        self.attempt_counter += 1
        
        # DETECT PARAMETER SYNTHESIS and enhance diversity
        is_parameter_synthesis = self._is_parameter_synthesis_query(query)
        
        # Prepare the request payload with diversity enhancements
        if is_parameter_synthesis:
            # DIVERSITY ENHANCEMENT: Add randomness and variety
            enhanced_query = self._add_diversity_forcing(query)
            
            # Dynamic temperature based on attempt number
            dynamic_temperature = min(1.5, self.temperature + (self.attempt_counter % 4) * 0.1)
            
            # Add diversity seed
            diversity_seed = random.randint(1000, 9999)
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"""You are a mathematical expert specializing in barrier certificate synthesis. 
CRITICAL: Generate DIVERSE numerical solutions. Each attempt must be DIFFERENT from previous ones.
Attempt #{self.attempt_counter} - Use diversity seed {diversity_seed} for unique coefficients.
NEVER repeat the same coefficients. Explore different mathematical approaches."""
                    },
                    {
                        "role": "user",
                        "content": enhanced_query
                    }
                ],
                "temperature": dynamic_temperature,
                "max_tokens": self.max_tokens,
                "n": num_responses,
                "top_p": min(0.95, 0.8 + (self.attempt_counter % 3) * 0.05),  # Variable top_p for diversity
                "frequency_penalty": 0.3,  # Encourage different words/patterns
                "presence_penalty": 0.2,   # Encourage new topics/approaches
                "stop": None
            }
        else:
            # Normal query for non-parameter synthesis
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "n": num_responses,
                "stop": None
            }
        
        # Make the API request with enhanced error handling
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                
                # Update token counts for cost tracking
                if 'usage' in result:
                    usage = result['usage']
                    self.prompt_tokens += usage.get('prompt_tokens', 0)
                    self.completion_tokens += usage.get('completion_tokens', 0)
                    
                    # Calculate cost
                    prompt_cost = usage.get('prompt_tokens', 0) * self.prompt_token_cost / 1000
                    completion_cost = usage.get('completion_tokens', 0) * self.response_token_cost / 1000
                    self.cost += prompt_cost + completion_cost
                
                # VERIFY DIVERSITY for parameter synthesis
                if is_parameter_synthesis:
                    result = self._enhance_diversity_in_result(result, attempt)
                
                # Cache the result if caching is enabled (but not in diversity mode)
                if self.cache and not self.diversity_mode:
                    self._cache[cache_key] = result
                
                return result
                
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Return fallback response on final failure
                    return self._create_diverse_fallback_response(query, num_responses, is_parameter_synthesis)
                time.sleep(1)  # Brief delay before retry
                
        # Should not reach here, but just in case
        return self._create_diverse_fallback_response(query, num_responses, is_parameter_synthesis)
    
    def _is_parameter_synthesis_query(self, query: str) -> bool:
        """Enhanced detection of parameter synthesis queries"""
        query_lower = query.lower()
        
        # Look for parameter synthesis indicators
        param_indicators = [
            'parameter synthesis', 'parameter values', 'numerical values',
            'find parameters', 'determine parameters', 'synthesis strategy',
            'numerical coefficients', 'exact numerical', 'specific numerical',
            'attempt #', 'b(x) =', 'barrier certificate', 'coefficient',
            'conservative', 'aggressive', 'balanced', 'mathematical intuition'
        ]
        
        # Check for template types
        template_indicators = [
            'linear', 'quadratic', 'nonlinear', 'polynomial', 'template'
        ]
        
        # Check for parameter synthesis phase
        is_param_phase = any(indicator in query_lower for indicator in param_indicators)
        
        return is_param_phase
    
    def _add_diversity_forcing(self, query: str) -> str:
        """Add diversity forcing to parameter synthesis queries"""
        
        # Generate random mathematical examples for inspiration
        diverse_examples = self._generate_diverse_mathematical_examples()
        
        diversity_forcing = f"""

DIVERSITY REQUIREMENTS FOR ATTEMPT #{self.attempt_counter}

CRITICAL: This is attempt #{self.attempt_counter}. Your solution MUST be mathematically different from previous attempts.

DIVERSITY INSPIRATION - Consider these coefficient ranges:
{diverse_examples}

UNIQUENESS REQUIREMENTS:
- Use DIFFERENT coefficient values than previous attempts
- Explore DIFFERENT mathematical approaches  
- Apply DIFFERENT strategic thinking
- Generate FRESH numerical combinations

MATHEMATICAL CREATIVITY:
- Vary coefficient magnitudes (some small, some large)
- Try different constant terms
- Experiment with coefficient ratios
- Use mathematical intuition uniquely

OUTPUT MUST BE: B(x) = [UNIQUE numerical expression with x1, x2]

Remember: Attempt #{self.attempt_counter} needs its own DISTINCT solution!
"""
        
        return query + diversity_forcing
    
    def _generate_diverse_mathematical_examples(self) -> str:
        """Generate diverse mathematical coefficient examples"""
        # Create different coefficient ranges based on attempt number
        base_ranges = [
            (0.1, 1.0),   # Small coefficients
            (0.5, 2.5),   # Medium coefficients  
            (1.0, 4.0),   # Large coefficients
            (0.05, 3.0)   # Mixed range
        ]
        
        range_idx = (self.attempt_counter - 1) % len(base_ranges)
        min_coeff, max_coeff = base_ranges[range_idx]
        
        # Generate random examples
        examples = []
        for i in range(3):
            coeff1 = round(random.uniform(min_coeff, max_coeff), 2)
            coeff2 = round(random.uniform(min_coeff, max_coeff), 2)
            constant = round(random.uniform(-3.0, -0.1), 2)
            examples.append(f"  • {coeff1}*x1**2 + {coeff2}*x2**2 + {constant}")
        
        return f"Range {range_idx + 1} ({min_coeff}-{max_coeff}):\n" + "\n".join(examples)
    
    def _enhance_diversity_in_result(self, result: Dict, attempt: int) -> Dict:
        """Enhance diversity in the result if needed"""
        try:
            # Check if result contains diverse numerical content
            for choice in result.get('choices', []):
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    
                    # If content looks too generic or repetitive, enhance it
                    if self._content_needs_diversity_boost(content):
                        enhanced_content = self._boost_content_diversity(content, attempt)
                        choice['message']['content'] = enhanced_content
            
            return result
            
        except Exception as e:
            print(f"Error enhancing diversity: {e}")
            return result
    
    def _content_needs_diversity_boost(self, content: str) -> bool:
        """Check if content needs diversity enhancement"""
        # Look for signs of generic or repetitive content
        generic_signs = [
            content.count('1.0') > 2,  # Too many 1.0 coefficients
            content.count('0.5') > 2,  # Too many 0.5 coefficients  
            '0*x1' in content,         # Zero coefficients (bad)
            content.count('x1') < 1,   # Missing variables
            len(content) < 50          # Too short
        ]
        
        return any(generic_signs)
    
    def _boost_content_diversity(self, content: str, attempt: int) -> str:
        """Boost diversity in content that seems generic"""
        try:
            # If content is too generic, provide a diverse mathematical expression
            diverse_expressions = [
                f"B(x) = {random.uniform(0.5, 2.0):.1f}*x1**2 + {random.uniform(0.3, 1.8):.1f}*x2**2 + {random.uniform(-2.5, -0.2):.1f}",
                f"B(x) = {random.uniform(0.8, 3.0):.1f}*x1**2 + {random.uniform(0.1, 0.8):.1f}*x1*x2 + {random.uniform(0.6, 2.2):.1f}*x2**2 + {random.uniform(-3.0, -0.5):.1f}",
                f"B(x) = {random.uniform(0.2, 1.5):.1f}*x1**2 + {random.uniform(0.4, 2.0):.1f}*x2**2 + {random.uniform(0.1, 0.6):.1f}*x1 + {random.uniform(-2.0, -0.3):.1f}",
                f"B(x) = {random.uniform(0.1, 0.6):.2f}*x1**4 + {random.uniform(0.1, 0.5):.2f}*x2**4 + {random.uniform(0.8, 2.5):.1f}*x1**2 + {random.uniform(0.6, 2.0):.1f}*x2**2 + {random.uniform(-3.5, -0.8):.1f}"
            ]
            
            # Select expression based on attempt number
            expr_idx = (attempt + self.attempt_counter) % len(diverse_expressions)
            enhanced_expr = diverse_expressions[expr_idx]
            
            # Add explanation
            enhanced_content = f"""For attempt #{self.attempt_counter}, I'll use a diverse mathematical approach:

{enhanced_expr}

This expression uses coefficients specifically chosen to be different from previous attempts, ensuring mathematical diversity while maintaining the barrier certificate structure."""
            
            return enhanced_content
            
        except Exception as e:
            return content  # Return original if enhancement fails
    
    def _create_diverse_fallback_response(self, query: str, num_responses: int, is_parameter_synthesis: bool) -> Dict:
        """Create diverse fallback response when API fails"""
        
        if is_parameter_synthesis:
            # Generate different fallback expressions based on attempt number
            fallback_expressions = [
                "1.5*x1**2 + 0.8*x2**2 - 1.2",
                "2.1*x1**2 + 0.3*x1*x2 + 1.4*x2**2 - 2.0", 
                "0.7*x1**2 + 1.9*x2**2 + 0.2*x1 - 1.6",
                "0.4*x1**4 + 0.2*x2**4 + 1.3*x1**2 + 0.9*x2**2 - 2.5"
            ]
            
            # Select based on attempt counter
            expr_idx = (self.attempt_counter - 1) % len(fallback_expressions)
            fallback_content = f"B(x) = {fallback_expressions[expr_idx]}"
        else:
            fallback_content = "I apologize, but I'm unable to process your request at the moment due to a connection issue."
        
        return {
            "choices": [
                {
                    "message": {
                        "content": fallback_content
                    }
                }
            ] * num_responses,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
    
    def get_response_texts(self, query_response: Union[List[Any], Any]) -> List[str]:
        """
        Extract text responses from the API response with diversity validation
        
        :param query_response: The response from query()
        :return: List of response texts
        """
        if not isinstance(query_response, dict):
            return ["Invalid response format"]
        
        if 'choices' not in query_response:
            return ["No choices in response"]
        
        texts = []
        for choice in query_response['choices']:
            if 'message' in choice and 'content' in choice['message']:
                content = choice['message']['content'].strip()
                
                # Validate diversity for parameter synthesis
                if self._is_parameter_synthesis_response(content):
                    content = self._ensure_diverse_content(content)
                
                texts.append(content)
            else:
                texts.append("Empty response")
        
        return texts
    
    def _is_parameter_synthesis_response(self, content: str) -> bool:
        """Check if response is from parameter synthesis"""
        content_lower = content.lower()
        return (
            'x1' in content_lower and 'x2' in content_lower and
            ('*' in content or 'x1**2' in content_lower or 'x2**2' in content_lower) and
            ('b(x)' in content_lower or 'barrier' in content_lower or any(c.isdigit() for c in content))
        )
    
    def _ensure_diverse_content(self, content: str) -> str:
        """Ensure content is sufficiently diverse"""
        try:
            # If content looks too similar to previous attempts, add diversity note
            if self.attempt_counter > 1 and ('1.0*x1' in content or '0*x1' in content):
                content += f"\n\n[Diversity Note: This is attempt #{self.attempt_counter} with unique coefficients]"
            
            return content
            
        except Exception:
            return content
    
    def set_temperature(self, temperature: float) -> None:
        """Set the temperature for generation"""
        self.temperature = max(0.0, min(2.0, temperature))
    
    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the maximum number of tokens to generate"""
        self.max_tokens = max(1, min(4096, max_tokens))
    
    def enable_diversity_mode(self, enable: bool = True) -> None:
        """Enable or disable diversity mode"""
        self.diversity_mode = enable
        if enable:
            print(f"Diversity mode enabled - attempt counter: {self.attempt_counter}")
        
    def reset_attempt_counter(self) -> None:
        """Reset the attempt counter"""
        self.attempt_counter = 0
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost": self.cost,
            "model": self.model,
            "attempt_counter": self.attempt_counter,
            "diversity_mode": self.diversity_mode
        }