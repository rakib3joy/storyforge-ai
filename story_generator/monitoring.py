"""
Token Monitoring and Cost Tracking - Production Monitoring with Real Tokenization
"""

import logging

# Setup logging
logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    logger.info("tiktoken available - using precise tokenization")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available - using heuristic approximation")

class TokenMonitor:
    """Track API usage and costs with precise tokenization"""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Approximate pricing (update with actual rates)
        self.input_cost_per_1k = 0.00015  # $0.00015 per 1K input tokens
        self.output_cost_per_1k = 0.0006   # $0.0006 per 1K output tokens
        
        # Initialize tokenizer if available
        if TIKTOKEN_AVAILABLE:
            try:
                # Use GPT-4 tokenizer as close approximation to Gemini
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
                self.tokenization_method = "tiktoken-precise"
                logger.info("Initialized precise tokenization with tiktoken")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}")
                self.tokenizer = None
                self.tokenization_method = "heuristic-fallback"
        else:
            self.tokenizer = None
            self.tokenization_method = "heuristic-approximation"
    
    def estimate_tokens(self, text):
        """
        Precise token estimation using tiktoken or heuristic fallback
        """
        if not text:
            return 0
            
        if self.tokenizer:
            try:
                # Use precise tokenization
                tokens = len(self.tokenizer.encode(text))
                logger.debug(f"Precise tokenization: {len(text)} chars → {tokens} tokens")
                return tokens
            except Exception as e:
                logger.warning(f"Tokenization failed, using fallback: {e}")
                
        # Fallback to heuristic (4 chars ≈ 1 token)
        heuristic_tokens = len(text) // 4
        logger.debug(f"Heuristic tokenization: {len(text)} chars → {heuristic_tokens} tokens")
        return heuristic_tokens
    
    def add_usage(self, input_text, output_text):
        """Add token usage with method tracking"""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        total_cost = input_cost + output_cost
        
        logger.info(f"Token usage tracked: {input_tokens}→{output_tokens} (method: {self.tokenization_method})")
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "tokenization_method": self.tokenization_method
        }
    
    def get_session_stats(self):
        """Get session statistics with tokenization method info"""
        total_cost = ((self.total_input_tokens / 1000) * self.input_cost_per_1k) + \
                    ((self.total_output_tokens / 1000) * self.output_cost_per_1k)
        
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": total_cost,
            "tokenization_method": self.tokenization_method,
            "precision_level": "high" if self.tokenizer else "approximate"
        }
    
    def reset_session(self):
        """Reset session statistics"""
        logger.info("Resetting token monitor session")
        self.total_input_tokens = 0
        self.total_output_tokens = 0

# Global token monitor instance
token_monitor = TokenMonitor()