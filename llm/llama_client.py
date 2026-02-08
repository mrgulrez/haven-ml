"""Llama 3.1 LLM integration with streaming support.

Provides empathetic text generation based on emotional context.
"""

import torch
from typing import Dict, Optional, Iterator, List
from loguru import logger

from utils.helpers import timeit


class LlamaClient:
    """
    Client for Llama 3.1 LLM with 4-bit quantization.
    
    Supports streaming generation for low-latency responses.
    """
    
    def __init__(
        self,
        model_path: str,
        context_length: int = 8192,
        n_gpu_layers: int = 40,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 512,
        streaming: bool = True
    ):
        """
        Initialize Llama client.
        
        Args:
            model_path: Path to GGUF model file
            context_length: Maximum context window
            n_gpu_layers: Number of layers to offload to GPU
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            streaming: Enable token streaming
        """
        self.model_path = model_path
        self.context_length = context_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.streaming = streaming
        
        try:
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_gpu_layers=n_gpu_layers,
                n_threads=8,
                verbose=False
            )
            
            logger.info(f"Llama client initialized from {model_path}")
            logger.info(f"GPU layers: {n_gpu_layers}, Context: {context_length}")
            
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise
    
    @timeit
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text completion (blocking).
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                stop=["<|eot_id|>", "\n\nUser:", "\n\nHuman:"],
                echo=False
            )
            
            text = response['choices'][0]['text'].strip()
            return text
            
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return ""
    
    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """
        Generate text completion (streaming).
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Yields:
            Generated tokens
        """
        try:
            stream = self.model(
                prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                stop=["<|eot_id|>", "\n\nUser:", "\n\nHuman:"],
                echo=False,
                stream=True
            )
            
            for output in stream:
                token = output['choices'][0]['text']
                yield token
                
        except Exception as e:
            logger.error(f"Error in generate_stream: {e}")
            yield ""
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4


class MockLlamaClient:
    """Mock LLM client for testing."""
    
    def __init__(self, *args, **kwargs):
        logger.warning("Using MockLlamaClient - install llama-cpp-python for real LLM")
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Return canned empathetic response
        return "I'm here for you. How can I help?"
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        response = "I'm here to support you."
        for word in response.split():
            yield word + " "
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
