"""Utility helper functions for the Empathy System."""

import time
from functools import wraps
from typing import Callable, Any
from loguru import logger

def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Useful for profiling latency-critical components.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        logger.debug(f"{func.__name__} took {elapsed:.2f}ms")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed:.2f}ms")
        return result
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            raise last_exception
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class LatencyTracker:
    """Track latency metrics for system components."""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, component: str, latency_ms: float):
        """Record a latency measurement."""
        if component not in self.metrics:
            self.metrics[component] = []
        self.metrics[component].append(latency_ms)
    
    def get_stats(self, component: str) -> dict:
        """Get statistics for a component."""
        if component not in self.metrics or not self.metrics[component]:
            return {}
        
        latencies = self.metrics[component]
        return {
            'count': len(latencies),
            'mean': sum(latencies) / len(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'p50': sorted(latencies)[len(latencies) // 2],
            'p95': sorted(latencies)[int(len(latencies) * 0.95)],
            'p99': sorted(latencies)[int(len(latencies) * 0.99)]
        }
    
    def report(self):
        """Log latency report for all components."""
        logger.info("=== Latency Report ===")
        for component in self.metrics:
            stats = self.get_stats(component)
            logger.info(
                f"{component}: "
                f"mean={stats['mean']:.1f}ms, "
                f"p50={stats['p50']:.1f}ms, "
                f"p95={stats['p95']:.1f}ms, "
                f"p99={stats['p99']:.1f}ms"
            )
