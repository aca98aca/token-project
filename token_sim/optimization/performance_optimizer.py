from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import time
import logging
from functools import wraps

class OptimizationStrategy(Enum):
    """Types of optimization strategies."""
    PARALLEL = "parallel"
    VECTORIZED = "vectorized"
    CACHED = "cached"
    BATCHED = "batched"

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    latency: float
    cache_hits: int
    cache_misses: int

class PerformanceOptimizer:
    """Optimizes simulation performance and resource usage."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize the performance optimizer.
        
        Args:
            max_workers: Maximum number of worker processes/threads
        """
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be at least 1")
            
        self.max_workers = max_workers or mp.cpu_count()
        self.metrics_history: List[PerformanceMetrics] = []
        self.cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def optimize_function(self, func: Callable, strategy: OptimizationStrategy) -> Callable:
        """Apply optimization strategy to a function.
        
        Args:
            func: Function to optimize
            strategy: Optimization strategy to apply
            
        Returns:
            Optimized function
            
        Raises:
            ValueError: If strategy is invalid or function is None
        """
        if func is None:
            raise ValueError("Function cannot be None")
            
        if not isinstance(strategy, OptimizationStrategy):
            raise ValueError(f"Invalid strategy: {strategy}. Must be an OptimizationStrategy enum value.")
            
        try:
            if strategy == OptimizationStrategy.PARALLEL:
                return self._parallelize_function(func)
            elif strategy == OptimizationStrategy.VECTORIZED:
                return self._vectorize_function(func)
            elif strategy == OptimizationStrategy.CACHED:
                return self._cache_function(func)
            elif strategy == OptimizationStrategy.BATCHED:
                return self._batch_function(func)
            else:
                raise ValueError(f"Unknown optimization strategy: {strategy}")
        except Exception as e:
            self.logger.error(f"Error optimizing function: {str(e)}")
            raise
    
    def _parallelize_function(self, func: Callable) -> Callable:
        """Parallelize a function using process pool.
        
        Args:
            func: Function to parallelize
            
        Returns:
            Parallelized function
        """
        @wraps(func)
        def parallelized_func(*args, **kwargs):
            start_time = time.time()
            
            # Determine if we should use processes or threads
            if self._is_cpu_bound(func):
                executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
            try:
                # Split input data for parallel processing
                chunks = self._split_data_for_parallel(args[0])
                
                # Process chunks in parallel
                results = list(executor.map(func, chunks))
                
                # Combine results
                combined_result = self._combine_parallel_results(results)
                
                # Record metrics
                self._record_metrics(start_time)
                
                return combined_result
            finally:
                executor.shutdown()
        
        return parallelized_func
    
    def _vectorize_function(self, func: Callable) -> Callable:
        """Vectorize a function using numpy operations.
        
        Args:
            func: Function to vectorize
            
        Returns:
            Vectorized function
        """
        @wraps(func)
        def vectorized_func(*args, **kwargs):
            start_time = time.time()
            
            # Convert inputs to numpy arrays
            numpy_args = [np.array(arg) if isinstance(arg, (list, tuple)) else arg for arg in args]
            
            # Apply vectorized operations
            result = func(*numpy_args, **kwargs)
            
            # Record metrics
            self._record_metrics(start_time)
            
            return result
        
        return vectorized_func
    
    def _cache_function(self, func: Callable) -> Callable:
        """Cache function results.
        
        Args:
            func: Function to cache
            
        Returns:
            Cached function
        """
        @wraps(func)
        def cached_func(*args, **kwargs):
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(func, args, kwargs)
            
            # Check cache
            if cache_key in self.cache:
                self.metrics_history[-1].cache_hits += 1
                return self.cache[cache_key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            self.cache[cache_key] = result
            self.metrics_history[-1].cache_misses += 1
            
            # Record metrics
            self._record_metrics(start_time)
            
            return result
        
        return cached_func
    
    def _batch_function(self, func: Callable) -> Callable:
        """Process data in batches.
        
        Args:
            func: Function to batch
            
        Returns:
            Batched function
        """
        @wraps(func)
        def batched_func(*args, **kwargs):
            start_time = time.time()
            
            # Get batch size from kwargs or use default
            batch_size = kwargs.pop('batch_size', 1000)
            
            # Split data into batches
            batches = self._split_into_batches(args[0], batch_size)
            
            # Process batches
            results = []
            for batch in batches:
                result = func(batch, **kwargs)
                results.append(result)
            
            # Combine results
            combined_result = self._combine_batch_results(results)
            
            # Record metrics
            self._record_metrics(start_time)
            
            return combined_result
        
        return batched_func
    
    def _is_cpu_bound(self, func: Callable) -> bool:
        """Determine if a function is CPU-bound.
        
        Args:
            func: Function to analyze
            
        Returns:
            True if function is CPU-bound, False otherwise
        """
        # This is a simple heuristic - could be improved with profiling
        return func.__name__ in ['calculate_hashrate', 'process_block', 'validate_transaction']
    
    def _split_data_for_parallel(self, data: Any) -> List[Any]:
        """Split data for parallel processing.
        
        Args:
            data: Data to split
            
        Returns:
            List of data chunks
        """
        if isinstance(data, (list, tuple)):
            chunk_size = len(data) // self.max_workers
            return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        elif isinstance(data, pd.DataFrame):
            return np.array_split(data, self.max_workers)
        else:
            return [data] * self.max_workers
    
    def _combine_parallel_results(self, results: List[Any]) -> Any:
        """Combine results from parallel processing.
        
        Args:
            results: List of results to combine
            
        Returns:
            Combined result
        """
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results)
        elif isinstance(results[0], (list, tuple)):
            return [item for sublist in results for item in sublist]
        else:
            return results
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate a cache key for a function call.
        
        Args:
            func: Function being cached
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Cache key string
        """
        # Convert args and kwargs to a string representation
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        return f"{func.__name__}:{args_str}:{kwargs_str}"
    
    def _split_into_batches(self, data: Any, batch_size: int) -> List[Any]:
        """Split data into batches.
        
        Args:
            data: Data to split
            batch_size: Size of each batch
            
        Returns:
            List of data batches
        """
        if isinstance(data, (list, tuple)):
            return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        elif isinstance(data, pd.DataFrame):
            return np.array_split(data, len(data) // batch_size + 1)
        else:
            return [data]
    
    def _combine_batch_results(self, results: List[Any]) -> Any:
        """Combine results from batch processing.
        
        Args:
            results: List of batch results
            
        Returns:
            Combined result
        """
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results)
        elif isinstance(results[0], (list, tuple)):
            return [item for sublist in results for item in sublist]
        else:
            return results
    
    def _record_metrics(self, start_time: float) -> None:
        """Record performance metrics.
        
        Args:
            start_time: Start time of the operation
        """
        execution_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_usage = psutil.Process().cpu_percent()
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=1 / execution_time if execution_time > 0 else 0,
            latency=execution_time,
            cache_hits=0,
            cache_misses=0
        )
        
        self.metrics_history.append(metrics)
        
        # Check memory usage and cleanup if necessary
        if memory_usage > 1000:  # If memory usage exceeds 1GB
            self._cleanup_memory()
    
    def _cleanup_memory(self) -> None:
        """Clean up memory by removing old cache entries and metrics."""
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # Clear cache if it's too large
        if len(self.cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(self.cache.items(), key=lambda x: x[1].get('timestamp', 0))
            self.cache = dict(sorted_cache[-1000:])
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.clear_cache()
        self.reset_metrics()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report.
        
        Returns:
            Dictionary containing performance metrics and analysis
        """
        if not self.metrics_history:
            return {}
        
        # Calculate aggregate metrics
        total_execution_time = sum(m.execution_time for m in self.metrics_history)
        avg_memory_usage = np.mean([m.memory_usage for m in self.metrics_history])
        avg_cpu_usage = np.mean([m.cpu_usage for m in self.metrics_history])
        total_throughput = sum(m.throughput for m in self.metrics_history)
        avg_latency = np.mean([m.latency for m in self.metrics_history])
        total_cache_hits = sum(m.cache_hits for m in self.metrics_history)
        total_cache_misses = sum(m.cache_misses for m in self.metrics_history)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': total_execution_time,
            'average_memory_usage': avg_memory_usage,
            'average_cpu_usage': avg_cpu_usage,
            'total_throughput': total_throughput,
            'average_latency': avg_latency,
            'cache_hits': total_cache_hits,
            'cache_misses': total_cache_misses,
            'cache_hit_ratio': total_cache_hits / (total_cache_hits + total_cache_misses) if (total_cache_hits + total_cache_misses) > 0 else 0
        }
        
        return report
    
    def monitor_performance(self, interval: float = 60.0) -> None:
        """Monitor performance metrics at regular intervals.
        
        Args:
            interval: Monitoring interval in seconds
        """
        import threading
        import time
        
        def monitor_loop():
            while True:
                try:
                    report = self.get_performance_report()
                    if report:
                        self.logger.info("Performance Metrics:")
                        self.logger.info(f"Memory Usage: {report['average_memory_usage']:.2f} MB")
                        self.logger.info(f"CPU Usage: {report['average_cpu_usage']:.2f}%")
                        self.logger.info(f"Throughput: {report['total_throughput']:.2f} ops/sec")
                        self.logger.info(f"Latency: {report['average_latency']:.2f} sec")
                        self.logger.info(f"Cache Hit Ratio: {report['cache_hit_ratio']:.2%}")
                    
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring: {str(e)}")
                    time.sleep(interval)
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def clear_cache(self) -> None:
        """Clear the function result cache."""
        self.cache.clear()
    
    def reset_metrics(self) -> None:
        """Reset performance metrics history."""
        self.metrics_history.clear() 