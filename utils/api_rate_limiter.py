"""
API Rate Limiting System

This module provides intelligent rate limiting for OpenRouter API calls with
exponential backoff, request queuing, and adaptive throttling.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from threading import Lock, Event
from queue import Queue, Empty
import random

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60  # Default OpenRouter free tier limit
    requests_per_hour: int = 1000
    max_concurrent_requests: int = 5
    base_delay: float = 1.0  # Base delay for exponential backoff
    max_delay: float = 60.0  # Maximum delay
    jitter: bool = True  # Add random jitter to prevent thundering herd


class APIRateLimiter:
    """
    Intelligent rate limiter for API calls with exponential backoff and queuing.
    
    Features:
    - Token bucket algorithm for rate limiting
    - Exponential backoff with jitter
    - Request queuing
    - Adaptive throttling based on API responses
    - Thread-safe operation
    """
    
    def __init__(self, config: RateLimitConfig = None):
        """Initialize the rate limiter."""
        self.config = config or RateLimitConfig()
        
        # Token bucket for rate limiting
        self.tokens = self.config.requests_per_minute
        self.max_tokens = self.config.requests_per_minute
        self.last_refill = time.time()
        
        # Request tracking
        self.request_times: List[float] = []
        self.hourly_request_times: List[float] = []
        
        # Concurrent request tracking
        self.active_requests = 0
        self.max_concurrent = self.config.max_concurrent_requests
        
        # Backoff state
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        
        # Thread safety
        self.lock = Lock()
        self.request_event = Event()
        self.request_event.set()  # Initially allow requests
        
        logger.info(f"Initialized API rate limiter: {self.config.requests_per_minute} req/min, {self.config.max_concurrent_requests} concurrent")
    
    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make an API request.
        
        Args:
            timeout: Maximum time to wait for permission
            
        Returns:
            True if permission granted, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                # Check if we're in backoff period
                if self._is_in_backoff():
                    backoff_delay = self._calculate_backoff_delay()
                    logger.debug(f"In backoff period, waiting {backoff_delay:.2f}s")
                    time.sleep(min(backoff_delay, timeout - (time.time() - start_time)))
                    continue
                
                # Refill token bucket
                self._refill_tokens()
                
                # Check rate limits
                if not self._check_rate_limits():
                    # Calculate wait time for next available slot
                    wait_time = self._calculate_wait_time()
                    if wait_time > 0:
                        logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                        time.sleep(min(wait_time, timeout - (time.time() - start_time)))
                        continue
                
                # Check concurrent request limit
                if self.active_requests >= self.max_concurrent:
                    logger.debug(f"Concurrent limit reached ({self.active_requests}/{self.max_concurrent})")
                    time.sleep(0.1)  # Brief wait before retry
                    continue
                
                # Permission granted
                self.tokens -= 1
                self.active_requests += 1
                current_time = time.time()
                self.request_times.append(current_time)
                self.hourly_request_times.append(current_time)
                
                # Clean old timestamps
                self._cleanup_timestamps()
                
                logger.debug(f"API request permission granted. Tokens: {self.tokens}, Active: {self.active_requests}")
                return True
        
        logger.warning(f"API request permission timeout after {timeout}s")
        return False
    
    def release(self, success: bool = True, response_headers: Dict[str, str] = None):
        """
        Release an API request slot and update rate limiting state.
        
        Args:
            success: Whether the request was successful
            response_headers: Response headers for adaptive throttling
        """
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
            
            if success:
                self.consecutive_failures = 0
                logger.debug(f"API request completed successfully. Active: {self.active_requests}")
            else:
                self.consecutive_failures += 1
                self.last_failure_time = time.time()
                logger.warning(f"API request failed. Consecutive failures: {self.consecutive_failures}")
            
            # Adaptive throttling based on response headers
            if response_headers:
                self._update_from_headers(response_headers)
    
    def _refill_tokens(self):
        """Refill the token bucket based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * (self.config.requests_per_minute / 60.0)
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = current_time
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        
        # Check token bucket
        if self.tokens < 1:
            return False
        
        # Check per-minute limit
        minute_ago = current_time - 60
        recent_requests = [t for t in self.request_times if t > minute_ago]
        if len(recent_requests) >= self.config.requests_per_minute:
            return False
        
        # Check per-hour limit
        hour_ago = current_time - 3600
        hourly_requests = [t for t in self.hourly_request_times if t > hour_ago]
        if len(hourly_requests) >= self.config.requests_per_hour:
            return False
        
        return True
    
    def _calculate_wait_time(self) -> float:
        """Calculate how long to wait before next request."""
        current_time = time.time()
        
        # Check when oldest request in current minute will expire
        minute_ago = current_time - 60
        recent_requests = [t for t in self.request_times if t > minute_ago]
        
        if len(recent_requests) >= self.config.requests_per_minute:
            # Wait until oldest request expires
            oldest_request = min(recent_requests)
            wait_time = (oldest_request + 60) - current_time
            return max(0, wait_time)
        
        # Check token bucket refill time
        if self.tokens < 1:
            tokens_needed = 1 - self.tokens
            refill_time = tokens_needed / (self.config.requests_per_minute / 60.0)
            return refill_time
        
        return 0.0
    
    def _is_in_backoff(self) -> bool:
        """Check if we're currently in exponential backoff period."""
        if self.consecutive_failures == 0:
            return False
        
        backoff_delay = self._calculate_backoff_delay()
        return time.time() - self.last_failure_time < backoff_delay
    
    def _calculate_backoff_delay(self) -> float:
        """Calculate exponential backoff delay."""
        if self.consecutive_failures == 0:
            return 0.0
        
        # Exponential backoff: base_delay * 2^(failures-1)
        delay = self.config.base_delay * (2 ** (self.consecutive_failures - 1))
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay
    
    def _cleanup_timestamps(self):
        """Remove old timestamps to prevent memory growth."""
        current_time = time.time()
        
        # Keep only last minute for per-minute tracking
        minute_ago = current_time - 60
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        # Keep only last hour for per-hour tracking
        hour_ago = current_time - 3600
        self.hourly_request_times = [t for t in self.hourly_request_times if t > hour_ago]
    
    def _update_from_headers(self, headers: Dict[str, str]):
        """Update rate limiting based on API response headers."""
        # Common rate limit headers
        remaining = headers.get('x-ratelimit-remaining') or headers.get('ratelimit-remaining')
        reset_time = headers.get('x-ratelimit-reset') or headers.get('ratelimit-reset')
        
        if remaining:
            try:
                remaining_requests = int(remaining)
                if remaining_requests < 5:  # Low remaining requests
                    logger.warning(f"Low API rate limit remaining: {remaining_requests}")
                    # Reduce token bucket to be more conservative
                    self.tokens = min(self.tokens, remaining_requests)
            except ValueError:
                pass
        
        if reset_time:
            try:
                reset_timestamp = int(reset_time)
                current_time = time.time()
                if reset_timestamp > current_time:
                    # Adjust our rate limiting to match API reset time
                    time_until_reset = reset_timestamp - current_time
                    if time_until_reset < 60:  # Less than a minute until reset
                        self.tokens = min(self.tokens, 1)  # Be very conservative
            except ValueError:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        current_time = time.time()
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        
        recent_requests = len([t for t in self.request_times if t > minute_ago])
        hourly_requests = len([t for t in self.hourly_request_times if t > hour_ago])
        
        return {
            'tokens_available': self.tokens,
            'max_tokens': self.max_tokens,
            'active_requests': self.active_requests,
            'max_concurrent': self.max_concurrent,
            'requests_last_minute': recent_requests,
            'requests_last_hour': hourly_requests,
            'consecutive_failures': self.consecutive_failures,
            'in_backoff': self._is_in_backoff(),
            'backoff_delay': self._calculate_backoff_delay() if self._is_in_backoff() else 0.0
        }


# Global rate limiter instance
_rate_limiter = None

def get_rate_limiter(config: RateLimitConfig = None) -> APIRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = APIRateLimiter(config)
    return _rate_limiter
