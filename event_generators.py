#!/usr/bin/env python3
"""
Event Generators for Stochastic Processes
Implements Section 3.3 and 3.4 from the LaTeX specification
"""

import numpy as np
import random
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import json

class HawkesEventGenerator:
    """
    Hawkes process event generator using Ogata's thinning algorithm.
    
    Implements the lean Hawkes process for near-side depletions:
    λ_s(t) = μ_s + α_s * Σ e^(-β_s * (t - t_k^s))
    
    where n_s = α_s/β_s < 1 (branching ratio < 1)
    """
    
    def __init__(self, mu: float, alpha: float, beta: float, 
                 mixture_proportion: float = 0.5, seed: int = 42):
        """
        Initialize Hawkes process parameters.
        
        Args:
            mu: Baseline rate
            alpha: Excitation parameter
            beta: Decay parameter
            mixture_proportion: Probability that depletion is caused by trade vs cancel
            seed: Random seed
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.mixture_proportion = mixture_proportion
        self.seed = seed
        
        # Validate parameters
        self.branching_ratio = alpha / beta if beta > 0 else 0
        if self.branching_ratio >= 1.0:
            raise ValueError(f"Branching ratio must be < 1, got {self.branching_ratio}")
        
        # State variables for Ogata thinning
        self.decayed_sum = 0.0  # S_s(t) = Σ e^(-β_s * (t - t_k^s))
        self.last_event_time = 0.0
        self.event_times = []  # Store event times for debugging
        
        # Set random seed
        np.random.seed(seed)
        random.seed(seed)
    
    def get_intensity(self, t: float) -> float:
        """
        Calculate the intensity at time t.
        
        Args:
            t: Current time
            
        Returns:
            Intensity value
        """
        return self.mu + self.alpha * self.decayed_sum
    
    def generate_events(self, start_time: float, end_time: float, 
                       callback: Callable[[float, str], None]) -> List[float]:
        """
        Generate events using Ogata's thinning algorithm.
        
        Args:
            start_time: Start time for generation
            end_time: End time for generation
            callback: Function to call when event is generated (time, event_type)
            
        Returns:
            List of event times
        """
        self.last_event_time = start_time
        self.decayed_sum = 0.0
        self.event_times = []
        
        current_time = start_time
        
        while current_time < end_time:
            # Calculate current intensity
            current_intensity = self.get_intensity(current_time)
            
            if current_intensity <= 0:
                # No events possible, advance time
                current_time += 0.001  # Small time step
                continue
            
            # Set dominating rate (current intensity, which only decreases)
            M = current_intensity
            
            # Draw inter-arrival time from exponential distribution
            delta = np.random.exponential(1.0 / M)
            candidate_time = current_time + delta
            
            if candidate_time >= end_time:
                break
            
            # Update decayed sum
            time_diff = candidate_time - self.last_event_time
            self.decayed_sum *= np.exp(-self.beta * time_diff)
            
            # Calculate intensity at candidate time
            candidate_intensity = self.get_intensity(candidate_time)
            
            # Accept with probability λ(t')/M
            if np.random.random() < (candidate_intensity / M):
                # Event accepted
                self.event_times.append(candidate_time)
                self.decayed_sum += 1.0
                self.last_event_time = candidate_time
                
                # Determine event type (trade vs cancel)
                event_type = "trade" if np.random.random() < self.mixture_proportion else "cancel"
                
                # Call callback
                callback(candidate_time, event_type)
            
            current_time = candidate_time
        
        return self.event_times
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'mu': self.mu,
            'alpha': self.alpha,
            'beta': self.beta,
            'branching_ratio': self.branching_ratio,
            'half_life': np.log(2) / self.beta if self.beta > 0 else 0,
            'mixture_proportion': self.mixture_proportion,
            'events_generated': len(self.event_times)
        }

class PoissonEventGenerator:
    """
    Inhomogeneous Poisson process event generator for liquidity additions.
    
    Uses the bucketed rates λ_add(t) calibrated from data.
    """
    
    def __init__(self, rates: Dict[str, float], size_histogram: Dict[str, int], 
                 seed: int = 42):
        """
        Initialize Poisson process generator.
        
        Args:
            rates: Dictionary of time bucket rates (bucket_key -> rate_per_minute)
            size_histogram: Dictionary of size frequencies (size -> count)
            seed: Random seed
        """
        self.rates = rates
        self.size_histogram = size_histogram
        self.seed = seed
        
        # Convert size histogram to probability distribution
        self.size_distribution = self._create_size_distribution()
        
        # Event tracking
        self.event_times = []
        
        # Set random seed
        np.random.seed(seed)
        random.seed(seed)
    
    def _create_size_distribution(self) -> List[int]:
        """Convert size histogram to list of sizes for sampling."""
        sizes = []
        for size_str, count in self.size_histogram.items():
            size = int(size_str)
            sizes.extend([size] * count)
        return sizes
    
    def generate_events(self, start_time: float, end_time: float,
                       callback: Callable[[float, int], None]) -> List[float]:
        """
        Generate events using inhomogeneous Poisson process.
        
        Args:
            start_time: Start time for generation
            end_time: End time for generation
            callback: Function to call when event is generated (time, size)
            
        Returns:
            List of event times
        """
        self.event_times = []  # Reset event times
        current_time = start_time
        
        # Use 1-minute buckets as specified
        bucket_size = 60.0  # 1 minute in seconds
        
        while current_time < end_time:
            bucket_end = min(current_time + bucket_size, end_time)
            
            # Get rate for this bucket - use first available rate if no exact match
            bucket_key = f"{int(current_time)}-{int(bucket_end)}"
            rate = self.rates.get(bucket_key, 0.0)
            
            # If no exact match, use the first available rate
            if rate == 0.0 and self.rates:
                rate = list(self.rates.values())[0]
            
            if rate > 0:
                # Generate events in this bucket
                bucket_events = self._generate_bucket_events(
                    current_time, bucket_end, rate, callback
                )
                self.event_times.extend(bucket_events)
            
            current_time = bucket_end
        
        return self.event_times
    
    def _generate_bucket_events(self, start_time: float, end_time: float, 
                               rate: float, callback: Callable[[float, int], None]) -> List[float]:
        """Generate events within a single time bucket."""
        bucket_events = []
        current_time = start_time
        
        # Convert rate from per-minute to per-second
        rate_per_second = rate / 60.0
        
        while current_time < end_time:
            # Draw inter-arrival time from exponential distribution
            delta = np.random.exponential(1.0 / rate_per_second)
            event_time = current_time + delta
            
            if event_time >= end_time:
                break
            
            # Draw size from histogram
            size = random.choice(self.size_distribution) if self.size_distribution else 100
            
            # Call callback
            callback(event_time, size)
            bucket_events.append(event_time)
            self.event_times.append(event_time)  # Track event time
            
            current_time = event_time
        
        return bucket_events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'num_buckets': len(self.rates),
            'avg_rate': np.mean(list(self.rates.values())) if self.rates else 0,
            'size_distribution_size': len(self.size_distribution),
            'events_generated': 0  # Will be updated during generation
        }

class EventGeneratorManager:
    """
    Manages all event generators and coordinates event generation.
    """
    
    def __init__(self, simulator, exchanges: List, latency_model, seed: int = 42):
        self.simulator = simulator
        self.exchanges = exchanges
        self.latency_model = latency_model
        self.seed = seed
        
        # Event generators
        self.hawkes_generators = {}  # side -> generator
        self.poisson_generator = None
        
        # Statistics
        self.stats = {
            'hawkes_events': 0,
            'poisson_events': 0,
            'total_events': 0
        }
    
    def setup_hawkes_generators(self, bid_params: Dict[str, float], 
                               ask_params: Dict[str, float]):
        """Setup Hawkes generators for bid and ask sides."""
        self.hawkes_generators['B'] = HawkesEventGenerator(
            mu=bid_params['mu'],
            alpha=bid_params['alpha'],
            beta=bid_params['beta'],
            mixture_proportion=bid_params.get('mixture_proportion', 0.5),
            seed=self.seed
        )
        
        self.hawkes_generators['A'] = HawkesEventGenerator(
            mu=ask_params['mu'],
            alpha=ask_params['alpha'],
            beta=ask_params['beta'],
            mixture_proportion=ask_params.get('mixture_proportion', 0.5),
            seed=self.seed + 1  # Different seed for ask side
        )
    
    def setup_poisson_generator(self, rates: Dict[str, float], 
                               size_histogram: Dict[str, int]):
        """Setup Poisson generator for liquidity additions."""
        self.poisson_generator = PoissonEventGenerator(rates, size_histogram, seed=self.seed + 2)
    
    def generate_hawkes_events(self, start_time: float, end_time: float):
        """Generate Hawkes events for both sides."""
        for side, generator in self.hawkes_generators.items():
            def callback(event_time, event_type):
                self._handle_hawkes_event(event_time, side, event_type)
            
            generator.generate_events(start_time, end_time, callback)
            self.stats['hawkes_events'] += len(generator.event_times)
    
    def generate_poisson_events(self, start_time: float, end_time: float):
        """Generate Poisson events for liquidity additions."""
        if not self.poisson_generator:
            return
        
        def callback(event_time, size):
            self._handle_poisson_event(event_time, size)
        
        self.poisson_generator.generate_events(start_time, end_time, callback)
        self.stats['poisson_events'] += len(self.poisson_generator.event_times)
    
    def _handle_hawkes_event(self, event_time: float, side: str, event_type: str):
        """Handle a Hawkes event (depletion)."""
        # Select a random exchange for depletion
        target_exchange = random.choice([ex.name for ex in self.exchanges])
        
        # Schedule depletion event
        self.simulator.schedule_event_at(
            event_time,
            "depletion_event",
            {
                'side': side,
                'event_type': event_type,
                'target_exchange': target_exchange
            }
        )
        
        self.stats['total_events'] += 1
    
    def _handle_poisson_event(self, event_time: float, size: int):
        """Handle a Poisson event (liquidity addition)."""
        # Select a random exchange
        target_exchange = random.choice([ex.name for ex in self.exchanges])
        
        # Schedule addition event
        self.simulator.schedule_event_at(
            event_time,
            "addition_event",
            {
                'size': size,
                'target_exchange': target_exchange
            }
        )
        
        self.stats['total_events'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        stats = self.stats.copy()
        
        # Add Hawkes generator stats
        for side, generator in self.hawkes_generators.items():
            stats[f'hawkes_{side}'] = generator.get_stats()
        
        # Add Poisson generator stats
        if self.poisson_generator:
            stats['poisson'] = self.poisson_generator.get_stats()
        
        return stats

if __name__ == "__main__":
    # Test the event generators
    print("Testing Event Generators")
    print("=" * 40)
    
    # Test Hawkes generator
    print("\n1. Testing Hawkes Generator:")
    print("-" * 30)
    
    hawkes_gen = HawkesEventGenerator(
        mu=0.1,      # 0.1 events/second baseline
        alpha=0.05,  # Moderate excitation
        beta=0.2,    # Decay rate
        mixture_proportion=0.6  # 60% trades, 40% cancels
    )
    
    events = []
    def hawkes_callback(time, event_type):
        events.append((time, event_type))
        print(f"  Hawkes event at {time:.3f}s: {event_type}")
    
    hawkes_gen.generate_events(0.0, 10.0, hawkes_callback)
    print(f"Generated {len(events)} Hawkes events")
    print(f"Stats: {hawkes_gen.get_stats()}")
    
    # Test Poisson generator
    print("\n2. Testing Poisson Generator:")
    print("-" * 30)
    
    # Sample rates and histogram
    rates = {
        "0-60": 5.0,    # 5 events/minute
        "60-120": 3.0,  # 3 events/minute
        "120-180": 7.0  # 7 events/minute
    }
    
    size_histogram = {
        "100": 10,
        "200": 15,
        "500": 5,
        "1000": 2
    }
    
    poisson_gen = PoissonEventGenerator(rates, size_histogram)
    
    poisson_events = []
    def poisson_callback(time, size):
        poisson_events.append((time, size))
        print(f"  Poisson event at {time:.3f}s: size {size}")
    
    poisson_gen.generate_events(0.0, 180.0, poisson_callback)
    print(f"Generated {len(poisson_events)} Poisson events")
    print(f"Stats: {poisson_gen.get_stats()}")
    
    print("\nEvent generator test complete!")
