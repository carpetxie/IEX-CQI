#!/usr/bin/env python3
"""
Latency Model for Market Data and Order Paths
Implements Section 2 from the LaTeX specification
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum

class LatencyRegime(Enum):
    """Latency dispersion regimes for robustness testing."""
    NARROW = "narrow"
    MEDIUM = "medium"
    WIDE = "wide"

class LatencyModel:
    """
    Latency model for data and order paths.
    
    Separates market-data and order paths:
    - L^data_{v→a}: one-way time for venue v's state change to reach agent/venue a
    - L^ord_{a→v}: one-way time for agent a's order to arrive at venue v
    
    IEX applies a fixed inbound delay: L^ord_{a→IEX} ← L^ord_{a→IEX} + 350μs
    """
    
    def __init__(self, regime: LatencyRegime = LatencyRegime.MEDIUM, seed: int = 42):
        self.regime = regime
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define latency parameters based on regime
        self._setup_latency_parameters()
        
        # IEX inbound delay (350 microseconds)
        self.iex_inbound_delay = 0.00035  # 350 microseconds in seconds
        
        # Cache for computed latencies
        self._latency_cache = {}
    
    def _setup_latency_parameters(self):
        """Setup latency parameters based on the chosen regime."""
        if self.regime == LatencyRegime.NARROW:
            # Tight latency distribution
            self.data_latency_base = 0.0001  # 100 microseconds
            self.data_latency_std = 0.00002  # 20 microseconds std
            self.order_latency_base = 0.00005  # 50 microseconds
            self.order_latency_std = 0.00001  # 10 microseconds std
        elif self.regime == LatencyRegime.MEDIUM:
            # Moderate latency distribution
            self.data_latency_base = 0.0005  # 500 microseconds
            self.data_latency_std = 0.0001   # 100 microseconds std
            self.order_latency_base = 0.0002  # 200 microseconds
            self.order_latency_std = 0.00005  # 50 microseconds std
        else:  # WIDE
            # Wide latency distribution
            self.data_latency_base = 0.001   # 1 millisecond
            self.data_latency_std = 0.0005   # 500 microseconds std
            self.order_latency_base = 0.0005  # 500 microseconds
            self.order_latency_std = 0.0002   # 200 microseconds std
    
    def get_data_latency(self, from_venue: str, to_agent: str) -> float:
        """
        Get data path latency from venue to agent.
        
        Args:
            from_venue: Source venue name
            to_agent: Destination agent name
            
        Returns:
            Latency in seconds
        """
        cache_key = f"data_{from_venue}_{to_agent}"
        
        if cache_key not in self._latency_cache:
            # Generate latency using log-normal distribution for realistic values
            base_latency = self.data_latency_base
            std_latency = self.data_latency_std
            
            # Add some venue-specific variation
            venue_multiplier = self._get_venue_multiplier(from_venue)
            agent_multiplier = self._get_agent_multiplier(to_agent)
            
            # Generate latency with some correlation structure
            correlation_factor = self._get_correlation_factor(from_venue, to_agent)
            
            # Base latency with venue/agent adjustments
            adjusted_base = base_latency * venue_multiplier * agent_multiplier
            adjusted_std = std_latency * (1 + correlation_factor)
            
            # Generate log-normal distributed latency
            mu = np.log(adjusted_base) - 0.5 * (adjusted_std / adjusted_base) ** 2
            sigma = adjusted_std / adjusted_base
            
            latency = np.random.lognormal(mu, sigma)
            
            # Ensure minimum latency
            latency = max(latency, 0.00001)  # 10 microseconds minimum
            
            self._latency_cache[cache_key] = latency
        
        return self._latency_cache[cache_key]
    
    def get_order_latency(self, from_agent: str, to_venue: str) -> float:
        """
        Get order path latency from agent to venue.
        
        Args:
            from_agent: Source agent name
            to_venue: Destination venue name
            
        Returns:
            Latency in seconds (includes IEX inbound delay if applicable)
        """
        cache_key = f"order_{from_agent}_{to_venue}"
        
        if cache_key not in self._latency_cache:
            # Generate base latency
            base_latency = self.order_latency_base
            std_latency = self.order_latency_std
            
            # Add agent/venue specific variation
            agent_multiplier = self._get_agent_multiplier(from_agent)
            venue_multiplier = self._get_venue_multiplier(to_venue)
            
            # Generate latency
            adjusted_base = base_latency * agent_multiplier * venue_multiplier
            adjusted_std = std_latency * (1 + self._get_correlation_factor(from_agent, to_venue))
            
            # Generate log-normal distributed latency
            mu = np.log(adjusted_base) - 0.5 * (adjusted_std / adjusted_base) ** 2
            sigma = adjusted_std / adjusted_base
            
            latency = np.random.lognormal(mu, sigma)
            
            # Ensure minimum latency
            latency = max(latency, 0.00001)  # 10 microseconds minimum
            
            # Add IEX inbound delay if targeting IEX
            if to_venue.upper() == 'IEX':
                latency += self.iex_inbound_delay
            
            self._latency_cache[cache_key] = latency
        
        return self._latency_cache[cache_key]
    
    def _get_venue_multiplier(self, venue: str) -> float:
        """Get venue-specific latency multiplier."""
        venue_multipliers = {
            'IEX': 1.0,      # Baseline
            'A': 0.8,        # Faster
            'B': 0.9,        # Slightly faster
            'NASDAQ': 1.1,   # Slightly slower
            'NYSE': 1.2,     # Slower
        }
        return venue_multipliers.get(venue.upper(), 1.0)
    
    def _get_agent_multiplier(self, agent: str) -> float:
        """Get agent-specific latency multiplier."""
        if 'HFT' in agent.upper():
            return 0.5  # HFT agents are faster
        elif 'LP' in agent.upper():
            return 1.2  # LP agents are slower
        else:
            return 1.0  # Default
    
    def _get_correlation_factor(self, from_entity: str, to_entity: str) -> float:
        """Get correlation factor for latency between entities."""
        # Add some correlation structure
        # Entities with similar names or types might have correlated latencies
        if from_entity == to_entity:
            return 0.0  # No correlation with self
        
        # Add some random correlation
        return random.uniform(-0.3, 0.3)
    
    def get_all_latencies(self, venues: List[str], agents: List[str]) -> Dict[str, float]:
        """
        Get all possible latencies for the given venues and agents.
        
        Returns:
            Dictionary with latency keys and values
        """
        latencies = {}
        
        # Data path latencies (venue -> agent)
        for venue in venues:
            for agent in agents:
                key = f"data_{venue}_{agent}"
                latencies[key] = self.get_data_latency(venue, agent)
        
        # Order path latencies (agent -> venue)
        for agent in agents:
            for venue in venues:
                key = f"order_{agent}_{venue}"
                latencies[key] = self.get_order_latency(agent, venue)
        
        return latencies
    
    def get_latency_summary(self, venues: List[str], agents: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of latencies organized by type.
        
        Returns:
            Dictionary with 'data' and 'order' keys containing latency matrices
        """
        latencies = self.get_all_latencies(venues, agents)
        
        summary = {
            'data': {},
            'order': {}
        }
        
        # Organize data latencies
        for venue in venues:
            summary['data'][venue] = {}
            for agent in agents:
                key = f"data_{venue}_{agent}"
                summary['data'][venue][agent] = latencies[key]
        
        # Organize order latencies
        for agent in agents:
            summary['order'][agent] = {}
            for venue in venues:
                key = f"order_{agent}_{venue}"
                summary['order'][agent][venue] = latencies[key]
        
        return summary
    
    def reset_cache(self):
        """Reset the latency cache (useful for testing)."""
        self._latency_cache.clear()
    
    def set_seed(self, seed: int):
        """Set new random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.reset_cache()

if __name__ == "__main__":
    # Test the latency model
    print("Testing Latency Model")
    print("=" * 40)
    
    venues = ['IEX', 'A', 'B']
    agents = ['HFT1', 'LP1', 'LP2']
    
    # Test different regimes
    for regime in LatencyRegime:
        print(f"\n{regime.value.upper()} regime:")
        print("-" * 20)
        
        model = LatencyModel(regime=regime, seed=42)
        summary = model.get_latency_summary(venues, agents)
        
        # Show data latencies
        print("Data latencies (venue -> agent):")
        for venue in venues:
            for agent in agents:
                latency = summary['data'][venue][agent]
                print(f"  {venue} -> {agent}: {latency*1000000:.1f} μs")
        
        # Show order latencies
        print("\nOrder latencies (agent -> venue):")
        for agent in agents:
            for venue in venues:
                latency = summary['order'][agent][venue]
                print(f"  {agent} -> {venue}: {latency*1000000:.1f} μs")
        
        # Show IEX inbound delay effect
        print(f"\nIEX inbound delay effect:")
        for agent in agents:
            iex_latency = summary['order'][agent]['IEX']
            other_latency = summary['order'][agent]['A']
            delay_effect = (iex_latency - other_latency) * 1000000
            print(f"  {agent}: +{delay_effect:.1f} μs to IEX vs A")
    
    print("\nLatency model test complete!")
