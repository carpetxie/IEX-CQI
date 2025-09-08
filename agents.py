#!/usr/bin/env python3
"""
Agent Classes for the Simulation
Implements Section 3.3 and 3.4 from the LaTeX specification
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional
from decimal import Decimal
from simulator import Simulator
from market_environment import Exchange
from latency_model import LatencyModel

class LPAgent:
    """
    Liquidity Provider (LP) Agent.
    
    Passive agent that replenishes the best side when triggered by Poisson events.
    Behavior: Passively replenish the best side. Arrival times follow an inhomogeneous 
    Poisson process with rate λ_add(t). Order size is drawn from f_add(s) (empirical).
    """
    
    def __init__(self, agent_id: str, simulator: Simulator, 
                 exchanges: List[Exchange], latency_model: LatencyModel,
                 size_distribution: List[int], seed: int = 42):
        """
        Initialize LP agent.
        
        Args:
            agent_id: Unique identifier for this agent
            simulator: Simulation engine
            exchanges: List of available exchanges
            latency_model: Latency model for order routing
            size_distribution: List of sizes for sampling
            seed: Random seed
        """
        self.agent_id = agent_id
        self.simulator = simulator
        self.exchanges = exchanges
        self.latency_model = latency_model
        self.size_distribution = size_distribution
        
        # Agent state
        self.current_time = 0.0
        self.market_views = {}  # symbol -> latest BBO data
        
        # Statistics
        self.stats = {
            'orders_sent': 0,
            'total_volume': 0,
            'symbols_traded': set()
        }
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Register event handlers
        self.simulator.register_handler("addition_event", self.handle_addition_event)
        self.simulator.register_handler("market_data", self.handle_market_data)
    
    def handle_addition_event(self, event):
        """Handle liquidity addition events."""
        # LP agents respond to addition events by adding liquidity
        # Draw size from distribution
        size = random.choice(self.size_distribution) if self.size_distribution else 100
        
        # Select a random symbol to add liquidity to
        symbols = list(self.market_views.keys())
        if not symbols:
            # If no market views, try to add liquidity to any available symbol
            # This ensures LP agents can start adding liquidity even without market data
            # For now, use a default symbol
            symbol = "AAPL"  # Default symbol
        else:
            symbol = random.choice(symbols)
        
        self._add_liquidity(symbol, size)
    
    def handle_market_data(self, event):
        """Handle market data updates."""
        data = event.data
        
        # Only process data intended for this agent or no specific recipient
        recipient = data.get('recipient')
        if recipient and recipient != self.agent_id:
            return
        
        symbol = data.get('symbol')
        
        if symbol:
            self.market_views[symbol] = {
                'bid_price': data.get('bbo', {}).get('bid_price'),
                'bid_size': data.get('bbo', {}).get('bid_size', 0),
                'ask_price': data.get('bbo', {}).get('ask_price'),
                'ask_size': data.get('bbo', {}).get('ask_size', 0),
                'timestamp': event.timestamp
            }
    
    def _add_liquidity(self, symbol: str, size: int):
        """Add liquidity for a specific symbol."""
        if symbol not in self.market_views:
            return
        
        market_data = self.market_views[symbol]
        bid_price = market_data.get('bid_price')
        ask_price = market_data.get('ask_price')
        
        if not bid_price or not ask_price:
            return
        
        # Determine which side to add liquidity to
        # For simplicity, randomly choose bid or ask
        side = random.choice(['B', 'A'])
        
        if side == 'B':
            # Add bid liquidity at or below current best bid
            price = bid_price - random.uniform(0, 0.01)  # Slightly below best bid
        else:
            # Add ask liquidity at or above current best ask
            price = ask_price + random.uniform(0, 0.01)  # Slightly above best ask
        
        # Select target exchange (prefer IEX for now)
        target_exchange = "IEX"
        
        # Calculate order arrival time with latency
        latency = self.latency_model.get_order_latency(self.agent_id, target_exchange)
        arrival_time = self.simulator.get_current_time() + latency
        
        # Create order
        order = {
            'action': 'add',
            'symbol': symbol,
            'side': side,
            'price': float(price),
            'size': size,
            'is_pegged': random.random() < 0.5,  # 50% chance of being pegged
            'venue': target_exchange
        }
        
        # Schedule order event
        self.simulator.schedule_event_at(
            arrival_time,
            "order",
            order
        )
        
        # Update statistics
        self.stats['orders_sent'] += 1
        self.stats['total_volume'] += size
        self.stats['symbols_traded'].add(symbol)
        
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = self.stats.copy()
        stats['symbols_traded'] = list(stats['symbols_traded'])
        return stats

class HFTAgent:
    """
    High-Frequency Trader (HFT) Agent.
    
    Deterministic agent that reacts to vulnerable market states.
    Behavior: Continuously maintains state features and fires when vulnerable state detected.
    """
    
    def __init__(self, agent_id: str, simulator: Simulator,
                 exchanges: List[Exchange], latency_model: LatencyModel,
                 reaction_delay: float = 0.0001,  # 100 microseconds
                 max_fill_size: int = 1000, seed: int = 42, logger=None):
        """
        Initialize HFT agent.
        
        Args:
            agent_id: Unique identifier for this agent
            simulator: Simulation engine
            exchanges: List of available exchanges
            latency_model: Latency model for order routing
            reaction_delay: Time delay before reacting to market data
            max_fill_size: Maximum order size
            seed: Random seed
        """
        self.agent_id = agent_id
        self.simulator = simulator
        self.exchanges = exchanges
        self.latency_model = latency_model
        self.reaction_delay = reaction_delay
        self.max_fill_size = max_fill_size
        self.logger = logger
        
        # Agent state
        self.current_time = 0.0
        self.market_views = {}  # symbol -> latest BBO data from each venue
        self.state_features = {}  # symbol -> feature history
        
        # Statistics
        self.stats = {
            'arbitrage_attempts': 0,
            'successful_arbitrages': 0,
            'blocked_attempts': 0,
            'total_pnl': 0.0,
            'symbols_traded': set()
        }
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Register event handlers
        self.simulator.register_handler("market_data", self.handle_market_data)
        self.simulator.register_handler("depletion_event", self.handle_depletion_event)
    
    def handle_market_data(self, event):
        """Handle market data updates."""
        data = event.data
        
        # Only process data intended for this agent or no specific recipient
        recipient = data.get('recipient')
        if recipient and recipient != self.agent_id:
            return
        
        symbol = data.get('symbol')
        venue = data.get('venue')
        bbo = data.get('bbo', {})
        
        if not symbol or not venue:
            return
        
        # Update market view
        if symbol not in self.market_views:
            self.market_views[symbol] = {}
        
        self.market_views[symbol][venue] = {
            'bid_price': bbo.get('bid_price'),
            'bid_size': bbo.get('bid_size', 0),
            'ask_price': bbo.get('ask_price'),
            'ask_size': bbo.get('ask_size', 0),
            'timestamp': event.timestamp
        }
        
        
        # Update state features
        self._update_state_features(symbol)
        
        # Check for vulnerable state
        if self._is_vulnerable_state(symbol):
            self._execute_arbitrage(symbol)
    
    def handle_depletion_event(self, event):
        """Handle depletion events (near-side pressure)."""
        data = event.data
        side = data.get('side')
        target_exchange = data.get('target_exchange')
        
        # HFT agents can react to depletion events by checking for arbitrage opportunities
        # This creates additional market stress
        if hasattr(self, 'market_views') and self.market_views:
            # Check all symbols for vulnerable states
            for symbol in self.market_views.keys():
                if self._is_vulnerable_state(symbol):
                    self._execute_arbitrage(symbol)
    
    def _update_state_features(self, symbol: str):
        """Update state features for a symbol."""
        if symbol not in self.market_views:
            return
        
        # Calculate current features
        features = self._calculate_features(symbol)
        
        # Store feature history (keep last 10 states)
        if symbol not in self.state_features:
            self.state_features[symbol] = []
        
        self.state_features[symbol].append(features)
        
        # Keep only last 10 states
        if len(self.state_features[symbol]) > 10:
            self.state_features[symbol] = self.state_features[symbol][-10:]
    
    def _calculate_features(self, symbol: str) -> Dict[str, Any]:
        """Calculate state features for a symbol."""
        if symbol not in self.market_views:
            return {}
        
        venue_data = self.market_views[symbol]
        
        # Count venues at NBB and NBO
        bid_prices = [data['bid_price'] for data in venue_data.values() 
                     if data['bid_price'] is not None]
        ask_prices = [data['ask_price'] for data in venue_data.values() 
                     if data['ask_price'] is not None]
        
        if not bid_prices or not ask_prices:
            return {'bids': 0, 'asks': 0, 'bids_lag1': 0, 'asks_lag1': 0}
        
        nbb = max(bid_prices)
        nbo = min(ask_prices)
        
        # Count venues at NBB and NBO
        bids_at_nbb = sum(1 for data in venue_data.values() 
                         if data['bid_price'] == nbb)
        asks_at_nbo = sum(1 for data in venue_data.values() 
                         if data['ask_price'] == nbo)
        
        # Get lag features
        bids_lag1 = 0
        asks_lag1 = 0
        if symbol in self.state_features and len(self.state_features[symbol]) > 0:
            last_features = self.state_features[symbol][-1]
            bids_lag1 = last_features.get('bids', 0)
            asks_lag1 = last_features.get('asks', 0)
        
        return {
            'bids': bids_at_nbb,
            'asks': asks_at_nbo,
            'bids_lag1': bids_lag1,
            'asks_lag1': asks_lag1,
            'nbb': nbb,
            'nbo': nbo
        }
    
    def _is_vulnerable_state(self, symbol: str) -> bool:
        """
        Check if the market is in a vulnerable state.
        
        Simple heuristic: if number of venues at NBBO decreases without price change,
        the market is vulnerable.
        """
        if symbol not in self.state_features or len(self.state_features[symbol]) < 2:
            return False
        
        current = self.state_features[symbol][-1]
        previous = self.state_features[symbol][-2]
        
        # Check if venues decreased without price change
        bids_decreased = current['bids'] < previous['bids']
        asks_decreased = current['asks'] < previous['asks']
        # Handle None values for NBBO (when venues disappear)
        current_nbb = current.get('nbb')
        previous_nbb = previous.get('nbb')
        current_nbo = current.get('nbo')
        previous_nbo = previous.get('nbo')
        
        bid_price_same = current_nbb == previous_nbb if (current_nbb is not None and previous_nbb is not None) else False
        ask_price_same = current_nbo == previous_nbo if (current_nbo is not None and previous_nbo is not None) else False
        
        # Per LaTeX heuristic: fire only when venues drop WITH price unchanged
        return (bids_decreased and bid_price_same) or (asks_decreased and ask_price_same)
    
    def _execute_arbitrage(self, symbol: str):
        """Execute arbitrage when vulnerable state detected."""
        if symbol not in self.state_features:
            return
        
        current_features = self.state_features[symbol][-1]
        
        # Find the stale venue (one with outdated price)
        stale_venue = self._find_stale_venue(symbol, current_features)
        if not stale_venue:
            return
        
        # Determine arbitrage direction
        if current_features['bids'] < current_features['bids_lag1']:
            # Bid side decreased - buy at stale venue, sell at true price
            side = 'B'
            target_price = current_features['nbb']
        else:
            # Ask side decreased - sell at stale venue, buy at true price
            side = 'A'
            target_price = current_features['nbo']
        
        # Calculate order size
        visible_size = self._get_visible_size(symbol, stale_venue, side, target_price)
        order_size = min(visible_size, self.max_fill_size)
        
        if order_size <= 0:
            # If no visible size, try a small order anyway
            order_size = min(100, self.max_fill_size)
        
        # Calculate arrival time with reaction delay and latency
        reaction_time = self.simulator.get_current_time() + self.reaction_delay
        latency = self.latency_model.get_order_latency(self.agent_id, stale_venue)
        arrival_time = reaction_time + latency
        
        # Create arbitrage order
        order = {
            'action': 'match',
            'symbol': symbol,
            'side': side,
            'price': target_price,
            'size': order_size,
            'is_arbitrage': True,
            'venue': stale_venue,
            'agent_id': self.agent_id
        }
        
        # Schedule order event
        self.simulator.schedule_event_at(
            arrival_time,
            "order",
            order
        )
        
        # Update statistics
        self.stats['arbitrage_attempts'] += 1
        self.stats['symbols_traded'].add(symbol)
        
        # Calculate potential P&L for this arbitrage
        # HFT profits from price differences when no protection is active
        if side == 'B':
            # Buying at stale bid, will sell at higher ask
            true_ask = current_features.get('nbo')
            if true_ask and target_price < true_ask:
                potential_pnl = (true_ask - target_price) * order_size
            else:
                potential_pnl = 0.0
        else:
            # Selling at stale ask, will buy at lower bid  
            true_bid = current_features.get('nbb')
            if true_bid and target_price > true_bid:
                potential_pnl = (target_price - true_bid) * order_size
            else:
                potential_pnl = 0.0
        
        # Result (success/blocked) will be logged by the exchange upon execution
        
    
    def _find_stale_venue(self, symbol: str, features: Dict[str, Any]) -> Optional[str]:
        """Find the venue with stale prices."""
        if symbol not in self.market_views:
            return None
        
        venue_data = self.market_views[symbol]
        nbb = features.get('nbb')
        nbo = features.get('nbo')
        
        # Check if NBBO prices are available
        if nbb is None and nbo is None:
            return None
        
        # Find venue that still has old prices
        for venue, data in venue_data.items():
            if ((nbb is not None and data['bid_price'] == nbb and data['bid_size'] > 0) or 
                (nbo is not None and data['ask_price'] == nbo and data['ask_size'] > 0)):
                return venue
        
        return None
    
    def _get_visible_size(self, symbol: str, venue: str, side: str, price: float) -> int:
        """Get visible size at a venue for a given side and price."""
        if symbol not in self.market_views or venue not in self.market_views[symbol]:
            return 0
        
        data = self.market_views[symbol][venue]
        if side == 'B':
            return data['bid_size'] if data['bid_price'] == price else 0
        else:
            return data['ask_size'] if data['ask_price'] == price else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = self.stats.copy()
        stats['symbols_traded'] = list(stats['symbols_traded'])
        return stats

if __name__ == "__main__":
    # Test the agents
    print("Testing Agents")
    print("=" * 30)
    
    from simulator import Simulator
    from market_environment import Exchange
    from latency_model import LatencyModel, LatencyRegime
    
    # Create test environment
    sim = Simulator(max_time=5.0)
    iex = Exchange("IEX", sim)
    iex.add_symbol("AAPL", Decimal('0.01'))
    
    latency_model = LatencyModel(regime=LatencyRegime.MEDIUM, seed=42)
    
    # Test LP Agent
    print("\n1. Testing LP Agent:")
    print("-" * 20)
    
    size_distribution = [100, 200, 500, 1000]
    lp_agent = LPAgent("LP1", sim, [iex], latency_model, size_distribution)
    
    # Simulate some market data
    sim.schedule_event_at(0.1, "market_data", {
        'symbol': 'AAPL',
        'venue': 'IEX',
        'bbo': {'bid_price': 150.0, 'bid_size': 1000, 'ask_price': 150.01, 'ask_size': 800}
    })
    
    # Simulate addition event
    sim.schedule_event_at(0.2, "addition_event", {
        'target_exchange': 'LP1',
        'size': 200
    })
    
    print(f"LP Agent stats: {lp_agent.get_stats()}")
    
    # Test HFT Agent
    print("\n2. Testing HFT Agent:")
    print("-" * 20)
    
    hft_agent = HFTAgent("HFT1", sim, [iex], latency_model)
    
    # Simulate market data updates
    sim.schedule_event_at(0.1, "market_data", {
        'symbol': 'AAPL',
        'venue': 'IEX',
        'bbo': {'bid_price': 150.0, 'bid_size': 1000, 'ask_price': 150.01, 'ask_size': 800}
    })
    
    sim.schedule_event_at(0.2, "market_data", {
        'symbol': 'AAPL',
        'venue': 'IEX',
        'bbo': {'bid_price': 150.0, 'bid_size': 500, 'ask_price': 150.01, 'ask_size': 800}
    })
    
    print(f"HFT Agent stats: {hft_agent.get_stats()}")
    
    print("\nAgent test complete!")
