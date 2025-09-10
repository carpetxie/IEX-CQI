#!/usr/bin/env python3
"""
CQS Models and Protection Logic
Implements Section 5 and 6 from the LaTeX specification
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from decimal import Decimal
from simulator import Simulator, Event

class SignalAgent:
    """
    Signal Agent that maintains global market state and calculates CQS features.
    
    This agent has a global view of all venues and calculates real-time features
    for the Crumbling Quote Signal models.
    """
    
    def __init__(self, simulator: Simulator, venues: List[str], 
                 history_length: int = 10, seed: int = 42, logger=None):
        """
        Initialize Signal Agent.
        
        Args:
            simulator: Simulation engine
            venues: List of venue names
            history_length: Number of historical states to maintain
            seed: Random seed
            logger: Event logger for recording events
        """
        self.simulator = simulator
        self.venues = venues
        self.history_length = history_length
        self.logger = logger
        
        # Global market state
        self.current_nbbo = {'bid': None, 'ask': None}
        self.venues_at_nbb = set()
        self.venues_at_nbo = set()
        
        # State history for lag features
        self.state_history = deque(maxlen=history_length)
        
        # Current features
        self.current_features = {
            'bids': 0,
            'asks': 0,
            'bids_lag1': 0,
            'asks_lag1': 0,
            'spread_bucket': 'unknown',
            'nbb': None,
            'nbo': None,
            'nbb_lag1': None,
            'nbo_lag1': None
        }
        
        # Statistics
        self.stats = {
            'states_processed': 0,
            'signal_firings': 0,
            'feature_calculations': 0
        }
        
        # Set random seed
        np.random.seed(seed)
        
        # Register event handlers - only handle market_data to avoid duplicates
        self.simulator.register_handler("market_data", self.handle_market_data)
    
    def handle_market_data(self, event):
        """Handle market data updates from venues."""
        data = event.data
        venue = data.get('venue')
        symbol = data.get('symbol')
        bbo = data.get('bbo', {})
        
        if not venue or not symbol or not bbo:
            return
        
        # Update venue state
        self._update_venue_state(venue, symbol, bbo)
        
        # Recalculate global state
        self._recalculate_global_state(symbol)
        
        # Calculate features for this state
        self._calculate_features(symbol)
        
        # Store state in history
        self._store_state()
        
        self.stats['states_processed'] += 1
    
    def handle_state_change(self, event):
        """Handle completed state changes."""
        data = event.data
        venue = data.get('venue')
        symbol = data.get('symbol')
        bbo = data.get('bbo', {})
        
        if not venue or not symbol or not bbo:
            return
        
        # Update venue state first
        self._update_venue_state(venue, symbol, bbo)
        
        # Recalculate global state
        self._recalculate_global_state(symbol)
        
        # Calculate features for this completed state
        self._calculate_features(symbol)
        
        # Store state in history
        self._store_state()
        
        self.stats['states_processed'] += 1
    
    def _update_venue_state(self, venue: str, symbol: str, bbo: Dict[str, Any]):
        """Update state for a specific venue."""
        # Store venue state for NBBO calculation
        if not hasattr(self, 'venue_states'):
            self.venue_states = {}
        
        if symbol not in self.venue_states:
            self.venue_states[symbol] = {}
        
        self.venue_states[symbol][venue] = {
            'bid_price': bbo.get('bid_price'),
            'bid_size': bbo.get('bid_size', 0),
            'ask_price': bbo.get('ask_price'),
            'ask_size': bbo.get('ask_size', 0)
        }
    
    def _recalculate_global_state(self, symbol: str):
        """Recalculate global NBBO and venue sets."""
        if not hasattr(self, 'venue_states') or symbol not in self.venue_states:
            return
        
        venue_states = self.venue_states[symbol]
        
        # Find best bid and ask across all venues
        best_bid_price = None
        best_ask_price = None
        venues_at_nbb = set()
        venues_at_nbo = set()
        
        for venue, state in venue_states.items():
            bid_price = state.get('bid_price')
            ask_price = state.get('ask_price')
            
            if bid_price is not None:
                if best_bid_price is None or bid_price > best_bid_price:
                    best_bid_price = bid_price
                    venues_at_nbb = {venue}
                elif bid_price == best_bid_price:
                    venues_at_nbb.add(venue)
            
            if ask_price is not None:
                if best_ask_price is None or ask_price < best_ask_price:
                    best_ask_price = ask_price
                    venues_at_nbo = {venue}
                elif ask_price == best_ask_price:
                    venues_at_nbo.add(venue)
        
        # Check for NBBO changes
        old_bid = self.current_nbbo.get('bid')
        old_ask = self.current_nbbo.get('ask')
        old_venues_at_bid = len(self.venues_at_nbb)
        old_venues_at_ask = len(self.venues_at_nbo)
        
        # Update global state
        self.current_nbbo = {
            'bid': best_bid_price,
            'ask': best_ask_price
        }
        self.venues_at_nbb = venues_at_nbb
        self.venues_at_nbo = venues_at_nbo
        
        # Check if NBBO changed
        new_bid = best_bid_price
        new_ask = best_ask_price
        new_venues_at_bid = len(venues_at_nbb)
        new_venues_at_ask = len(venues_at_nbo)
        
        # Log NBBO changes if we have a logger
        # Log both price AND venue count changes since both CQS and HFT target these events
        if hasattr(self, 'logger') and self.logger:
            if (old_bid != new_bid or old_ask != new_ask or 
                old_venues_at_bid != new_venues_at_bid or old_venues_at_ask != new_venues_at_ask):
                
                current_time = self.simulator.get_current_time()
                self.logger.log_nbbo_change(
                    current_time, symbol, old_bid, old_ask, new_bid, new_ask,
                    new_venues_at_bid, new_venues_at_ask
                )
        
        # Calculate features immediately when state changes
        self._calculate_features(symbol)
    
    def _calculate_features(self, symbol: str):
        """Calculate CQS features for the current state."""
        # Get current venue counts
        current_bids = len(self.venues_at_nbb)
        current_asks = len(self.venues_at_nbo)
        
        # Get lag features from history
        bids_lag1 = 0
        asks_lag1 = 0
        nbb_lag1 = None
        nbo_lag1 = None
        
        if len(self.state_history) > 0:
            last_state = self.state_history[-1]
            bids_lag1 = last_state.get('bids', 0)
            asks_lag1 = last_state.get('asks', 0)
            nbb_lag1 = last_state.get('nbb')
            nbo_lag1 = last_state.get('nbo')
        
        # Calculate spread bucket
        spread_bucket = self._calculate_spread_bucket()
        
        # Update current features
        self.current_features = {
            'bids': current_bids,
            'asks': current_asks,
            'bids_lag1': bids_lag1,
            'asks_lag1': asks_lag1,
            'spread_bucket': spread_bucket,
            'nbb': self.current_nbbo['bid'],
            'nbo': self.current_nbbo['ask'],
            'nbb_lag1': nbb_lag1,
            'nbo_lag1': nbo_lag1
        }
        
        self.stats['feature_calculations'] += 1
        
        # Evaluate CQS models if we have a CQS manager
        if hasattr(self, 'cqs_manager') and self.cqs_manager:
            current_time = self.simulator.get_current_time()
            # Only evaluate if we haven't already evaluated at this timestamp
            if not hasattr(self, '_last_eval_time') or self._last_eval_time != current_time:
                self._last_eval_time = current_time
                # Manager will handle deduplication and scheduling/logging
                _ = self.cqs_manager.evaluate_signal(self.current_features, current_time)
    
    def _calculate_spread_bucket(self) -> str:
        """Calculate spread bucket category."""
        if not self.current_nbbo['bid'] or not self.current_nbbo['ask']:
            return 'unknown'
        
        spread = float(self.current_nbbo['ask'] - self.current_nbbo['bid'])
        
        if spread <= 0.01:  # 1 tick
            return '1-tick'
        elif spread <= 0.02:  # 2 ticks
            return '2-tick'
        else:
            return '>2-tick'
    
    def _store_state(self):
        """Store current state in history."""
        state = {
            'timestamp': self.simulator.get_current_time(),
            'bids': self.current_features['bids'],
            'asks': self.current_features['asks'],
            'nbb': self.current_features['nbb'],
            'nbo': self.current_features['nbo'],
            'nbbo': self.current_nbbo.copy()
        }
        
        self.state_history.append(state)
    
    def get_features(self) -> Dict[str, Any]:
        """Get current features."""
        return self.current_features.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = self.stats.copy()
        stats['history_length'] = len(self.state_history)
        return stats

class CQSModel:
    """
    Base class for CQS models.
    """
    
    def __init__(self, name: str, simulator: Simulator):
        self.name = name
        self.simulator = simulator
        self.stats = {
            'evaluations': 0,
            'firings': 0,
            'total_protection_time': 0.0
        }
    
    def evaluate(self, features: Dict[str, Any], current_time: float) -> bool:
        """
        Evaluate whether the signal should fire.
        
        Args:
            features: Current market features
            current_time: Current simulation time
            
        Returns:
            True if signal should fire, False otherwise
        """
        self.stats['evaluations'] += 1
        return False
    
    def fire_signal(self, current_time: float, duration: float = 0.002):
        """Fire the CQS signal for the specified duration."""
        # Debounce: if manager indicates pending/on-window, skip duplicate fires
        manager = getattr(self, 'manager', None)
        if manager and (manager.is_signal_active or (manager._pending_activation and current_time <= manager._pending_until)):
            return
        
        self.stats['firings'] += 1
        self.stats['total_protection_time'] += duration
        
        # Schedule signal activation
        self.simulator.schedule_event_at(
            current_time,
            "signal_activation",
            {'model': self.name, 'duration': duration}
        )
        
        # Schedule signal deactivation
        self.simulator.schedule_event_at(
            current_time + duration,
            "signal_deactivation",
            {'model': self.name}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return self.stats.copy()

class ControlModel(CQSModel):
    """
    Model 1: Control (No Signal)
    
    No protection. Resting pegged and non-pegged orders are always available.
    This upper bounds HFT profit.
    """
    
    def __init__(self, simulator: Simulator):
        super().__init__("Control", simulator)
    
    def evaluate(self, features: Dict[str, Any], current_time: float) -> bool:
        """Control model never fires."""
        super().evaluate(features, current_time)
        return False

class HeuristicModel(CQSModel):
    """
    Model 2: Simple Heuristic Benchmark
    
    A static, rule-based signal that fires immediately when the number of venues
    at the NBBO decreases without a price change.
    """
    
    def __init__(self, simulator: Simulator):
        super().__init__("Heuristic", simulator)
        self.last_nbbo = {'bid': None, 'ask': None}
    
    def evaluate(self, features: Dict[str, Any], current_time: float) -> bool:
        """Evaluate heuristic firing condition."""
        super().evaluate(features, current_time)
        
        # Check if venues decreased AND price unchanged (per LaTeX Section 6.2)
        bids_decreased = features['bids'] < features['bids_lag1']
        asks_decreased = features['asks'] < features['asks_lag1']
        
        # Check if prices are unchanged
        bid_price_unchanged = (features.get('nbb') == features.get('nbb_lag1') and 
                              features.get('nbb') is not None and 
                              features.get('nbb_lag1') is not None)
        ask_price_unchanged = (features.get('nbo') == features.get('nbo_lag1') and 
                              features.get('nbo') is not None and 
                              features.get('nbo_lag1') is not None)
        
        # Fire if venues decreased without price change on either side
        should_fire = ((bids_decreased and bid_price_unchanged) or 
                      (asks_decreased and ask_price_unchanged))
        
        return should_fire

class LogisticModel(CQSModel):
    """
    Model 3: Logistic Risk-Score CQS
    
    A state-dependent risk score that anticipates imminent ticks using compact features.
    """
    
    def __init__(self, simulator: Simulator, threshold: float = 0.5, 
                 coefficients: Optional[Dict[str, float]] = None):
        super().__init__("Logistic", simulator)
        self.threshold = threshold
        
        # Default coefficients (fire when many venues at NBBO)
        self.coefficients = coefficients or {
            'c0': -2.0,     # Intercept (negative base)
            'c1': 1.0,      # bids coefficient (positive - more venues = higher risk)
            'c2': 1.0,      # asks coefficient (positive - more venues = higher risk)
            'c3': -0.5,     # bids_lag1 coefficient (negative - fewer lag venues = higher risk)
            'c4': -0.5      # asks_lag1 coefficient (negative - fewer lag venues = higher risk)
        }
    
    def evaluate(self, features: Dict[str, Any], current_time: float) -> bool:
        """Evaluate logistic risk score."""
        super().evaluate(features, current_time)
        
        # Calculate risk score: p(t) = σ(z)
        z = (self.coefficients['c0'] + 
             self.coefficients['c1'] * features['bids'] +
             self.coefficients['c2'] * features['asks'] +
             self.coefficients['c3'] * features['bids_lag1'] +
             self.coefficients['c4'] * features['asks_lag1'])
        
        # Sigmoid function
        p = 1.0 / (1.0 + np.exp(-z))
        
        # Fire if risk score exceeds threshold
        should_fire = p > self.threshold
        
        return should_fire
    
    def get_risk_score(self, features: Dict[str, Any]) -> float:
        """Get the current risk score without firing."""
        z = (self.coefficients['c0'] + 
             self.coefficients['c1'] * features['bids'] +
             self.coefficients['c2'] * features['asks'] +
             self.coefficients['c3'] * features['bids_lag1'] +
             self.coefficients['c4'] * features['asks_lag1'])
        
        return 1.0 / (1.0 + np.exp(-z))

class CQSManager:
    """
    Manages CQS models and coordinates signal firing.
    """
    
    def __init__(self, simulator: Simulator, venues: List[str]):
        self.simulator = simulator
        self.venues = venues
        
        # Create models
        self.models = {
            'control': ControlModel(simulator),
            'heuristic': HeuristicModel(simulator),
            'logistic': LogisticModel(simulator)
        }
        # Give models a back-reference to the manager for deduplication
        for m in self.models.values():
            setattr(m, 'manager', self)
        
        # Current active model
        self.active_model = 'control'
        
        # Signal state
        self.is_signal_active = False
        self.signal_start_time = 0.0
        # Pending activation window to debounce multiple firings at same time
        self._pending_activation = False
        self._pending_until = 0.0
        
        # Statistics
        self.stats = {
            'model_evaluations': 0,
            'signal_activations': 0,
            'signal_deactivations': 0,
            'firings': 0,
            'total_protection_time': 0.0
        }
        
        # Register event handlers
        self.simulator.register_handler("signal_activation", self.handle_signal_activation)
        self.simulator.register_handler("signal_deactivation", self.handle_signal_deactivation)
    
    def set_active_model(self, model_name: str):
        """Set the active CQS model."""
        if model_name in self.models:
            self.active_model = model_name
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def evaluate_signal(self, features: Dict[str, Any], current_time: float) -> bool:
        """Evaluate whether the signal should fire."""
        model = self.models[self.active_model]
        should_fire = model.evaluate(features, current_time)
        
        if should_fire:
            # Only fire the signal if not already active or pending
            # This prevents duplicate logging while debouncing activation
            will_actually_fire = not (self.is_signal_active or (self._pending_activation and current_time <= self._pending_until))
            
            if will_actually_fire:
                # Log the fire only when signal will actually activate
                if hasattr(self, 'logger') and self.logger:
                    risk_score = None
                    if self.active_model == 'logistic':
                        risk_score = self.models['logistic'].get_risk_score(features)
                    self.logger.log_cqs_fire(current_time, self.active_model, features, risk_score)
            
            # Fire the signal (this handles activation/deactivation with its own debouncing)
            self.fire_signal(current_time)
        
        self.stats['model_evaluations'] += 1
        return should_fire
    
    def fire_signal(self, current_time: float, duration: float = 0.002):
        """Fire the CQS signal for the specified duration."""
        # Debounce: if already active or pending, skip
        if self.is_signal_active or (self._pending_activation and current_time <= self._pending_until):
            return
        
        self.stats['firings'] += 1
        self.stats['total_protection_time'] += duration
        
        # Schedule signal activation
        self.simulator.schedule_event_at(
            current_time,
            "signal_activation",
            {'model': self.active_model, 'duration': duration}
        )
        
        # Schedule signal deactivation
        self.simulator.schedule_event_at(
            current_time + duration,
            "signal_deactivation",
            {'model': self.active_model}
        )
    
    def handle_signal_activation(self, event):
        """Handle signal activation event."""
        self.is_signal_active = True
        self.signal_start_time = event.timestamp
        self.stats['signal_activations'] += 1
        # Mark pending during the on-window
        self._pending_activation = True
        self._pending_until = event.data.get('duration', 0.002) + event.timestamp
        
        print(f"  CQS Signal ACTIVATED at {event.timestamp:.6f}s by {event.data['model']}")
    
    def handle_signal_deactivation(self, event):
        """Handle signal deactivation event."""
        if self.is_signal_active:
            duration = event.timestamp - self.signal_start_time
            self.stats['total_protection_time'] += duration
            self.stats['signal_deactivations'] += 1
        
        self.is_signal_active = False
        self._pending_activation = False
        
        print(f"  CQS Signal DEACTIVATED at {event.timestamp:.6f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        stats = self.stats.copy()
        stats['active_model'] = self.active_model
        stats['is_signal_active'] = self.is_signal_active
        
        # Add model-specific stats
        for name, model in self.models.items():
            stats[f'model_{name}'] = model.get_stats()
        
        return stats

if __name__ == "__main__":
    # Test the CQS models
    print("Testing CQS Models")
    print("=" * 30)
    
    from simulator import Simulator
    
    # Create test environment
    sim = Simulator(max_time=1.0)
    venues = ["IEX", "A", "B"]
    
    # Test Signal Agent
    print("\n1. Testing Signal Agent:")
    print("-" * 25)
    
    signal_agent = SignalAgent(sim, venues)
    
    # Test feature calculation
    features = signal_agent.get_features()
    print(f"Initial features: {features}")
    
    # Test CQS Models
    print("\n2. Testing CQS Models:")
    print("-" * 25)
    
    cqs_manager = CQSManager(sim, venues)
    
    # Test control model
    cqs_manager.set_active_model('control')
    test_features = {'bids': 3, 'asks': 2, 'bids_lag1': 4, 'asks_lag1': 3, 'spread_bucket': '1-tick'}
    should_fire = cqs_manager.evaluate_signal(test_features, 0.1)
    print(f"Control model should fire: {should_fire}")
    
    # Test heuristic model
    cqs_manager.set_active_model('heuristic')
    should_fire = cqs_manager.evaluate_signal(test_features, 0.2)
    print(f"Heuristic model should fire: {should_fire}")
    
    # Test logistic model
    cqs_manager.set_active_model('logistic')
    should_fire = cqs_manager.evaluate_signal(test_features, 0.3)
    print(f"Logistic model should fire: {should_fire}")
    
    # Test risk score calculation
    logistic_model = cqs_manager.models['logistic']
    risk_score = logistic_model.get_risk_score(test_features)
    print(f"Logistic risk score: {risk_score:.4f}")
    
    print(f"\nCQS Manager stats: {cqs_manager.get_stats()}")
    
    print("\nCQS models test complete!")
