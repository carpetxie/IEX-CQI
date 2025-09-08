#!/usr/bin/env python3
"""
Market Environment Components
Implements Section 2 from the LaTeX specification
"""

import numpy as np
from decimal import Decimal
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from orderbook import OrderBook
from simulator import Simulator, MarketDataEvent, OrderEvent

class LimitOrderBook:
    """
    Enhanced Limit Order Book for the simulation.
    
    Extends the basic OrderBook with additional features needed for the simulation:
    - Pegged vs non-pegged buckets
    - Signal protection
    - Order matching with protection logic
    """
    
    def __init__(self, symbol: str, tick_size: Decimal = Decimal('0.01')):
        self.symbol = symbol
        self.tick_size = tick_size
        
        # Price levels: {price: {'total_size': int, 'pegged_size': int, 'non_pegged_size': int}}
        self.bids = {}  # price -> size info
        self.asks = {}  # price -> size info
        
        # Track best prices
        self.best_bid_price = None
        self.best_ask_price = None
        
        # Signal protection state
        self.is_signal_active = False
        self.pegged_fraction = 0.5  # Default: 50% of depth is pegged
    
    def set_pegged_fraction(self, fraction: float):
        """Set the fraction of depth that is pegged (0.0 to 1.0)."""
        self.pegged_fraction = max(0.0, min(1.0, fraction))
    
    def set_signal_active(self, active: bool):
        """Set the Crumbling Quote Signal state."""
        self.is_signal_active = active
    
    def add_order(self, side: str, price: Decimal, size: int, is_pegged: bool = False) -> None:
        """
        Add an order to the book.
        
        Args:
            side: 'B' for bid, 'A' for ask
            price: Order price
            size: Order size (positive for additions, negative for cancellations)
            is_pegged: Whether this order is pegged
        """
        if side == 'B':
            if price in self.bids:
                self.bids[price]['total_size'] += size
                if is_pegged:
                    self.bids[price]['pegged_size'] += size
                else:
                    self.bids[price]['non_pegged_size'] += size
            else:
                pegged_size = size if is_pegged else 0
                non_pegged_size = size - pegged_size
                self.bids[price] = {
                    'total_size': size,
                    'pegged_size': pegged_size,
                    'non_pegged_size': non_pegged_size
                }
            
            # Remove empty price levels
            if self.bids[price]['total_size'] <= 0:
                del self.bids[price]
            
            # Update best bid
            if self.bids:
                self.best_bid_price = max(self.bids.keys())
            else:
                self.best_bid_price = None
                
        elif side == 'A':
            if price in self.asks:
                self.asks[price]['total_size'] += size
                if is_pegged:
                    self.asks[price]['pegged_size'] += size
                else:
                    self.asks[price]['non_pegged_size'] += size
            else:
                pegged_size = size if is_pegged else 0
                non_pegged_size = size - pegged_size
                self.asks[price] = {
                    'total_size': size,
                    'pegged_size': pegged_size,
                    'non_pegged_size': non_pegged_size
                }
            
            # Remove empty price levels
            if self.asks[price]['total_size'] <= 0:
                del self.asks[price]
            
            # Update best ask
            if self.asks:
                self.best_ask_price = min(self.asks.keys())
            else:
                self.best_ask_price = None
    
    def _update_best_bid(self):
        """Update the best bid price after order book changes."""
        if self.bids:
            self.best_bid_price = max(self.bids.keys())
        else:
            self.best_bid_price = None
    
    def _update_best_ask(self):
        """Update the best ask price after order book changes."""
        if self.asks:
            self.best_ask_price = min(self.asks.keys())
        else:
            self.best_ask_price = None
    
    def get_best_bid_offer(self) -> Tuple[Optional[Decimal], int, Optional[Decimal], int]:
        """
        Get the Best Bid and Offer (BBO).
        
        Returns:
            Tuple of (best_bid_price, best_bid_size, best_ask_price, best_ask_size)
        """
        best_bid_size = self.bids.get(self.best_bid_price, {}).get('total_size', 0) if self.best_bid_price else 0
        best_ask_size = self.asks.get(self.best_ask_price, {}).get('total_size', 0) if self.best_ask_price else 0
        
        return (self.best_bid_price, best_bid_size, 
                self.best_ask_price, best_ask_size)
    
    def get_available_size(self, side: str, price: Decimal, is_signal_active: bool = False) -> int:
        """
        Get available size for matching at a specific price.
        
        If signal is active, only non-pegged size is available.
        """
        if side == 'B' and price in self.bids:
            size_info = self.bids[price]
            if self.is_signal_active:
                return size_info['non_pegged_size']
            else:
                return size_info['total_size']
        elif side == 'A' and price in self.asks:
            size_info = self.asks[price]
            if self.is_signal_active:
                return size_info['non_pegged_size']
            else:
                return size_info['total_size']
        return 0
    
    def match_order(self, side: str, price: Decimal, size: int, is_signal_active: bool = False) -> Tuple[int, List[Dict]]:
        """
        Match an incoming order against the book.
        
        Args:
            side: 'B' for bid (buy), 'A' for ask (sell)
            price: Order price
            size: Order size
            
        Returns:
            Tuple of (filled_size, list_of_fills)
        """
        fills = []
        remaining_size = size
        
        if side == 'B':  # Buy order - match against asks
            if not self.asks or not self.best_ask_price:
                return 0, fills
            
            # Check if we can cross the spread
            if price >= self.best_ask_price:
                # Match at best ask price
                available_size = self.get_available_size('A', self.best_ask_price, is_signal_active)
                fill_size = min(remaining_size, available_size)
                
                if fill_size > 0:
                    fills.append({
                        'price': self.best_ask_price,
                        'size': fill_size,
                        'side': 'A'
                    })
                    
                    # Update book
                    self.add_order('A', self.best_ask_price, -fill_size)
                    remaining_size -= fill_size
        
        elif side == 'A':  # Sell order - match against bids
            if not self.bids or not self.best_bid_price:
                return 0, fills
            
            # Check if we can cross the spread
            if price <= self.best_bid_price:
                # Match at best bid price
                available_size = self.get_available_size('B', self.best_bid_price, is_signal_active)
                fill_size = min(remaining_size, available_size)
                
                if fill_size > 0:
                    fills.append({
                        'price': self.best_bid_price,
                        'size': fill_size,
                        'side': 'B'
                    })
                    
                    # Update book
                    self.add_order('B', self.best_bid_price, -fill_size)
                    remaining_size -= fill_size
        
        filled_size = size - remaining_size
        return filled_size, fills
    
    def clear(self) -> None:
        """Clear the entire order book."""
        self.bids.clear()
        self.asks.clear()
        self.best_bid_price = None
        self.best_ask_price = None
        self.is_signal_active = False

class Exchange:
    """
    Exchange class representing a trading venue.
    
    Each exchange maintains a limit order book and handles order processing.
    """
    
    def __init__(self, name: str, simulator: Simulator, latency_model=None, agents=None):
        self.name = name
        self.simulator = simulator
        self.latency_model = latency_model
        self.agents = agents or []  # List of agent IDs to deliver data to
        self.order_books = {}  # symbol -> LimitOrderBook
        self.tick_size = Decimal('0.01')  # Default tick size
        
        # Register event handlers
        self.simulator.register_handler("order", self.handle_order)
        self.simulator.register_handler("signal", self.handle_signal)
        self.simulator.register_handler("signal_activation", self.handle_signal_activation)
        self.simulator.register_handler("signal_deactivation", self.handle_signal_deactivation)
        self.simulator.register_handler("depletion_event", self.handle_depletion_event)
        self.simulator.register_handler("addition_event", self.handle_addition_event)
        
        # print(f"Exchange {self.name} registered handlers for depletion_event and addition_event")
    
    def add_symbol(self, symbol: str, tick_size: Decimal = Decimal('0.01')):
        """Add a new symbol to this exchange."""
        self.order_books[symbol] = LimitOrderBook(symbol, tick_size)
        self.tick_size = tick_size
        
        # Initialize with some starting liquidity
        self._initialize_liquidity(symbol)
    
    def _initialize_liquidity(self, symbol: str):
        """Initialize order book with starting liquidity."""
        order_book = self.order_books[symbol]
        
        # Add some initial orders to create a market
        # Use smaller sizes for more realistic market dynamics
        base_price = Decimal('150.00')  # Starting price
        
        # Add bid orders (very small sizes for rapid crumbling)
        order_book.add_order('B', base_price, 100, is_pegged=False)
        order_book.add_order('B', base_price - Decimal('0.01'), 80, is_pegged=True)
        order_book.add_order('B', base_price - Decimal('0.02'), 60, is_pegged=False)
        order_book.add_order('B', base_price - Decimal('0.03'), 40, is_pegged=True)
        
        # Add ask orders (very small sizes for rapid crumbling)
        order_book.add_order('A', base_price + Decimal('0.01'), 90, is_pegged=False)
        order_book.add_order('A', base_price + Decimal('0.02'), 70, is_pegged=True)
        order_book.add_order('A', base_price + Decimal('0.03'), 50, is_pegged=False)
        order_book.add_order('A', base_price + Decimal('0.04'), 30, is_pegged=True)
        
        # Publish initial market data
        self.publish_market_data({
            'type': 'book_update',
            'symbol': symbol,
            'venue': self.name,
            'bbo': self.get_bbo(symbol)
        })
    
    def get_order_book(self, symbol: str) -> Optional[LimitOrderBook]:
        """Get the order book for a symbol."""
        return self.order_books.get(symbol)
    
    def handle_order(self, event: OrderEvent):
        """Handle incoming order events."""
        if hasattr(event, 'venue') and event.venue != self.name:
            return
        
        order = event.data
        symbol = order.get('symbol')
        
        if symbol not in self.order_books:
            print(f"Warning: Symbol {symbol} not found on {self.name}")
            return
        
        order_book = self.order_books[symbol]
        
        # Process the order
        if order.get('action') == 'add':
            # Add order to book
            side = order.get('side')
            price = Decimal(str(order.get('price', 0)))
            size = order.get('size', 0)
            is_pegged = order.get('is_pegged', False)
            
            order_book.add_order(side, price, size, is_pegged)
            
        elif order.get('action') == 'match':
            # Match order against book
            side = order.get('side')
            price = Decimal(str(order.get('price', 0)))
            size = order.get('size', 0)
            agent_id = order.get('agent_id')
            
            # Enforce protection only on IEX
            is_signal_active = False
            if self.name == "IEX":
                is_signal_active = any(ob.is_signal_active for ob in self.order_books.values())
            
            filled_size, fills = order_book.match_order(side, price, size, is_signal_active=is_signal_active)
            
            # Publish market data events for fills
            for fill in fills:
                self.publish_market_data({
                    'type': 'trade',
                    'symbol': symbol,
                    'price': float(fill['price']),
                    'size': fill['size'],
                    'side': fill['side']
                })
            
            # If this was an HFT arbitrage order, log success/blocked
            if order.get('is_arbitrage') and agent_id:
                success = filled_size > 0
                pnl = 0.0
                event_data = {
                    'agent_id': agent_id,
                    'symbol': symbol,
                    'side': side,
                    'price': float(price),
                    'size': filled_size if success else size,
                    'success': success,
                    'pnl': pnl
                }
                self.simulator.schedule_event_at(
                    self.simulator.get_current_time(),
                    "hft_arbitrage",
                    event_data
                )
        
        # Publish market data event for book update
        self.publish_market_data({
            'type': 'book_update',
            'symbol': symbol,
            'venue': self.name,
            'bbo': self.get_bbo(symbol)
        })
    
    def handle_depletion_event(self, event):
        """Handle depletion events (near-side pressure)."""
        data = event.data
        side = data.get('side')
        target_exchange = data.get('target_exchange')
        
        # print(f"    {self.name} handling depletion_event for {target_exchange}")
        
        if target_exchange != self.name:
            # print(f"    {self.name} skipping - not target")
            return
        
        # Apply depletion to a random symbol
        if not self.order_books:
            return
        
        symbol = list(self.order_books.keys())[0]  # Use first symbol
        order_book = self.order_books[symbol]
        
        # Get current BBO
        bbo = order_book.get_best_bid_offer()
        if not bbo[0] or not bbo[2]:  # No valid BBO
            return
        
        # print(f"  DEPLETION: {self.name} {side} side, BBO before: {bbo}")
        
        # Apply depletion (reduce size at best price)
        if side == 'B' and bbo[0]:
            # Reduce bid size
            if bbo[0] in order_book.bids:
                current_size = order_book.bids[bbo[0]]['total_size']
                depletion_size = min(150, current_size)  # Remove up to all available, min 150
                if depletion_size > 0:
                    order_book.bids[bbo[0]]['total_size'] = max(0, current_size - depletion_size)
                    order_book.bids[bbo[0]]['pegged_size'] = max(0, order_book.bids[bbo[0]]['pegged_size'] - depletion_size // 2)
                    order_book.bids[bbo[0]]['non_pegged_size'] = max(0, order_book.bids[bbo[0]]['non_pegged_size'] - depletion_size // 2)
                    
                    if order_book.bids[bbo[0]]['total_size'] == 0:
                        del order_book.bids[bbo[0]]
                        order_book._update_best_bid()
                    # print(f"    Reduced bid size by {depletion_size}")
        
        elif side == 'A' and bbo[2]:
            # Reduce ask size
            if bbo[2] in order_book.asks:
                current_size = order_book.asks[bbo[2]]['total_size']
                depletion_size = min(150, current_size)  # Remove up to all available, min 150
                if depletion_size > 0:
                    order_book.asks[bbo[2]]['total_size'] = max(0, current_size - depletion_size)
                    order_book.asks[bbo[2]]['pegged_size'] = max(0, order_book.asks[bbo[2]]['pegged_size'] - depletion_size // 2)
                    order_book.asks[bbo[2]]['non_pegged_size'] = max(0, order_book.asks[bbo[2]]['non_pegged_size'] - depletion_size // 2)
                    
                    if order_book.asks[bbo[2]]['total_size'] == 0:
                        del order_book.asks[bbo[2]]
                        order_book._update_best_ask()
                    # print(f"    Reduced ask size by {depletion_size}")
        
        # Get new BBO
        new_bbo = order_book.get_best_bid_offer()
        # print(f"  BBO after: {new_bbo}")
        
        # Publish market data event
        self.publish_market_data({
            'type': 'book_update',
            'symbol': symbol,
            'venue': self.name,
            'bbo': self.get_bbo(symbol)
        })
    
    def handle_addition_event(self, event):
        """Handle addition events (liquidity additions)."""
        data = event.data
        size = data.get('size', 100)
        target_exchange = data.get('target_exchange')
        
        if target_exchange != self.name:
            return
        
        # Add liquidity to a random symbol
        if not self.order_books:
            return
        
        symbol = list(self.order_books.keys())[0]  # Use first symbol
        order_book = self.order_books[symbol]
        
        # Get current BBO
        bbo = order_book.get_best_bid_offer()
        if not bbo[0] or not bbo[2]:  # No valid BBO
            return
        
        # print(f"  ADDITION: {self.name} adding {size} shares, BBO before: {bbo}")
        
        # Add liquidity at best price
        side = 'B' if np.random.random() < 0.5 else 'A'
        
        if side == 'B' and bbo[0]:
            # Add to bid
            if bbo[0] not in order_book.bids:
                order_book.bids[bbo[0]] = {'total_size': 0, 'pegged_size': 0, 'non_pegged_size': 0}
            order_book.bids[bbo[0]]['total_size'] += size
            order_book.bids[bbo[0]]['non_pegged_size'] += size  # Assume additions are non-pegged
            # print(f"    Added {size} to bid at {bbo[0]}")
        elif side == 'A' and bbo[2]:
            # Add to ask
            if bbo[2] not in order_book.asks:
                order_book.asks[bbo[2]] = {'total_size': 0, 'pegged_size': 0, 'non_pegged_size': 0}
            order_book.asks[bbo[2]]['total_size'] += size
            order_book.asks[bbo[2]]['non_pegged_size'] += size  # Assume additions are non-pegged
            # print(f"    Added {size} to ask at {bbo[2]}")
        
        # Get new BBO
        new_bbo = order_book.get_best_bid_offer()
        # print(f"  BBO after: {new_bbo}")
        
        # Publish market data event
        self.publish_market_data({
            'type': 'book_update',
            'symbol': symbol,
            'venue': self.name,
            'bbo': self.get_bbo(symbol)
        })
    
    def handle_signal(self, event):
        """Handle Crumbling Quote Signal events."""
        if event.signal_type == 'activate':
            # Activate signal for all order books
            for order_book in self.order_books.values():
                order_book.set_signal_active(True)
        elif event.signal_type == 'deactivate':
            # Deactivate signal for all order books
            for order_book in self.order_books.values():
                order_book.set_signal_active(False)
    
    def handle_signal_activation(self, event):
        """Handle CQS signal activation events."""
        if self.name == "IEX":  # Only IEX has CQS protection
            for order_book in self.order_books.values():
                order_book.set_signal_active(True)
            print(f"  IEX CQS Signal ACTIVATED at {event.timestamp:.6f}s")
    
    def handle_signal_deactivation(self, event):
        """Handle CQS signal deactivation events."""
        if self.name == "IEX":  # Only IEX has CQS protection
            for order_book in self.order_books.values():
                order_book.set_signal_active(False)
            print(f"  IEX CQS Signal DEACTIVATED at {event.timestamp:.6f}s")
    
    def get_bbo(self, symbol: str) -> Dict[str, Any]:
        """Get the Best Bid and Offer for a symbol."""
        order_book = self.get_order_book(symbol)
        if not order_book:
            return {'bid_price': None, 'bid_size': 0, 'ask_price': None, 'ask_size': 0}
        
        best_bid, best_bid_size, best_ask, best_ask_size = order_book.get_best_bid_offer()
        
        return {
            'bid_price': float(best_bid) if best_bid else None,
            'bid_size': best_bid_size,
            'ask_price': float(best_ask) if best_ask else None,
            'ask_size': best_ask_size
        }
    
    def publish_market_data(self, data: Dict[str, Any]):
        """Publish market data events to the simulator with proper latency."""
        # Add venue information to the data
        data['venue'] = self.name
        current_time = self.simulator.get_current_time()
        
        if self.latency_model and self.agents:
            # Deliver to each agent with individual data latency
            for agent_id in self.agents:
                data_latency = self.latency_model.get_data_latency(self.name, agent_id)
                arrival_time = current_time + data_latency
                
                # Create agent-specific market data event
                agent_data = data.copy()
                agent_data['recipient'] = agent_id
                
                self.simulator.schedule_event_at(
                    arrival_time,
                    "market_data",
                    agent_data
                )
                
                # Also schedule book_update event for compatibility
                self.simulator.schedule_event_at(
                    arrival_time,
                    "book_update", 
                    agent_data
                )
        else:
            # Fallback to immediate delivery (for backward compatibility)
            self.simulator.schedule_event_at(
                current_time,
                "market_data",
                data
            )
            
            self.simulator.schedule_event_at(
                current_time,
                "book_update",
                data
            )
        
# Removed ground truth state change - using original market_data approach

if __name__ == "__main__":
    # Test the market environment
    from simulator import Simulator
    
    # Create simulator and exchange
    sim = Simulator(max_time=5.0)
    exchange = Exchange("IEX", sim)
    
    # Add a test symbol
    exchange.add_symbol("AAPL", Decimal('0.01'))
    
    # Test order book
    order_book = exchange.get_order_book("AAPL")
    
    # Add some orders
    order_book.add_order('B', Decimal('150.00'), 1000, is_pegged=False)
    order_book.add_order('B', Decimal('149.99'), 500, is_pegged=True)
    order_book.add_order('A', Decimal('150.01'), 800, is_pegged=False)
    order_book.add_order('A', Decimal('150.02'), 300, is_pegged=True)
    
    print("Initial order book state:")
    print(f"BBO: {exchange.get_bbo('AAPL')}")
    
    # Test signal activation
    print("\nActivating signal...")
    order_book.set_signal_active(True)
    print(f"Available bid size at 150.00: {order_book.get_available_size('B', Decimal('150.00'))}")
    print(f"Available ask size at 150.01: {order_book.get_available_size('A', Decimal('150.01'))}")
    
    # Test order matching
    print("\nTesting order matching:")
    filled_size, fills = order_book.match_order('B', Decimal('150.01'), 200)
    print(f"Filled {filled_size} shares: {fills}")
    print(f"BBO after match: {exchange.get_bbo('AAPL')}")
    
    print("\nMarket environment test complete!")
