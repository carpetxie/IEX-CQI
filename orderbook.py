#!/usr/bin/env python3
"""
Limit Order Book (LOB) Implementation
Implements Section 3.1 from the LaTeX specification
"""

from decimal import Decimal
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import csv
from datetime import datetime

class OrderBook:
    """
    Limit Order Book implementation for IEX DEEP data reconstruction.
    
    Maintains separate bid and ask sides with price-time priority.
    """
    
    def __init__(self):
        # Price levels: {price: total_size}
        self.bids = {}  # price -> size (sorted descending)
        self.asks = {}  # price -> size (sorted ascending)
        
        # Track best prices
        self.best_bid_price = None
        self.best_ask_price = None
        
    def add_order(self, side: str, price: Decimal, size: int) -> None:
        """
        Add an order to the book.
        
        Args:
            side: 'B' for bid, 'A' for ask
            price: Order price
            size: Order size (positive for additions, negative for cancellations)
        """
        if side == 'B':
            if price in self.bids:
                self.bids[price] += size
            else:
                self.bids[price] = size
            
            # Remove empty price levels
            if self.bids[price] <= 0:
                del self.bids[price]
            
            # Update best bid
            if self.bids:
                self.best_bid_price = max(self.bids.keys())
            else:
                self.best_bid_price = None
                
        elif side == 'A':
            if price in self.asks:
                self.asks[price] += size
            else:
                self.asks[price] = size
            
            # Remove empty price levels
            if self.asks[price] <= 0:
                del self.asks[price]
            
            # Update best ask
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
        best_bid_size = self.bids.get(self.best_bid_price, 0) if self.best_bid_price else 0
        best_ask_size = self.asks.get(self.best_ask_price, 0) if self.best_ask_price else 0
        
        return (self.best_bid_price, best_bid_size, 
                self.best_ask_price, best_ask_size)
    
    def apply_trade(self, side: str, price: Decimal, size: int) -> None:
        """
        Apply a trade to the book.
        
        Args:
            side: 'B' for bid side trade, 'A' for ask side trade
            price: Trade price
            size: Trade size
        """
        if side == 'B':
            # Trade on bid side reduces bid size
            if price in self.bids:
                self.bids[price] = max(0, self.bids[price] - size)
                if self.bids[price] == 0:
                    del self.bids[price]
                    # Update best bid
                    if self.bids:
                        self.best_bid_price = max(self.bids.keys())
                    else:
                        self.best_bid_price = None
        elif side == 'A':
            # Trade on ask side reduces ask size
            if price in self.asks:
                self.asks[price] = max(0, self.asks[price] - size)
                if self.asks[price] == 0:
                    del self.asks[price]
                    # Update best ask
                    if self.asks:
                        self.best_ask_price = min(self.asks.keys())
                    else:
                        self.best_ask_price = None
    
    def clear(self) -> None:
        """Clear the entire order book."""
        self.bids.clear()
        self.asks.clear()
        self.best_bid_price = None
        self.best_ask_price = None
    
    def __str__(self) -> str:
        """String representation of the order book."""
        result = "OrderBook:\n"
        result += "Bids:\n"
        for price in sorted(self.bids.keys(), reverse=True)[:5]:  # Top 5 bids
            result += f"  {price}: {self.bids[price]}\n"
        result += "Asks:\n"
        for price in sorted(self.asks.keys())[:5]:  # Top 5 asks
            result += f"  {price}: {self.asks[price]}\n"
        return result
