#!/usr/bin/env python3
"""
Event-Driven Simulation Engine
Implements Section 2 from the LaTeX specification
"""

import heapq
from typing import Any, Callable, Optional
from datetime import datetime, timedelta
import time

class Event:
    """
    Event class for the simulation engine.
    
    Attributes:
        timestamp: When the event should be processed
        priority: Secondary ordering (lower number = higher priority)
        event_type: Type of event for routing
        data: Event-specific data
    """
    
    def __init__(self, timestamp: float, event_type: str, data: Any = None, priority: int = 0):
        self.timestamp = timestamp
        self.priority = priority
        self.event_type = event_type
        self.data = data
    
    def __lt__(self, other):
        """Less than comparison for priority queue ordering."""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority
    
    def __repr__(self):
        return f"Event(t={self.timestamp:.6f}, type={self.event_type}, priority={self.priority})"

class Simulator:
    """
    Main simulation engine with priority queue.
    
    Manages event scheduling and processing in time order.
    """
    
    def __init__(self, max_time: Optional[float] = None):
        self.event_queue = []  # Priority queue
        self.current_time = 0.0
        self.max_time = max_time
        self.event_handlers = {}  # event_type -> handler function
        self.stats = {
            'events_processed': 0,
            'events_scheduled': 0,
            'simulation_time': 0.0
        }
    
    def register_handler(self, event_type: str, handler: Callable[[Event], None]):
        """Register an event handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def schedule_event(self, event: Event):
        """Schedule an event for future processing."""
        heapq.heappush(self.event_queue, event)
        self.stats['events_scheduled'] += 1
    
    def schedule_event_at(self, timestamp: float, event_type: str, data: Any = None, priority: int = 0):
        """Convenience method to schedule an event at a specific time."""
        event = Event(timestamp, event_type, data, priority)
        self.schedule_event(event)
    
    def schedule_event_after(self, delay: float, event_type: str, data: Any = None, priority: int = 0):
        """Convenience method to schedule an event after a delay."""
        timestamp = self.current_time + delay
        self.schedule_event_at(timestamp, event_type, data, priority)
    
    def process_event(self, event: Event):
        """Process a single event by calling its handler."""
        if event.event_type in self.event_handlers:
            # Call all handlers for this event type
            handlers = self.event_handlers[event.event_type]
            if isinstance(handlers, list):
                for handler in handlers:
                    handler(event)
            else:
                handlers(event)
        else:
            print(f"Warning: No handler registered for event type '{event.event_type}'")
    
    def run(self, verbose: bool = False):
        """
        Run the simulation by processing events in time order.
        
        Args:
            verbose: If True, print event processing information
        """
        print(f"Starting simulation (max_time={self.max_time})")
        print("=" * 50)
        
        start_time = time.time()
        
        while self.event_queue:
            # Check if we've exceeded max time
            if self.max_time is not None and self.current_time >= self.max_time:
                print(f"Simulation stopped at max_time={self.max_time}")
                break
            
            # Get next event
            event = heapq.heappop(self.event_queue)
            
            # Update simulation time
            self.current_time = event.timestamp
            
            # Process the event
            if verbose:
                print(f"Processing {event} at time {self.current_time:.6f}")
            
            self.process_event(event)
            self.stats['events_processed'] += 1
            
            # Progress indicator
            if self.stats['events_processed'] % 1000 == 0:
                print(f"Processed {self.stats['events_processed']} events, current_time={self.current_time:.6f}")
        
        # Update final stats
        self.stats['simulation_time'] = time.time() - start_time
        
        print(f"\nSimulation complete!")
        print(f"Events processed: {self.stats['events_processed']}")
        print(f"Events scheduled: {self.stats['events_scheduled']}")
        print(f"Final time: {self.current_time:.6f}")
        print(f"Real time: {self.stats['simulation_time']:.2f} seconds")
    
    def get_current_time(self) -> float:
        """Get the current simulation time."""
        return self.current_time
    
    def get_stats(self) -> dict:
        """Get simulation statistics."""
        return self.stats.copy()

# Example event types for the IEX simulation
class MarketDataEvent(Event):
    """Event representing market data updates."""
    def __init__(self, timestamp: float, venue: str, data: dict, priority: int = 1):
        super().__init__(timestamp, "market_data", data, priority)
        self.venue = venue

class OrderEvent(Event):
    """Event representing order arrivals."""
    def __init__(self, timestamp: float, venue: str, order: dict, priority: int = 2):
        super().__init__(timestamp, "order", order, priority)
        self.venue = venue

class SignalEvent(Event):
    """Event representing Crumbling Quote Signal activation/deactivation."""
    def __init__(self, timestamp: float, signal_type: str, data: dict = None, priority: int = 0):
        super().__init__(timestamp, "signal", data, priority)
        self.signal_type = signal_type  # 'activate' or 'deactivate'

if __name__ == "__main__":
    # Test the simulator
    def test_handler(event):
        print(f"  Processing: {event}")
    
    # Create simulator
    sim = Simulator(max_time=10.0)
    sim.register_handler("test", test_handler)
    
    # Schedule some test events
    sim.schedule_event_at(1.0, "test", "first event")
    sim.schedule_event_at(3.0, "test", "second event")
    sim.schedule_event_at(2.0, "test", "third event", priority=1)
    sim.schedule_event_at(2.0, "test", "fourth event", priority=2)
    
    print("Test events scheduled:")
    for event in sim.event_queue:
        print(f"  {event}")
    
    print("\nRunning simulation:")
    sim.run(verbose=True)
