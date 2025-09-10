#!/usr/bin/env python3
"""
Logging and Ground Truth Labeling System
Implements Section 7 from the LaTeX specification
"""

import json
import csv
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict

@dataclass
class LogEvent:
    """Base class for logged events."""
    timestamp: float
    event_type: str
    data: Dict[str, Any]

@dataclass
class CQSFireEvent(LogEvent):
    """CQS signal fire event."""
    model: str
    features: Dict[str, Any]
    risk_score: Optional[float] = None

@dataclass
class HFTArbitrageEvent(LogEvent):
    """HFT arbitrage attempt event."""
    agent_id: str
    symbol: str
    side: str
    price: float
    size: int
    success: bool
    pnl: float = 0.0

@dataclass
class NBBOChangeEvent(LogEvent):
    """NBBO price change (tick) event."""
    symbol: str
    old_bid: Optional[float]
    old_ask: Optional[float]
    new_bid: Optional[float]
    new_ask: Optional[float]
    venues_at_bid: int
    venues_at_ask: int

@dataclass
class GroundTruthLabel:
    """Ground truth label for CQS fires."""
    fire_timestamp: float
    model: str
    label: str  # 'TP', 'FP', or 'FN'
    tick_timestamp: Optional[float] = None
    tick_direction: Optional[str] = None  # 'bid_up', 'ask_down', 'bid_down', 'ask_up'
    vpa: float = 0.0
    eoc: float = 0.0
    available_shares: float = 100.0  # Available shares at stale price for VPA calculation (will be set dynamically)

class EventLogger:
    """
    Centralized event logging system.
    
    Logs all key simulation events for analysis and ground truth labeling.
    """
    
    def __init__(self, log_file: str = "simulation_log.jsonl"):
        self.log_file = log_file
        self.events = []
        self.cqs_fires = []
        self.hft_attempts = []
        self.nbbo_changes = []
        
        # Ground truth labeling state
        self.tick_window = 0.002  # 2ms window
        self.pending_fires = deque()  # Fires waiting for tick evaluation
        
    def log_cqs_fire(self, timestamp: float, model: str, features: Dict[str, Any], 
                    risk_score: Optional[float] = None):
        """Log a CQS signal fire event."""
        event = CQSFireEvent(
            timestamp=timestamp,
            event_type="cqs_fire",
            model=model,
            features=features,
            risk_score=risk_score,
            data={'model': model, 'features': features, 'risk_score': risk_score}
        )
        self.events.append(event)
        self.cqs_fires.append(event)
        
        # Add to pending fires for ground truth evaluation
        self.pending_fires.append(event)
        
        risk_str = f"{risk_score:.4f}" if risk_score is not None else "N/A"
        print(f"  CQS FIRE: {model} at {timestamp:.6f}s (risk={risk_str})")
    
    def log_hft_arbitrage(self, timestamp: float, agent_id: str, symbol: str, 
                         side: str, price: float, size: int, success: bool, pnl: float = 0.0):
        """Log an HFT arbitrage attempt."""
        event = HFTArbitrageEvent(
            timestamp=timestamp,
            event_type="hft_arbitrage",
            agent_id=agent_id,
            symbol=symbol,
            side=side,
            price=price,
            size=size,
            success=success,
            pnl=pnl,
            data={
                'agent_id': agent_id,
                'symbol': symbol,
                'side': side,
                'price': price,
                'size': size,
                'success': success,
                'pnl': pnl
            }
        )
        self.events.append(event)
        self.hft_attempts.append(event)
        
        status = "SUCCESS" if success else "BLOCKED"
        print(f"  HFT ARBITRAGE: {agent_id} {symbol} {side} {size}@{price:.2f} - {status} (P&L={pnl:.2f})")
    
    def log_nbbo_change(self, timestamp: float, symbol: str, old_bid: Optional[float],
                       old_ask: Optional[float], new_bid: Optional[float], new_ask: Optional[float],
                       venues_at_bid: int, venues_at_ask: int):
        """Log an NBBO price change (tick)."""
        event = NBBOChangeEvent(
            timestamp=timestamp,
            event_type="nbbo_change",
            symbol=symbol,
            old_bid=old_bid,
            old_ask=old_ask,
            new_bid=new_bid,
            new_ask=new_ask,
            venues_at_bid=venues_at_bid,
            venues_at_ask=venues_at_ask,
            data={
                'symbol': symbol,
                'old_bid': old_bid,
                'old_ask': old_ask,
                'new_bid': new_bid,
                'new_ask': new_ask,
                'venues_at_bid': venues_at_bid,
                'venues_at_ask': venues_at_ask
            }
        )
        self.events.append(event)
        self.nbbo_changes.append(event)
        
        # Process pending fires for ground truth labeling
        self._process_pending_fires(timestamp)
        
        print(f"  NBBO TICK: {symbol} bid {old_bid}->{new_bid}, ask {old_ask}->{new_ask} at {timestamp:.6f}s")
    
    def _process_pending_fires(self, current_timestamp: float):
        """Process pending fires against the latest tick."""
        if not self.nbbo_changes:
            return
        
        latest_tick = self.nbbo_changes[-1]
        
        # Check each pending fire
        fires_to_remove = []
        for i, fire in enumerate(self.pending_fires):
            # Check if fire is within the 2ms window
            if current_timestamp - fire.timestamp <= self.tick_window:
                # Determine if this is a TP or FP
                label = self._evaluate_fire_label(fire, latest_tick)
                
                if label == 'TP':
                    print(f"    -> TRUE POSITIVE: Fire at {fire.timestamp:.6f}s predicted tick at {latest_tick.timestamp:.6f}s")
                elif label == 'FP':
                    print(f"    -> FALSE POSITIVE: Fire at {fire.timestamp:.6f}s, no tick in window")
                
                fires_to_remove.append(i)
        
        # Remove processed fires (in reverse order to maintain indices)
        for i in reversed(fires_to_remove):
            del self.pending_fires[i]
    
    def _evaluate_fire_label(self, fire: CQSFireEvent, tick: NBBOChangeEvent) -> str:
        """Evaluate whether a fire is TP or FP based on the tick."""
        # Check if tick occurred within the window
        if tick.timestamp - fire.timestamp > self.tick_window:
            return 'FP'  # No tick in window
        
        # Check if the tick direction matches the fire prediction
        # For simplicity, we'll assume any tick in the window is a TP
        # In a full implementation, we'd check the specific direction
        return 'TP'
    
    def finalize_logging(self):
        """Finalize logging and process any remaining pending fires."""
        # Mark any remaining pending fires as FP
        for fire in self.pending_fires:
            print(f"    -> FALSE POSITIVE: Fire at {fire.timestamp:.6f}s, no tick in final window")
        
        self.pending_fires.clear()
        
        # Save events to file
        self._save_events()
    
    def _save_events(self):
        """Save all events to the log file."""
        with open(self.log_file, 'w') as f:
            for event in self.events:
                f.write(json.dumps(asdict(event)) + '\n')
        
        print(f"Saved {len(self.events)} events to {self.log_file}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'total_events': len(self.events),
            'cqs_fires': len(self.cqs_fires),
            'hft_attempts': len(self.hft_attempts),
            'nbbo_changes': len(self.nbbo_changes),
            'pending_fires': len(self.pending_fires)
        }

class GroundTruthLabeler:
    """
    Ground truth labeling system for CQS fires.
    
    Implements the 2ms rule for TP/FP/FN classification.
    """
    
    def __init__(self, tick_window: float = 0.002):
        self.tick_window = tick_window
        self.labels = []
    
    def label_fires(self, cqs_fires: List[CQSFireEvent], 
                   nbbo_changes: List[NBBOChangeEvent]) -> List[GroundTruthLabel]:
        """
        Label CQS fires as TP, FP, or FN.
        
        Args:
            cqs_fires: List of CQS fire events
            nbbo_changes: List of NBBO change events
            
        Returns:
            List of ground truth labels
        """
        labels = []
        
        # Process each CQS fire
        for fire in cqs_fires:
            label = self._label_single_fire(fire, nbbo_changes)
            labels.append(label)
        
        # Find False Negatives (ticks without fires)
        fns = self._find_false_negatives(cqs_fires, nbbo_changes)
        labels.extend(fns)
        
        self.labels = labels
        return labels
    
    def _label_single_fire(self, fire: CQSFireEvent, 
                          nbbo_changes: List[NBBOChangeEvent]) -> GroundTruthLabel:
        """Label a single CQS fire according to 2ms rule from latex.txt."""
        # Find ticks within the 2ms window AFTER the fire
        # CQS fire at time T should predict ticks in [T, T+2ms]
        window_end = fire.timestamp + self.tick_window
        relevant_ticks = [
            tick for tick in nbbo_changes
            if fire.timestamp <= tick.timestamp <= window_end
        ]
        
        if not relevant_ticks:
            # No tick in window - False Positive
            return GroundTruthLabel(
                fire_timestamp=fire.timestamp,
                model=fire.model,
                label='FP'
            )
        
        # Check if any tick matches the predicted direction
        # Determine predicted direction from fire features
        predicted_direction = self._get_predicted_direction(fire)
        
        for tick in relevant_ticks:
            tick_direction = self._determine_tick_direction(tick)
            # Check if tick direction matches predicted direction per LaTeX specification
            if self._directions_match(predicted_direction, tick_direction):
                return GroundTruthLabel(
                    fire_timestamp=fire.timestamp,
                    model=fire.model,
                    label='TP',
                    tick_timestamp=tick.timestamp,
                    tick_direction=tick_direction,
                    available_shares=100  # Default estimate for VPA calculation
                )
            # If tick in wrong direction, continue to check other ticks in window
        
        # Tick occurred but in wrong direction - False Positive
        return GroundTruthLabel(
            fire_timestamp=fire.timestamp,
            model=fire.model,
            label='FP',
            tick_timestamp=relevant_ticks[0].timestamp,
            tick_direction=self._determine_tick_direction(relevant_ticks[0])
        )
    
    def _determine_tick_direction(self, tick: NBBOChangeEvent) -> str:
        """Determine the direction of a tick."""
        # Check for actual price changes first
        if tick.old_bid and tick.new_bid and tick.new_bid > tick.old_bid:
            return 'bid_up'
        elif tick.old_ask and tick.new_ask and tick.new_ask < tick.old_ask:
            return 'ask_down'
        elif tick.old_bid and tick.new_bid and tick.new_bid < tick.old_bid:
            return 'bid_down'
        elif tick.old_ask and tick.new_ask and tick.new_ask > tick.old_ask:
            return 'ask_up'
        # If no price change but venue counts changed, count as any movement
        elif (tick.old_bid != tick.new_bid or tick.old_ask != tick.new_ask or 
              tick.venues_at_bid != tick.venues_at_ask):
            return 'any_movement'
        else:
            return 'unknown'
    
    def _get_predicted_direction(self, fire: CQSFireEvent) -> str:
        """Determine predicted direction from fire features."""
        features = fire.features
        
        # Check which side had venues decrease
        bids_decreased = features.get('bids', 0) < features.get('bids_lag1', 0)
        asks_decreased = features.get('asks', 0) < features.get('asks_lag1', 0)
        
        if bids_decreased and not asks_decreased:
            # Bid side decreased - expect bid price to drop or ask price to rise
            return 'bid_down_or_ask_up'
        elif asks_decreased and not bids_decreased:
            # Ask side decreased - expect ask price to drop or bid price to rise
            return 'ask_down_or_bid_up'
        elif bids_decreased and asks_decreased:
            # Both sides decreased - expect any price movement
            return 'any_movement'
        else:
            # No clear prediction
            return 'unknown'
    
    def _directions_match(self, predicted: str, actual: str) -> bool:
        """Check if actual tick direction matches prediction."""
        if predicted == 'unknown':
            return False
        
        if predicted == 'any_movement':
            return actual in ['bid_up', 'bid_down', 'ask_up', 'ask_down', 'any_movement']
        
        if predicted == 'bid_down_or_ask_up':
            return actual in ['bid_down', 'ask_up', 'any_movement']
        
        if predicted == 'ask_down_or_bid_up':
            return actual in ['ask_down', 'bid_up', 'any_movement']
        
        return actual == predicted
    
    def _find_false_negatives(self, cqs_fires: List[CQSFireEvent],
                             nbbo_changes: List[NBBOChangeEvent]) -> List[GroundTruthLabel]:
        """Find False Negatives - ticks without fires."""
        fns = []
        
        for tick in nbbo_changes:
            # Check if there was a fire within 2ms before this tick
            window_start = tick.timestamp - self.tick_window
            relevant_fires = [
                fire for fire in cqs_fires
                if window_start <= fire.timestamp <= tick.timestamp
            ]
            
            if not relevant_fires:
                # No fire before this tick - False Negative
                fns.append(GroundTruthLabel(
                    fire_timestamp=tick.timestamp,
                    model='FN',
                    label='FN',
                    tick_timestamp=tick.timestamp,
                    tick_direction=self._determine_tick_direction(tick)
                ))
        
        return fns
    
    def calculate_precision_recall(self) -> Dict[str, float]:
        """Calculate precision and recall from labels."""
        tps = sum(1 for label in self.labels if label.label == 'TP')
        fps = sum(1 for label in self.labels if label.label == 'FP')
        fns = sum(1 for label in self.labels if label.label == 'FN')
        
        precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tps': tps,
            'fps': fps,
            'fns': fns
        }
    
    def save_labels(self, filename: str = "ground_truth_labels.csv"):
        """Save labels to CSV file."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'fire_timestamp', 'model', 'label', 'tick_timestamp', 
                'tick_direction', 'vpa', 'eoc'
            ])
            
            for label in self.labels:
                writer.writerow([
                    label.fire_timestamp,
                    label.model,
                    label.label,
                    label.tick_timestamp or '',
                    label.tick_direction or '',
                    label.vpa,
                    label.eoc
                ])
        
        print(f"Saved {len(self.labels)} labels to {filename}")

if __name__ == "__main__":
    # Test the logging system
    print("Testing Logging System")
    print("=" * 30)
    
    # Create logger
    logger = EventLogger("test_log.jsonl")
    
    # Test CQS fire logging
    print("\n1. Testing CQS Fire Logging:")
    print("-" * 30)
    
    features = {'bids': 3, 'asks': 2, 'bids_lag1': 4, 'asks_lag1': 3}
    logger.log_cqs_fire(0.1, "heuristic", features)
    logger.log_cqs_fire(0.2, "logistic", features, risk_score=0.75)
    
    # Test HFT arbitrage logging
    print("\n2. Testing HFT Arbitrage Logging:")
    print("-" * 30)
    
    logger.log_hft_arbitrage(0.15, "HFT1", "AAPL", "B", 150.01, 100, True, 5.0)
    logger.log_hft_arbitrage(0.25, "HFT1", "AAPL", "A", 150.00, 200, False, -1.0)
    
    # Test NBBO change logging
    print("\n3. Testing NBBO Change Logging:")
    print("-" * 30)
    
    logger.log_nbbo_change(0.12, "AAPL", 150.00, 150.01, 150.01, 150.02, 2, 3)
    logger.log_nbbo_change(0.22, "AAPL", 150.01, 150.02, 150.02, 150.03, 1, 2)
    
    # Finalize logging
    logger.finalize_logging()
    
    # Test ground truth labeling
    print("\n4. Testing Ground Truth Labeling:")
    print("-" * 30)
    
    labeler = GroundTruthLabeler()
    labels = labeler.label_fires(logger.cqs_fires, logger.nbbo_changes)
    
    print(f"Generated {len(labels)} labels")
    
    # Calculate precision/recall
    metrics = labeler.calculate_precision_recall()
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    
    # Save labels
    labeler.save_labels("test_labels.csv")
    
    print(f"\nLogger stats: {logger.get_stats()}")
    print("\nLogging system test complete!")
