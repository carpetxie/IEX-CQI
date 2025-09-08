#!/usr/bin/env python3
"""
Economic Metrics (KPIs) Calculation
Implements Section 8 from the LaTeX specification
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from logging_system import GroundTruthLabel, CQSFireEvent, HFTArbitrageEvent, NBBOChangeEvent

@dataclass
class KPIMetrics:
    """Container for KPI metrics."""
    vpa: float  # Value of Prevented Arbitrage
    eoc: float  # Exchange Opportunity Cost
    hft_pnl: float  # HFT Arbitrage Profitability
    precision: float
    recall: float
    f1_score: float
    total_events: int
    tps: int
    fps: int
    fns: int

class KPICalculator:
    """
    Calculator for economic metrics (KPIs).
    
    Implements the three main KPIs from the LaTeX specification:
    1. Value of Prevented Arbitrage (VPA)
    2. Exchange Opportunity Cost (EOC)
    3. HFT Arbitrage Profitability
    """
    
    def __init__(self, exchange_fee: float = 0.0001, max_fill_size: int = 1000, 
                 attempt_cost: float = 0.01):
        """
        Initialize KPI calculator.
        
        Args:
            exchange_fee: Per-share exchange fee (default: 0.01 bps)
            max_fill_size: Maximum fill size for VPA calculation
            attempt_cost: Cost per blocked arbitrage attempt
        """
        self.exchange_fee = exchange_fee
        self.max_fill_size = max_fill_size
        self.attempt_cost = attempt_cost
    
    def calculate_vpa(self, labels: List[GroundTruthLabel], 
                     nbbo_changes: List[NBBOChangeEvent]) -> float:
        """
        Calculate Value of Prevented Arbitrage (VPA).
        
        For each true positive (a protected stale trade):
        VPA_event = |P_stale - P_true| × min(S_available, max_fill_size)
        
        Args:
            labels: Ground truth labels
            nbbo_changes: NBBO change events
            
        Returns:
            Total VPA value
        """
        vpa_total = 0.0
        
        for label in labels:
            if label.label == 'TP' and label.tick_timestamp:
                # Find the corresponding tick
                tick = self._find_tick_at_time(nbbo_changes, label.tick_timestamp)
                if tick:
                    # Calculate price difference
                    price_diff = self._calculate_price_difference(tick)
                    
                    # Calculate available size (simplified)
                    available_size = min(tick.venues_at_bid + tick.venues_at_ask, self.max_fill_size)
                    
                    # VPA for this event
                    vpa_event = price_diff * available_size
                    vpa_total += vpa_event
                    
                    # Update label with VPA
                    label.vpa = vpa_event
        
        return vpa_total
    
    def calculate_eoc(self, labels: List[GroundTruthLabel], 
                     cqs_fires: List[CQSFireEvent]) -> float:
        """
        Calculate Exchange Opportunity Cost (EOC).
        
        For each missed trade (TP or FP):
        EOC_event = V_missed × F_exchange
        
        Args:
            labels: Ground truth labels
            cqs_fires: CQS fire events
            
        Returns:
            Total EOC value
        """
        eoc_total = 0.0
        
        for label in labels:
            if label.label in ['TP', 'FP']:
                # Find corresponding CQS fire
                fire = self._find_fire_at_time(cqs_fires, label.fire_timestamp)
                if fire:
                    # Calculate missed volume (simplified)
                    # In practice, this would be based on protected pegged size
                    missed_volume = self._estimate_missed_volume(fire)
                    
                    # EOC for this event
                    eoc_event = missed_volume * self.exchange_fee
                    eoc_total += eoc_event
                    
                    # Update label with EOC
                    label.eoc = eoc_event
        
        return eoc_total
    
    def calculate_hft_pnl(self, labels: List[GroundTruthLabel], 
                         hft_attempts: List[HFTArbitrageEvent]) -> float:
        """
        Calculate HFT Arbitrage Profitability.
        
        P&L = Σ(wins on FNs) VPA_event - Σ(blocked TPs) C_attempt
        
        Args:
            labels: Ground truth labels
            hft_attempts: HFT arbitrage attempt events
            
        Returns:
            Total HFT P&L
        """
        pnl_total = 0.0
        
        # Calculate wins on False Negatives
        fn_vpa = sum(label.vpa for label in labels if label.label == 'FN')
        
        # Calculate losses from blocked True Positives
        blocked_tps = sum(1 for label in labels if label.label == 'TP')
        blocked_cost = blocked_tps * self.attempt_cost
        
        # Calculate actual HFT P&L from attempts
        hft_pnl = sum(attempt.pnl for attempt in hft_attempts)
        
        # Total P&L
        pnl_total = fn_vpa + hft_pnl - blocked_cost
        
        return pnl_total
    
    def calculate_precision_recall(self, labels: List[GroundTruthLabel]) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            labels: Ground truth labels
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        tps = sum(1 for label in labels if label.label == 'TP')
        fps = sum(1 for label in labels if label.label == 'FP')
        fns = sum(1 for label in labels if label.label == 'FN')
        
        precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score
    
    def calculate_all_metrics(self, labels: List[GroundTruthLabel], 
                             nbbo_changes: List[NBBOChangeEvent],
                             cqs_fires: List[CQSFireEvent],
                             hft_attempts: List[HFTArbitrageEvent]) -> KPIMetrics:
        """
        Calculate all KPI metrics.
        
        Args:
            labels: Ground truth labels
            nbbo_changes: NBBO change events
            cqs_fires: CQS fire events
            hft_attempts: HFT arbitrage attempt events
            
        Returns:
            KPIMetrics object with all calculated metrics
        """
        # Calculate individual metrics
        vpa = self.calculate_vpa(labels, nbbo_changes)
        eoc = self.calculate_eoc(labels, cqs_fires)
        hft_pnl = self.calculate_hft_pnl(labels, hft_attempts)
        
        # Calculate precision/recall
        precision, recall, f1_score = self.calculate_precision_recall(labels)
        
        # Count events
        tps = sum(1 for label in labels if label.label == 'TP')
        fps = sum(1 for label in labels if label.label == 'FP')
        fns = sum(1 for label in labels if label.label == 'FN')
        
        return KPIMetrics(
            vpa=vpa,
            eoc=eoc,
            hft_pnl=hft_pnl,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_events=len(labels),
            tps=tps,
            fps=fps,
            fns=fns
        )
    
    def _find_tick_at_time(self, nbbo_changes: List[NBBOChangeEvent], 
                          timestamp: float) -> Optional[NBBOChangeEvent]:
        """Find NBBO change event at specific timestamp."""
        for tick in nbbo_changes:
            if abs(tick.timestamp - timestamp) < 0.0001:  # 0.1ms tolerance
                return tick
        return None
    
    def _find_fire_at_time(self, cqs_fires: List[CQSFireEvent], 
                          timestamp: float) -> Optional[CQSFireEvent]:
        """Find CQS fire event at specific timestamp."""
        for fire in cqs_fires:
            if abs(fire.timestamp - timestamp) < 0.0001:  # 0.1ms tolerance
                return fire
        return None
    
    def _calculate_price_difference(self, tick: NBBOChangeEvent) -> float:
        """Calculate price difference for VPA calculation."""
        bid_diff = 0.0
        ask_diff = 0.0
        
        if tick.old_bid and tick.new_bid:
            bid_diff = abs(tick.new_bid - tick.old_bid)
        
        if tick.old_ask and tick.new_ask:
            ask_diff = abs(tick.new_ask - tick.old_ask)
        
        # Return the maximum price difference
        return max(bid_diff, ask_diff)
    
    def _estimate_missed_volume(self, fire: CQSFireEvent) -> int:
        """Estimate missed volume for EOC calculation."""
        # Simplified estimation based on features
        # In practice, this would be based on actual protected pegged size
        bids = fire.features.get('bids', 0)
        asks = fire.features.get('asks', 0)
        
        # Estimate missed volume as average of bid/ask venues
        return int((bids + asks) / 2) * 100  # 100 shares per venue estimate

class MetricsAnalyzer:
    """
    Analyzer for KPI metrics across multiple simulation runs.
    
    Calculates confidence intervals and statistical measures.
    """
    
    def __init__(self):
        self.runs = []
    
    def add_run(self, metrics: KPIMetrics, run_id: str = None):
        """Add a simulation run's metrics."""
        run_data = {
            'run_id': run_id or f"run_{len(self.runs)}",
            'metrics': metrics
        }
        self.runs.append(run_data)
    
    def calculate_confidence_intervals(self, confidence: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for all metrics.
        
        Args:
            confidence: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Dictionary with confidence intervals for each metric
        """
        if not self.runs:
            return {}
        
        # Extract metrics
        vpa_values = [run['metrics'].vpa for run in self.runs]
        eoc_values = [run['metrics'].eoc for run in self.runs]
        hft_pnl_values = [run['metrics'].hft_pnl for run in self.runs]
        precision_values = [run['metrics'].precision for run in self.runs]
        recall_values = [run['metrics'].recall for run in self.runs]
        f1_values = [run['metrics'].f1_score for run in self.runs]
        
        # Calculate confidence intervals
        def ci(values):
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            se = std / np.sqrt(n)
            t_val = 1.96  # Approximate for 95% CI
            margin = t_val * se
            return {
                'mean': mean,
                'std': std,
                'lower': mean - margin,
                'upper': mean + margin
            }
        
        return {
            'vpa': ci(vpa_values),
            'eoc': ci(eoc_values),
            'hft_pnl': ci(hft_pnl_values),
            'precision': ci(precision_values),
            'recall': ci(recall_values),
            'f1_score': ci(f1_values)
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all runs."""
        if not self.runs:
            return {}
        
        cis = self.calculate_confidence_intervals()
        
        return {
            'num_runs': len(self.runs),
            'confidence_intervals': cis,
            'efficient_frontier': self._calculate_efficient_frontier(),
            'hft_tipping_point': self._calculate_hft_tipping_point()
        }
    
    def _calculate_efficient_frontier(self) -> List[Dict[str, float]]:
        """Calculate efficient frontier (VPA vs EOC)."""
        frontier_points = []
        
        for run in self.runs:
            metrics = run['metrics']
            frontier_points.append({
                'vpa': metrics.vpa,
                'eoc': metrics.eoc,
                'run_id': run['run_id']
            })
        
        # Sort by EOC for frontier calculation
        frontier_points.sort(key=lambda x: x['eoc'])
        
        return frontier_points
    
    def _calculate_hft_tipping_point(self) -> Optional[float]:
        """Calculate HFT tipping point (where P&L crosses zero)."""
        # Find runs where HFT P&L is closest to zero
        zero_crossings = []
        
        for run in self.runs:
            metrics = run['metrics']
            if abs(metrics.hft_pnl) < 1.0:  # Close to zero
                zero_crossings.append(metrics.eoc)
        
        return np.mean(zero_crossings) if zero_crossings else None

if __name__ == "__main__":
    # Test the KPI calculator
    print("Testing KPI Calculator")
    print("=" * 30)
    
    from logging_system import GroundTruthLabel, CQSFireEvent, NBBOChangeEvent, HFTArbitrageEvent
    
    # Create test data
    labels = [
        GroundTruthLabel(0.1, "heuristic", "TP", 0.12, "bid_up", 5.0, 2.0),
        GroundTruthLabel(0.2, "logistic", "FP", None, None, 0.0, 1.5),
        GroundTruthLabel(0.3, "FN", "FN", 0.32, "ask_down", 3.0, 0.0)
    ]
    
    nbbo_changes = [
        NBBOChangeEvent(0.12, "nbbo_change", {}, "AAPL", 150.00, 150.01, 150.01, 150.02, 2, 3),
        NBBOChangeEvent(0.32, "nbbo_change", {}, "AAPL", 150.01, 150.02, 150.01, 150.01, 1, 2)
    ]
    
    cqs_fires = [
        CQSFireEvent(0.1, "cqs_fire", {}, "heuristic", {"bids": 3, "asks": 2}, None),
        CQSFireEvent(0.2, "cqs_fire", {}, "logistic", {"bids": 2, "asks": 1}, 0.75)
    ]
    
    hft_attempts = [
        HFTArbitrageEvent(0.15, "hft_arbitrage", {}, "HFT1", "AAPL", "B", 150.01, 100, True, 5.0),
        HFTArbitrageEvent(0.25, "hft_arbitrage", {}, "HFT1", "AAPL", "A", 150.00, 200, False, -1.0)
    ]
    
    # Test KPI calculator
    print("\n1. Testing KPI Calculator:")
    print("-" * 30)
    
    calculator = KPICalculator()
    metrics = calculator.calculate_all_metrics(labels, nbbo_changes, cqs_fires, hft_attempts)
    
    print(f"VPA: ${metrics.vpa:.2f}")
    print(f"EOC: ${metrics.eoc:.2f}")
    print(f"HFT P&L: ${metrics.hft_pnl:.2f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"F1 Score: {metrics.f1_score:.3f}")
    print(f"TPs: {metrics.tps}, FPs: {metrics.fps}, FNs: {metrics.fns}")
    
    # Test metrics analyzer
    print("\n2. Testing Metrics Analyzer:")
    print("-" * 30)
    
    analyzer = MetricsAnalyzer()
    
    # Add multiple runs
    for i in range(5):
        # Simulate different runs with some variation
        run_metrics = KPIMetrics(
            vpa=metrics.vpa + np.random.normal(0, 0.5),
            eoc=metrics.eoc + np.random.normal(0, 0.2),
            hft_pnl=metrics.hft_pnl + np.random.normal(0, 1.0),
            precision=metrics.precision + np.random.normal(0, 0.05),
            recall=metrics.recall + np.random.normal(0, 0.05),
            f1_score=metrics.f1_score + np.random.normal(0, 0.05),
            total_events=metrics.total_events,
            tps=metrics.tps,
            fps=metrics.fps,
            fns=metrics.fns
        )
        analyzer.add_run(run_metrics, f"run_{i}")
    
    # Calculate confidence intervals
    cis = analyzer.calculate_confidence_intervals()
    print(f"VPA 95% CI: ${cis['vpa']['lower']:.2f} - ${cis['vpa']['upper']:.2f}")
    print(f"EOC 95% CI: ${cis['eoc']['lower']:.2f} - ${cis['eoc']['upper']:.2f}")
    print(f"HFT P&L 95% CI: ${cis['hft_pnl']['lower']:.2f} - ${cis['hft_pnl']['upper']:.2f}")
    
    # Get summary stats
    summary = analyzer.get_summary_stats()
    print(f"Number of runs: {summary['num_runs']}")
    print(f"Efficient frontier points: {len(summary['efficient_frontier'])}")
    tipping_point = summary['hft_tipping_point']
    print(f"HFT tipping point: {tipping_point:.2f}" if tipping_point is not None else "HFT tipping point: None")
    
    print("\nKPI calculator test complete!")
