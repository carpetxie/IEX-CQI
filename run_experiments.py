#!/usr/bin/env python3
"""
Experimental Harness
Implements Section 9 from the LaTeX specification
"""

import json
import csv
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import time
import os
from datetime import datetime

from simulator import Simulator
from market_environment import Exchange
from latency_model import LatencyModel, LatencyRegime
from event_generators import EventGeneratorManager
from agents import LPAgent, HFTAgent
from cqs_models import CQSManager, SignalAgent
from logging_system import EventLogger, GroundTruthLabeler
from economic_metrics import KPICalculator, MetricsAnalyzer, KPIMetrics
from decimal import Decimal

def load_calibrated_parameters() -> Optional[Dict[str, Any]]:
    """Load calibrated parameters from file if available."""
    try:
        with open('calibrated_parameters.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: calibrated_parameters.json not found, using defaults")
        return None
    except Exception as e:
        print(f"Warning: Error loading calibrated parameters: {e}")
        return None

class ExperimentRunner:
    """
    Main experiment runner for the CQS simulation.
    
    Implements both primary and secondary experiments as specified.
    """
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Experiment parameters
        self.primary_thresholds = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
        self.secondary_params = {
            'exchange_fee': [0.00005, 0.0001, 0.0002],  # 0.5, 1, 2 bps
            'signal_window': [0.001, 0.002, 0.005],  # 1ms, 2ms, 5ms
            'attempt_cost': [0.005, 0.01, 0.02],  # $0.005, $0.01, $0.02
            'pegged_fraction': [0.2, 0.4, 0.6],
            'latency_regime': ['NARROW', 'MEDIUM', 'WIDE'],
            'hft_reaction_delay': [0.00005, 0.0001, 0.0002]  # 50μs, 100μs, 200μs
        }
        
        # Results storage
        self.results = []
        self.analyzer = MetricsAnalyzer()
    
    def run_single_simulation(self, config: Dict[str, Any], run_id: str) -> KPIMetrics:
        """
        Run a single simulation with given configuration.
        
        Args:
            config: Simulation configuration
            run_id: Unique identifier for this run
            
        Returns:
            KPI metrics for this run
        """
        print(f"Running simulation {run_id} with config: {config}")
        
        # Create simulation environment
        sim = Simulator(max_time=config.get('max_time', 60.0))  # Increased to 60 seconds
        
        # Create latency model first
        regime = LatencyRegime[config.get('latency_regime', 'MEDIUM')]
        latency_model = LatencyModel(regime=regime, seed=config.get('seed', 42))
        
        # Agent IDs for latency routing
        agent_ids = ["LP1", "HFT1"]
        
        # Create exchanges with latency model and agent list
        iex = Exchange("IEX", sim, latency_model, agent_ids)
        exchange_a = Exchange("A", sim, latency_model, agent_ids)
        exchange_b = Exchange("B", sim, latency_model, agent_ids)
        exchanges = [iex, exchange_a, exchange_b]
        
        # Add symbols
        symbols = config.get('symbols', ["AAPL", "MSFT"])
        for symbol in symbols:
            iex.add_symbol(symbol, Decimal('0.01'))
            exchange_a.add_symbol(symbol, Decimal('0.01'))
            exchange_b.add_symbol(symbol, Decimal('0.01'))
        
# latency_model already created above
        
        # Create CQS Manager
        venues = ["IEX", "A", "B"]
        cqs_manager = CQSManager(sim, venues)
        
        # Set active model
        model_name = config.get('model', 'control')
        cqs_manager.set_active_model(model_name)
        
        # Configure logistic model threshold if needed
        if model_name == 'logistic':
            threshold = config.get('threshold', 0.5)
            cqs_manager.models['logistic'].threshold = threshold
        
        # Create logger first
        logger = EventLogger(f"{self.output_dir}/log_{run_id}.jsonl")
        
        # Create Signal Agent with logger
        signal_agent = SignalAgent(sim, venues, logger=logger)
        
        # Connect Signal Agent to CQS Manager
        signal_agent.cqs_manager = cqs_manager
        # Attach logger to CQS Manager for centralized fire logging
        cqs_manager.logger = logger
        
        # Load calibrated parameters if available
        calibrated_params = load_calibrated_parameters()
        
        # Create event generator manager with proper seed
        event_manager = EventGeneratorManager(sim, exchanges, latency_model, seed=config.get('seed', 42))
        
        # Setup Hawkes generators using calibrated params if available
        if calibrated_params:
            bid_params = calibrated_params['hawkes_params']['bid']
            ask_params = calibrated_params['hawkes_params']['ask']
        else:
            # Fallback to config or defaults
            bid_params = config.get('hawkes_bid', {'mu': 2.0, 'alpha': 1.0, 'beta': 2.0, 'mixture_proportion': 0.6})
            ask_params = config.get('hawkes_ask', {'mu': 1.8, 'alpha': 0.9, 'beta': 1.8, 'mixture_proportion': 0.5})
        
        event_manager.setup_hawkes_generators(bid_params, ask_params)
        
        # Setup Poisson generator using calibrated params if available
        if calibrated_params:
            rates = calibrated_params['poisson_params']['rates']
            size_histogram = calibrated_params['addition_histogram']
        else:
            # Fallback to config or defaults
            rates = config.get('poisson_rates', {
                "0-60": 10.0,    # 10 events per minute
                "60-120": 8.0,   # 8 events per minute
                "120-180": 12.0  # 12 events per minute
            })
            size_histogram = config.get('size_histogram', {
                "100": 10,
                "200": 15,
                "500": 5
            })
        
        event_manager.setup_poisson_generator(rates, size_histogram)
        
        # Create agents with proper seeds
        size_distribution = [100, 200, 500]
        agent_seed = config.get('seed', 42)
        lp_agent = LPAgent("LP1", sim, exchanges, latency_model, size_distribution, seed=agent_seed + 10)
        hft_reaction_delay = config.get('hft_reaction_delay', 0.0001)
        hft_agent = HFTAgent("HFT1", sim, exchanges, latency_model,
                            reaction_delay=hft_reaction_delay, seed=agent_seed + 20, logger=logger)
        
        # Register logger with simulation events
        sim.register_handler("cqs_fire", lambda event: logger.log_cqs_fire(
            event.timestamp, event.data['model'], event.data.get('features', {}),
            event.data.get('risk_score')
        ))
        
        sim.register_handler("hft_arbitrage", lambda event: logger.log_hft_arbitrage(
            event.timestamp, event.data['agent_id'], event.data['symbol'],
            event.data['side'], event.data['price'], event.data['size'],
            event.data['success'], event.data.get('pnl', 0.0)
        ))
        
        sim.register_handler("nbbo_change", lambda event: logger.log_nbbo_change(
            event.timestamp, event.data['symbol'], event.data.get('old_bid'),
            event.data.get('old_ask'), event.data.get('new_bid'), event.data.get('new_ask'),
            event.data.get('venues_at_bid', 0), event.data.get('venues_at_ask', 0)
        ))
        
        # Generate events
        max_time = config.get('max_time', 60.0)
        event_manager.generate_hawkes_events(0.0, max_time)
        event_manager.generate_poisson_events(0.0, max_time)
        
        print(f"Generated {event_manager.stats['hawkes_events']} Hawkes events")
        print(f"Generated {event_manager.stats['poisson_events']} Poisson events")
        
        # Run simulation
        sim.run(verbose=False)
        
        # Finalize logging
        logger.finalize_logging()
        
        # Calculate ground truth labels
        labeler = GroundTruthLabeler()
        labels = labeler.label_fires(logger.cqs_fires, logger.nbbo_changes)
        
        # Calculate KPIs
        calculator = KPICalculator(
            exchange_fee=config.get('exchange_fee', 0.0001),
            max_fill_size=config.get('max_fill_size', 1000),
            attempt_cost=config.get('attempt_cost', 0.01)
        )
        
        metrics = calculator.calculate_all_metrics(
            labels, logger.nbbo_changes, logger.cqs_fires, logger.hft_attempts
        )
        
        # Store results
        result = {
            'run_id': run_id,
            'config': config,
            'metrics': asdict(metrics),
            'simulation_stats': sim.get_stats(),
            'logger_stats': logger.get_stats()
        }
        
        self.results.append(result)
        self.analyzer.add_run(metrics, run_id)
        
        return metrics
    
    def run_primary_experiment(self, num_runs: int = 100) -> Dict[str, Any]:
        """
        Run primary experiment: Model comparison with threshold sweep.
        
        Args:
            num_runs: Number of runs per configuration
            
        Returns:
            Experiment results
        """
        print("=" * 60)
        print("PRIMARY EXPERIMENT: Model Comparison with Threshold Sweep")
        print("=" * 60)
        
        models = ['control', 'heuristic', 'logistic']
        results = {}
        
        for model in models:
            print(f"\nTesting model: {model}")
            print("-" * 40)
            
            model_results = []
            
            if model == 'logistic':
                # Test different thresholds
                for threshold in self.primary_thresholds:
                    print(f"  Threshold: {threshold}")
                    
                    for run in range(num_runs):
                        config = {
                            'model': model,
                            'threshold': threshold,
                            'seed': run,
                            'max_time': 10.0
                        }
                        
                        run_id = f"{model}_t{threshold}_r{run}"
                        metrics = self.run_single_simulation(config, run_id)
                        model_results.append({
                            'threshold': threshold,
                            'run': run,
                            'metrics': asdict(metrics)
                        })
            else:
                # Control and heuristic models (no threshold)
                for run in range(num_runs):
                    config = {
                        'model': model,
                        'seed': run,
                        'max_time': 10.0
                    }
                    
                    run_id = f"{model}_r{run}"
                    metrics = self.run_single_simulation(config, run_id)
                    model_results.append({
                        'threshold': None,
                        'run': run,
                        'metrics': asdict(metrics)
                    })
            
            results[model] = model_results
        
        # Save results
        self._save_experiment_results("primary_experiment", results)
        
        return results
    
    def run_secondary_experiment(self, num_runs: int = 50) -> Dict[str, Any]:
        """
        Run secondary experiment: Sensitivity analysis.
        
        Args:
            num_runs: Number of runs per configuration
            
        Returns:
            Experiment results
        """
        print("=" * 60)
        print("SECONDARY EXPERIMENT: Sensitivity Analysis")
        print("=" * 60)
        
        results = {}
        
        # Test each parameter
        for param_name, param_values in self.secondary_params.items():
            print(f"\nTesting parameter: {param_name}")
            print("-" * 40)
            
            param_results = []
            
            for value in param_values:
                print(f"  Value: {value}")
                
                for run in range(num_runs):
                    config = {
                        'model': 'logistic',
                        'threshold': 0.5,  # Fixed threshold
                        'seed': run,
                        'max_time': 10.0,
                        param_name: value
                    }
                    
                    run_id = f"{param_name}_{value}_r{run}"
                    metrics = self.run_single_simulation(config, run_id)
                    param_results.append({
                        'parameter': param_name,
                        'value': value,
                        'run': run,
                        'metrics': asdict(metrics)
                    })
            
            results[param_name] = param_results
        
        # Save results
        self._save_experiment_results("secondary_experiment", results)
        
        return results
    
    def _save_experiment_results(self, experiment_name: str, results: Dict[str, Any]):
        """Save experiment results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = f"{self.output_dir}/{experiment_name}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        csv_file = f"{self.output_dir}/{experiment_name}_{timestamp}.csv"
        self._save_csv_summary(csv_file, results)
        
        print(f"Saved results to {json_file} and {csv_file}")
    
    def _save_csv_summary(self, filename: str, results: Dict[str, Any]):
        """Save CSV summary of results."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'model', 'threshold', 'parameter', 'value', 'run',
                'vpa', 'eoc', 'hft_pnl', 'precision', 'recall', 'f1_score'
            ])
            
            for model, model_results in results.items():
                for result in model_results:
                    metrics = result['metrics']
                    writer.writerow([
                        model,
                        result.get('threshold', ''),
                        result.get('parameter', ''),
                        result.get('value', ''),
                        result.get('run', ''),
                        metrics['vpa'],
                        metrics['eoc'],
                        metrics['hft_pnl'],
                        metrics['precision'],
                        metrics['recall'],
                        metrics['f1_score']
                    ])
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        print("=" * 60)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 60)
        
        # Calculate confidence intervals
        cis = self.analyzer.calculate_confidence_intervals()
        
        # Generate efficient frontier
        frontier = self.analyzer._calculate_efficient_frontier()
        
        # Generate HFT tipping point
        tipping_point = self.analyzer._calculate_hft_tipping_point()
        
        # Create benchmark table
        benchmark = self._create_benchmark_table()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_runs': len(self.results),
            'confidence_intervals': cis,
            'efficient_frontier': frontier,
            'hft_tipping_point': tipping_point,
            'benchmark_table': benchmark
        }
        
        # Save report
        report_file = f"{self.output_dir}/analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to {report_file}")
        
        return report
    
    def _create_benchmark_table(self) -> Dict[str, Any]:
        """Create benchmark table comparing models."""
        # Group results by model
        model_groups = {}
        for result in self.results:
            model = result['config'].get('model', 'unknown')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(result['metrics'])
        
        # Calculate averages for each model
        benchmark = {}
        for model, metrics_list in model_groups.items():
            if not metrics_list:
                continue
            
            # Calculate means
            vpa_mean = np.mean([m['vpa'] for m in metrics_list])
            eoc_mean = np.mean([m['eoc'] for m in metrics_list])
            hft_pnl_mean = np.mean([m['hft_pnl'] for m in metrics_list])
            precision_mean = np.mean([m['precision'] for m in metrics_list])
            recall_mean = np.mean([m['recall'] for m in metrics_list])
            f1_mean = np.mean([m['f1_score'] for m in metrics_list])
            
            benchmark[model] = {
                'vpa': vpa_mean,
                'eoc': eoc_mean,
                'hft_pnl': hft_pnl_mean,
                'precision': precision_mean,
                'recall': recall_mean,
                'f1_score': f1_mean,
                'num_runs': len(metrics_list)
            }
        
        return benchmark

def main():
    """Main function to run experiments."""
    print("CQS Simulation Experiment Runner")
    print("=" * 50)
    
    # Create experiment runner
    runner = ExperimentRunner("experiment_results")
    
    # Run primary experiment
    print("\nStarting Primary Experiment...")
    primary_results = runner.run_primary_experiment(num_runs=100)  # As specified in LaTeX
    
    # Run secondary experiment
    print("\nStarting Secondary Experiment...")
    secondary_results = runner.run_secondary_experiment(num_runs=100)  # As specified in LaTeX
    
    # Generate analysis report
    print("\nGenerating Analysis Report...")
    report = runner.generate_analysis_report()
    
    print("\n" + "=" * 50)
    print("EXPERIMENTS COMPLETE!")
    print("=" * 50)
    print(f"Total runs: {len(runner.results)}")
    print(f"Results saved to: {runner.output_dir}")
    
    # Print summary
    print("\nBenchmark Table:")
    print("-" * 30)
    for model, metrics in report['benchmark_table'].items():
        print(f"{model:12} | VPA: ${metrics['vpa']:6.2f} | EOC: ${metrics['eoc']:6.2f} | HFT: ${metrics['hft_pnl']:6.2f}")

if __name__ == "__main__":
    main()
