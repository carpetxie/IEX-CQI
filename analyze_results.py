#!/usr/bin/env python3
"""
IEX CQS Simulation: Results Analysis and Visualization

Generates plots and analysis for CQS model comparison including
efficient frontier, HFT tipping point, and sensitivity analysis.

Authors: Jeffrey Xie (Dartmouth College), Praneel Patel (Ohio State University)
Paper: "An Agent-Based Simulation for Modeling the Economic Impact 
       of the IEX Crumbling Quote Signal"
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime

class ResultsAnalyzer:
    """
    Analyzer for experiment results with plotting capabilities.
    
    Generates the plots and analysis described in Section 10.
    """
    
    def __init__(self, results_dir: str = "experiment_results"):
        self.results_dir = results_dir
        self.primary_data = None
        self.secondary_data = None
        
    def load_results(self, primary_file: str = None, secondary_file: str = None):
        """Load experiment results from files."""
        if primary_file:
            with open(primary_file, 'r') as f:
                self.primary_data = json.load(f)
        
        if secondary_file:
            with open(secondary_file, 'r') as f:
                self.secondary_data = json.load(f)
    
    def plot_efficient_frontier(self, output_file: str = "efficient_frontier.png"):
        """
        Generate the Efficient Frontier plot (VPA vs EOC).
        
        Shows the trade-off between Value of Prevented Arbitrage and
        Exchange Opportunity Cost across different threshold settings.
        """
        if not self.primary_data:
            print("No primary experiment data loaded")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Extract data for logistic model
        logistic_data = self.primary_data.get('logistic', [])
        if not logistic_data:
            print("No logistic model data found")
            return
        
        # Group by threshold
        threshold_groups = {}
        for result in logistic_data:
            threshold = result['threshold']
            if threshold not in threshold_groups:
                threshold_groups[threshold] = []
            threshold_groups[threshold].append(result['metrics'])
        
        # Calculate means and confidence intervals
        thresholds = sorted(threshold_groups.keys())
        vpa_means = []
        eoc_means = []
        vpa_stds = []
        eoc_stds = []
        
        group_sizes = []
        for threshold in thresholds:
            metrics_list = threshold_groups[threshold]
            vpa_values = [m['vpa'] for m in metrics_list]
            eoc_values = [m['eoc'] for m in metrics_list]
            
            vpa_means.append(np.mean(vpa_values))
            eoc_means.append(np.mean(eoc_values))
            vpa_stds.append(np.std(vpa_values))
            eoc_stds.append(np.std(eoc_values))
            group_sizes.append(max(1, len(metrics_list)))
        
        # Calculate 95% confidence intervals (mean ± 1.96 × SE)
        vpa_cis = 1.96 * np.array(vpa_stds) / np.sqrt(np.array(group_sizes))
        eoc_cis = 1.96 * np.array(eoc_stds) / np.sqrt(np.array(group_sizes))
        
        # Plot efficient frontier with 95% confidence intervals
        plt.errorbar(eoc_means, vpa_means, 
                    xerr=eoc_cis, yerr=vpa_cis,
                    fmt='o-', capsize=5, capthick=2,
                    label='Logistic Model', linewidth=2, markersize=8)
        
        # Add threshold labels
        for i, threshold in enumerate(thresholds):
            plt.annotate(f'τ={threshold}', 
                        (eoc_means[i], vpa_means[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9)
        
        # Calculate and plot discrete derivative (ΔVPA/ΔEOC)
        if len(thresholds) > 1:
            deltas_vpa = np.diff(vpa_means)
            deltas_eoc = np.diff(eoc_means)
            derivatives = deltas_vpa / deltas_eoc
            
            # Find knee point (maximum derivative)
            knee_idx = np.argmax(derivatives)
            knee_eoc = eoc_means[knee_idx]
            knee_vpa = vpa_means[knee_idx]
            
            plt.plot(knee_eoc, knee_vpa, 'ro', markersize=12, 
                    label=f'Knee Point (τ={thresholds[knee_idx]})')
        
        plt.xlabel('Exchange Opportunity Cost (EOC) - $')
        plt.ylabel('Value of Prevented Arbitrage (VPA) - $')
        plt.title('Efficient Frontier: VPA vs EOC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{self.results_dir}/{output_file}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Efficient frontier plot saved to {output_file}")
    
    def plot_hft_tipping_point(self, output_file: str = "hft_tipping_point.png"):
        """
        Generate the HFT Tipping Point plot (HFT P&L vs EOC).
        
        Shows where HFT profitability crosses zero and identifies
        the tipping point for different models.
        """
        if not self.primary_data:
            print("No primary experiment data loaded")
            return
        
        plt.figure(figsize=(12, 8))
        
        # For logistic model, plot each threshold separately to find tipping point
        # LaTeX Section 10: "HFT P&L vs. EOC; report the x-intercept and its τ"
        logistic_data = self.primary_data.get('logistic', [])
        if logistic_data:
            # Group by threshold
            from collections import defaultdict
            threshold_groups = defaultdict(list)
            
            for result in logistic_data:
                threshold = result.get('threshold', 0.5)
                threshold_groups[threshold].append(result['metrics'])
            
            # Plot each threshold as a separate point
            eoc_points = []
            hft_pnl_points = []
            thresholds = []
            
            for threshold in sorted(threshold_groups.keys()):
                metrics_list = threshold_groups[threshold]
                eoc_values = [m['eoc'] for m in metrics_list]
                hft_pnl_values = [m['hft_pnl'] for m in metrics_list]
                
                eoc_mean = np.mean(eoc_values)
                hft_pnl_mean = np.mean(hft_pnl_values)
                eoc_std = np.std(eoc_values) / np.sqrt(len(eoc_values))  # Standard error
                hft_pnl_std = np.std(hft_pnl_values) / np.sqrt(len(hft_pnl_values))  # Standard error
                
                eoc_points.append(eoc_mean)
                hft_pnl_points.append(hft_pnl_mean)
                thresholds.append(threshold)
                
                # Plot each threshold point
                plt.errorbar(eoc_mean, hft_pnl_mean,
                            xerr=eoc_std, yerr=hft_pnl_std,
                            fmt='o', capsize=3, capthick=1,
                            color='green', markersize=8, alpha=0.8)
                
                # Annotate with threshold value
                plt.annotate(f'τ={threshold}', (eoc_mean, hft_pnl_mean), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            # Connect the points to show the tipping point curve
            if len(eoc_points) > 1:
                plt.plot(eoc_points, hft_pnl_points, 'g-', alpha=0.6, linewidth=2, 
                        label=f'Logistic Model (6 thresholds)')
        
        # Plot control and heuristic as single points (they don't have threshold sweeps)
        baseline_models = ['control', 'heuristic']
        baseline_colors = ['red', 'blue']
        
        for model, color in zip(baseline_models, baseline_colors):
            model_data = self.primary_data.get(model, [])
            if not model_data:
                continue
            
            # Extract EOC and HFT P&L values
            eoc_values = []
            hft_pnl_values = []
            
            for result in model_data:
                metrics = result['metrics']
                eoc_values.append(metrics['eoc'])
                hft_pnl_values.append(metrics['hft_pnl'])
            
            # Calculate means and confidence intervals
            eoc_mean = np.mean(eoc_values)
            hft_pnl_mean = np.mean(hft_pnl_values)
            eoc_std = np.std(eoc_values) / np.sqrt(len(eoc_values))  # Standard error
            hft_pnl_std = np.std(hft_pnl_values) / np.sqrt(len(hft_pnl_values))  # Standard error
            
            # Plot with error bars
            plt.errorbar(eoc_mean, hft_pnl_mean,
                        xerr=eoc_std, yerr=hft_pnl_std,
                        fmt='o', capsize=5, capthick=2,
                        label=f'{model.title()} Model',
                        color=color, markersize=12)
        
        # Add zero line
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero P&L')
        
        # Find and plot tipping points
        all_models = ['control', 'heuristic', 'logistic']
        for model in all_models:
            model_data = self.primary_data.get(model, [])
            if not model_data:
                continue
            
            eoc_values = [result['metrics']['eoc'] for result in model_data]
            hft_pnl_values = [result['metrics']['hft_pnl'] for result in model_data]
            
            # Find where P&L crosses zero (simplified)
            if len(eoc_values) > 1:
                # Linear interpolation to find zero crossing
                eoc_sorted, hft_sorted = zip(*sorted(zip(eoc_values, hft_pnl_values)))
                eoc_sorted = np.array(eoc_sorted)
                hft_sorted = np.array(hft_sorted)
                
                # Find zero crossing
                zero_crossings = np.where(np.diff(np.sign(hft_sorted)))[0]
                if len(zero_crossings) > 0:
                    # Interpolate to find exact EOC value
                    idx = zero_crossings[0]
                    x1, x2 = eoc_sorted[idx], eoc_sorted[idx + 1]
                    y1, y2 = hft_sorted[idx], hft_sorted[idx + 1]
                    tipping_eoc = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
                    
                    plt.axvline(x=tipping_eoc, color=color, linestyle=':', alpha=0.7,
                              label=f'{model.title()} Tipping Point')
        
        plt.xlabel('Exchange Opportunity Cost (EOC) - $')
        plt.ylabel('HFT Arbitrage P&L - $')
        plt.title('HFT Tipping Point: P&L vs EOC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{self.results_dir}/{output_file}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"HFT tipping point plot saved to {output_file}")
    
    def generate_benchmark_table(self, output_file: str = "benchmark_table.csv"):
        """
        Generate benchmark table comparing models.
        
        Creates a summary table with VPA, EOC, and HFT P&L for each model.
        For logistic model, shows results for each threshold separately.
        """
        if not self.primary_data:
            print("No primary experiment data loaded")
            return
        
        # Calculate statistics for each model
        benchmark_data = []
        
        for model_name, model_data in self.primary_data.items():
            if not model_data:
                continue
            
            if model_name == 'logistic':
                # Group logistic results by threshold
                from collections import defaultdict
                threshold_groups = defaultdict(list)
                
                for result in model_data:
                    threshold = result.get('threshold', 0.5)
                    threshold_groups[threshold].append(result['metrics'])
                
                # Calculate stats for each threshold
                for threshold in sorted(threshold_groups.keys()):
                    metrics_list = threshold_groups[threshold]
                    
                    vpa_values = [m['vpa'] for m in metrics_list]
                    eoc_values = [m['eoc'] for m in metrics_list]
                    hft_pnl_values = [m['hft_pnl'] for m in metrics_list]
                    precision_values = [m['precision'] for m in metrics_list]
                    recall_values = [m['recall'] for m in metrics_list]
                    f1_values = [m['f1_score'] for m in metrics_list]
                    
                    stats = {
                        'model': f'logistic_t{threshold}',
                        'threshold': threshold,
                        'num_runs': len(metrics_list),
                        'vpa_mean': np.mean(vpa_values),
                        'vpa_std': np.std(vpa_values),
                        'eoc_mean': np.mean(eoc_values),
                        'eoc_std': np.std(eoc_values),
                        'hft_pnl_mean': np.mean(hft_pnl_values),
                        'hft_pnl_std': np.std(hft_pnl_values),
                        'precision_mean': np.mean(precision_values),
                        'precision_std': np.std(precision_values),
                        'recall_mean': np.mean(recall_values),
                        'recall_std': np.std(recall_values),
                        'f1_mean': np.mean(f1_values),
                        'f1_std': np.std(f1_values)
                    }
                    
                    benchmark_data.append(stats)
            else:
                # Control and heuristic models (no threshold)
                vpa_values = [result['metrics']['vpa'] for result in model_data]
                eoc_values = [result['metrics']['eoc'] for result in model_data]
                hft_pnl_values = [result['metrics']['hft_pnl'] for result in model_data]
                precision_values = [result['metrics']['precision'] for result in model_data]
                recall_values = [result['metrics']['recall'] for result in model_data]
                f1_values = [result['metrics']['f1_score'] for result in model_data]
                
                stats = {
                    'model': model_name,
                    'threshold': None,
                    'num_runs': len(model_data),
                    'vpa_mean': np.mean(vpa_values),
                    'vpa_std': np.std(vpa_values),
                    'eoc_mean': np.mean(eoc_values),
                    'eoc_std': np.std(eoc_values),
                    'hft_pnl_mean': np.mean(hft_pnl_values),
                    'hft_pnl_std': np.std(hft_pnl_values),
                    'precision_mean': np.mean(precision_values),
                    'precision_std': np.std(precision_values),
                    'recall_mean': np.mean(recall_values),
                    'recall_std': np.std(recall_values),
                    'f1_mean': np.mean(f1_values),
                    'f1_std': np.std(f1_values)
                }
                
                benchmark_data.append(stats)
        
        # Create DataFrame
        df = pd.DataFrame(benchmark_data)
        
        # Save to CSV
        df.to_csv(f"{self.results_dir}/{output_file}", index=False)
        
        # Print formatted table
        print("\n" + "=" * 90)
        print("BENCHMARK TABLE - SHOWING THRESHOLD SENSITIVITY")
        print("=" * 90)
        print(f"{'Model':<15} {'Threshold':<10} {'Runs':<6} {'VPA ($)':<12} {'EOC ($)':<12} {'HFT P&L ($)':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 90)
        
        for _, row in df.iterrows():
            threshold_str = f"{row['threshold']:.2f}" if row['threshold'] is not None else "N/A"
            print(f"{row['model']:<15} {threshold_str:<10} {row['num_runs']:<6} "
                  f"{row['vpa_mean']:<8.2f}±{row['vpa_std']:<3.2f} "
                  f"{row['eoc_mean']:<8.2f}±{row['eoc_std']:<3.2f} "
                  f"{row['hft_pnl_mean']:<8.2f}±{row['hft_pnl_std']:<3.2f} "
                  f"{row['precision_mean']:<8.3f}±{row['precision_std']:<3.3f} "
                  f"{row['recall_mean']:<8.3f}±{row['recall_std']:<3.3f} "
                  f"{row['f1_mean']:<8.3f}±{row['f1_std']:<3.3f}")
        
        print("=" * 90)
        print(f"Benchmark table saved to {output_file}")
    
    def plot_sensitivity_analysis(self, output_file: str = "sensitivity_analysis.png"):
        """
        Generate sensitivity analysis plots for secondary experiment.
        
        Shows how different parameters affect the KPIs.
        """
        if not self.secondary_data:
            print("No secondary experiment data loaded")
            return
        
        # Create subplots for each parameter
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        param_names = list(self.secondary_data.keys())
        metrics = ['vpa', 'eoc', 'hft_pnl']
        
        for i, param_name in enumerate(param_names[:6]):  # Limit to 6 parameters
            ax = axes[i]
            param_data = self.secondary_data[param_name]
            
            # Group by parameter value
            value_groups = {}
            for result in param_data:
                value = result['value']
                if value not in value_groups:
                    value_groups[value] = []
                value_groups[value].append(result['metrics'])
            
            # Calculate means for each metric
            values = sorted(value_groups.keys())
            vpa_means = []
            eoc_means = []
            hft_pnl_means = []
            
            for value in values:
                metrics_list = value_groups[value]
                vpa_means.append(np.mean([m['vpa'] for m in metrics_list]))
                eoc_means.append(np.mean([m['eoc'] for m in metrics_list]))
                hft_pnl_means.append(np.mean([m['hft_pnl'] for m in metrics_list]))
            
            # Plot metrics
            ax.plot(values, vpa_means, 'o-', label='VPA', linewidth=2)
            ax.plot(values, eoc_means, 's-', label='EOC', linewidth=2)
            ax.plot(values, hft_pnl_means, '^-', label='HFT P&L', linewidth=2)
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('KPI Value ($)')
            ax.set_title(f'Sensitivity: {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(param_names), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/{output_file}", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sensitivity analysis plot saved to {output_file}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report with all plots and tables."""
        print("=" * 60)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 60)
        
        # Generate all plots and tables
        self.plot_efficient_frontier()
        self.plot_hft_tipping_point()
        self.generate_benchmark_table()
        self.plot_sensitivity_analysis()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Generated files:")
        print("- efficient_frontier.png")
        print("- hft_tipping_point.png")
        print("- benchmark_table.csv")
        print("- sensitivity_analysis.png")

def main():
    """Main function to run analysis."""
    print("CQS Simulation Results Analyzer")
    print("=" * 40)
    
    # Create analyzer
    analyzer = ResultsAnalyzer("experiment_results")
    
    # Look for result files (use latest)
    primary_file = None
    secondary_file = None
    
    # Find latest primary experiment file
    primary_files = []
    secondary_files = []
    
    for file in os.listdir("experiment_results"):
        if file.startswith("primary_experiment") and file.endswith(".json"):
            file_path = f"experiment_results/{file}"
            mtime = os.path.getmtime(file_path)
            primary_files.append((mtime, file_path))
        elif file.startswith("secondary_experiment") and file.endswith(".json"):
            file_path = f"experiment_results/{file}"
            mtime = os.path.getmtime(file_path)
            secondary_files.append((mtime, file_path))
    
    # Use the most recent files
    if primary_files:
        primary_file = max(primary_files, key=lambda x: x[0])[1]
        print(f"Using primary experiment file: {primary_file}")
    
    if secondary_files:
        secondary_file = max(secondary_files, key=lambda x: x[0])[1]
        print(f"Using secondary experiment file: {secondary_file}")
    
    if not primary_file:
        print("No primary experiment results found. Please run experiments first.")
        return
    
    # Load results
    analyzer.load_results(primary_file, secondary_file)
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main()
