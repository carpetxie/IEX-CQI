# IEX Crumbling Quote Signal (CQS) Simulation

An agent-based simulation framework for modeling the economic impact of the IEX Crumbling Quote Signal, implementing the specifications from the LaTeX document.

## Overview

This simulation models a multi-venue market environment with three exchanges (IEX, A, B) and implements three CQS models:
- **Control Model**: No protection (baseline)
- **Heuristic Model**: Rule-based signal firing
- **Logistic Model**: Risk-score based signal with threshold sweep

## Key Features

- **Event-driven simulation** with Hawkes and Poisson processes
- **Multi-venue market environment** with realistic latency modeling
- **Agent-based architecture** with LP and HFT agents
- **CQS signal models** with protection mechanisms
- **Economic metrics calculation** (VPA, EOC, HFT P&L)
- **Comprehensive experimental framework** with threshold sweeps

## Architecture

### Core Components
- `simulator.py`: Event-driven simulation engine
- `market_environment.py`: Exchange and order book management
- `agents.py`: LP and HFT agent implementations
- `cqs_models.py`: CQS signal models and protection logic
- `event_generators.py`: Hawkes and Poisson event generation
- `latency_model.py`: Network latency modeling

### Analysis
- `run_experiments.py`: Experimental harness
- `analyze_results.py`: Results analysis and visualization
- `economic_metrics.py`: KPI calculations

## Usage

### Running Experiments
```python
from run_experiments import ExperimentRunner

# Create experiment runner
runner = ExperimentRunner('results/')

# Run primary experiment (threshold sweep)
results = runner.run_primary_experiment(num_runs=100)

# Run secondary experiment (sensitivity analysis)
results = runner.run_secondary_experiment(num_runs=20)
```

### Analyzing Results
```python
from analyze_results import ResultsAnalyzer

# Load and analyze results
analyzer = ResultsAnalyzer('results/')
analyzer.generate_comprehensive_analysis()
```

## Experimental Design

### Primary Experiment
- **Models**: Control, Heuristic, Logistic (6 thresholds: 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
- **Runs**: 100 per configuration
- **Metrics**: VPA, EOC, HFT P&L, Precision, Recall, F1

### Secondary Experiment
- **Parameters**: Latency regimes, pegged fractions, Hawkes parameters
- **Runs**: 20 per configuration
- **Focus**: Sensitivity analysis

## Key Metrics

- **VPA (Value of Prevented Arbitrage)**: Economic value of protected trades
- **EOC (Economic Opportunity Cost)**: Cost of missed trades due to protection
- **HFT P&L**: High-frequency trader profitability

## Results Visualization

The analysis generates three key plots:
1. **Efficient Frontier**: VPA vs EOC scatter plot across thresholds
2. **HFT Tipping Point**: HFT P&L vs EOC with threshold curve
3. **Sensitivity Analysis**: Parameter sensitivity plots

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy
- Pandas

## Installation

```bash
pip install numpy matplotlib scipy pandas
```

## References

This implementation follows the specifications in `latex.txt`, which describes the complete experimental design and methodology for evaluating IEX's Crumbling Quote Signal.
