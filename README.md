# IEX Crumbling Quote Signal Simulation

An agent-based simulation framework for modeling the economic impact of the IEX Crumbling Quote Signal (CQS) on market microstructure.

## Paper Reference

**"An Agent-Based Simulation for Modeling the Economic Impact of the IEX Crumbling Quote Signal"**  
Authors: Jeffrey Xie (Dartmouth College), Praneel Patel (Ohio State University)

## Overview

This repository contains a complete implementation of the simulation described in the paper. The framework models:

- **Multi-venue market structure** with IEX, Exchange A, and Exchange B
- **Three CQS models**: Control (no protection), Heuristic (rule-based), and Logistic (risk-score based)
- **Agent-based behavior**: Liquidity providers and high-frequency traders
- **Economic metrics**: Value of Prevented Arbitrage (VPA), Exchange Opportunity Cost (EOC), and HFT Profitability

## Key Features

- Event-driven simulation with realistic latency modeling
- Hawkes process for liquidity depletion events (crumbling pressure)
- Poisson process for liquidity addition events
- IEX speed bump (350μs inbound delay) implementation
- Pegged vs. non-pegged order bucket protection logic
- Comprehensive ground truth labeling and performance metrics

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd IEX-CQS-Simulation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data** (optional - calibrated parameters included)
   ```bash
   # If you have IEX DEEP data file
   python preprocess.py data.pcap.gz
   python calibrate_from_real_data.py
   ```

## Usage

### Basic Experiment Run

```bash
# Run complete experiment suite (all models, 100 seeds each)
python run_experiments.py

# Generate analysis plots and tables
python analyze_results.py
```

### Quick Test

```bash
# Run with fewer iterations for testing
python -c "
from run_experiments import ExperimentRunner
runner = ExperimentRunner('results')
runner.run_primary_experiment(num_runs=10)
"
```

### Custom Configuration

Modify parameters in `run_experiments.py`:
- `num_runs`: Number of random seeds per configuration
- CQS thresholds: Logistic model firing thresholds
- Simulation duration: Maximum simulation time
- Market parameters: Latency regimes, pegged fractions

## Results Structure

```
experiment_results/
├── benchmark_table.csv          # Model comparison summary
├── efficient_frontier.png       # VPA vs EOC trade-off
├── hft_tipping_point.png       # HFT profitability analysis  
├── sensitivity_analysis.png     # Parameter sensitivity
└── primary_experiment_*.json   # Raw experimental data
```

## Key Metrics

- **VPA (Value of Prevented Arbitrage)**: Dollar value of stale trades prevented
- **EOC (Exchange Opportunity Cost)**: Lost fee revenue from blocked trades
- **HFT P&L**: High-frequency trader profit/loss from arbitrage attempts
- **Precision/Recall**: CQS signal prediction accuracy

## Repository Structure

### Core Simulation
- `simulator.py` - Event-driven simulation engine
- `market_environment.py` - Exchange and order book logic
- `orderbook.py` - Limit order book implementation
- `latency_model.py` - Network latency modeling

### CQS Implementation
- `cqs_models.py` - Three CQS models and signal logic
- `agents.py` - Liquidity provider and HFT agent behavior
- `event_generators.py` - Hawkes and Poisson event generation

### Data Processing
- `preprocess.py` - IEX DEEP data parser
- `calibrate_from_real_data.py` - Model parameter calibration
- `reconstruct_book.py` - Order book reconstruction

### Analysis
- `logging_system.py` - Event logging and ground truth labeling
- `economic_metrics.py` - KPI calculation (VPA, EOC, HFT P&L)
- `run_experiments.py` - Experimental framework
- `analyze_results.py` - Results visualization

## Replication

The simulation is designed for full replication of paper results:

1. **Deterministic**: Fixed random seeds ensure reproducible results
2. **Calibrated**: Parameters fitted from real IEX DEEP data
3. **Validated**: Ground truth labeling follows paper specification
4. **Comprehensive**: 95% confidence intervals from 100+ runs per configuration

## Expected Results

| Model     | VPA ($) | EOC ($) | HFT P&L ($) | Precision | Recall | F1   |
|-----------|---------|---------|-------------|-----------|--------|------|
| Control   | 0.00    | 0.00    | 4.28        | 0.00      | 0.00   | 0.00 |
| Heuristic | 0.62    | 0.01    | 3.04        | 0.79      | 0.25   | 0.38 |
| Logistic  | 0.72    | 0.02    | 1.70        | 0.59      | 0.58   | 0.52 |

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@article{xie2024iex,
  title={An Agent-Based Simulation for Modeling the Economic Impact of the IEX Crumbling Quote Signal},
  author={Xie, Jeffrey and Patel, Praneel},
  institution={Dartmouth College and Ohio State University},
  year={2024}
}
```

## Contact

- Jeffrey Xie: [email] (Dartmouth College)
- Praneel Patel: [email] (Ohio State University)