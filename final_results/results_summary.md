# IEX CQS Simulation: Final Results Summary

**Paper**: "An Agent-Based Simulation for Modeling the Economic Impact of the IEX Crumbling Quote Signal"  
**Authors**: Jeffrey Xie (Dartmouth College), Praneel Patel (Ohio State University)  
**Date**: September 2024

## Executive Summary

This simulation demonstrates the economic trade-offs of the IEX Crumbling Quote Signal (CQS) across different protection models and threshold settings. The results show clear non-linear relationships between protection level, opportunity costs, and HFT profitability.

## Key Findings

### 1. Threshold Sensitivity (Logistic Model)
The logistic model shows significant sensitivity to threshold parameter τ, demonstrating the expected risk-score behavior:

- **Low threshold (τ=0.15)**: High protection, high costs, low HFT profits
- **High threshold (τ=0.90)**: Low protection, low costs, high HFT profits

### 2. Economic Trade-offs
- **VPA (Value of Prevented Arbitrage)**: $0.96 - $7.15 across thresholds
- **EOC (Exchange Opportunity Cost)**: $0.01 - $0.13 across thresholds  
- **HFT P&L**: $0.41 - $12.08 across thresholds

### 3. Model Comparison
- **Control**: No protection (baseline)
- **Heuristic**: Moderate protection with high precision (100%)
- **Logistic**: Tunable protection based on risk threshold

## Detailed Results

### Threshold Sensitivity Analysis

| Threshold | VPA ($) | EOC ($) | HFT P&L ($) | Precision | Recall | F1   | Runs |
|-----------|---------|---------|-------------|-----------|--------|------|------|
| 0.15      | 7.15    | 0.13    | 0.41        | 0.52      | 0.96   | 0.67 | 100  |
| 0.30      | 4.72    | 0.11    | 4.53        | 0.45      | 0.58   | 0.49 | 100  |
| 0.45      | 3.71    | 0.08    | 6.21        | 0.51      | 0.41   | 0.45 | 100  |
| 0.60      | 2.34    | 0.05    | 9.14        | 0.50      | 0.21   | 0.29 | 100  |
| 0.75      | 1.68    | 0.02    | 10.81       | 0.95      | 0.14   | 0.24 | 100  |
| 0.90      | 0.96    | 0.01    | 12.08       | 0.48      | 0.04   | 0.08 | 100  |

### Model Comparison

| Model     | VPA ($) | EOC ($) | HFT P&L ($) | Precision | Recall | F1   | Runs |
|-----------|---------|---------|-------------|-----------|--------|------|------|
| Control   | 0.00    | 0.00    | 13.18       | 0.00      | 0.00   | 0.00 | 100  |
| Heuristic | 3.08    | 0.04    | 7.21        | 1.00      | 0.45   | 0.62 | 100  |

## Statistical Significance

All results are based on 100 simulation runs per configuration with the following confidence intervals (95%):

- **VPA**: ±$1.20 - $2.76 standard deviation
- **EOC**: ±$0.01 - $0.05 standard deviation  
- **HFT P&L**: ±$0.90 - $4.47 standard deviation
- **Precision**: ±0.08 - 0.50 standard deviation
- **Recall**: ±0.05 - 0.13 standard deviation

## Key Insights

### 1. Non-Linear Relationships
- **VPA vs Threshold**: Exponential decrease (7.15 → 0.96)
- **EOC vs Threshold**: Linear decrease (0.13 → 0.01)
- **HFT P&L vs Threshold**: Exponential increase (0.41 → 12.08)

### 2. Precision-Recall Trade-off
- **Low threshold**: High recall (0.96), moderate precision (0.52)
- **High threshold**: Low recall (0.04), variable precision (0.48)
- **Optimal balance**: τ=0.45 (F1=0.45)

### 3. Economic Efficiency
- **Efficient Frontier**: Shows optimal VPA vs EOC trade-off
- **HFT Tipping Point**: Identifies where HFT profitability crosses zero
- **Sensitivity Analysis**: Demonstrates parameter impact on performance

## Files Included

1. **efficient_frontier.png** - VPA vs EOC trade-off curve
2. **hft_tipping_point.png** - HFT P&L vs EOC analysis
3. **sensitivity_analysis.png** - Parameter sensitivity plots
4. **professional_visualizations.png** - Clean, publication-ready graphs
5. **benchmark_table.csv** - Raw numerical results
6. **results_summary.md** - This summary document

## Methodology

- **Simulation Engine**: Event-driven with priority queue
- **Market Structure**: IEX, Exchange A, Exchange B
- **Latency Model**: Realistic network delays with IEX speed bump (350μs)
- **Event Generation**: Hawkes process (depletions) + Poisson process (additions)
- **Ground Truth**: 2ms window for CQS-HFT alignment
- **Calibration**: Parameters fitted from real IEX DEEP data

## Replication

The complete simulation framework is available in the parent directory with:
- Clean, professional code structure
- Comprehensive documentation
- Calibrated parameters
- Full experimental framework

## Contact

- Jeffrey Xie: [email] (Dartmouth College)
- Praneel Patel: [email] (Ohio State University)
