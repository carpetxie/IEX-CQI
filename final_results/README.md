# IEX CQS Simulation: Final Results

This folder contains the complete results from the IEX Crumbling Quote Signal simulation study.

## 📊 Visualizations

### Core Analysis Plots
- **`efficient_frontier.png`** - VPA vs EOC trade-off curve showing optimal threshold selection
- **`hft_tipping_point.png`** - HFT P&L vs EOC analysis identifying profitability break-even points
- **`sensitivity_analysis.png`** - Parameter sensitivity plots for secondary experiment
- **`professional_visualizations.png`** - Clean, publication-ready multi-panel analysis

### Graph Descriptions
1. **Efficient Frontier**: Shows the fundamental trade-off between protection (VPA) and costs (EOC)
2. **HFT Tipping Point**: Identifies where HFT arbitrage becomes unprofitable
3. **Sensitivity Analysis**: Demonstrates how different parameters affect performance
4. **Professional Visualizations**: Clean, academic-style plots ready for publication

## 📈 Data Files

### Raw Results
- **`benchmark_table.csv`** - Complete benchmark table with all model comparisons
- **`detailed_results.csv`** - Extended results with 95% confidence intervals
- **`raw_results.json`** - Comprehensive JSON file with metadata and full results

### Summary Documents
- **`results_summary.md`** - Executive summary with key findings and insights
- **`README.md`** - This file

## 🔍 Key Findings

### Threshold Sensitivity (Logistic Model)
- **τ=0.15**: High protection (VPA=$7.15), high costs (EOC=$0.13), low HFT profits ($0.41)
- **τ=0.90**: Low protection (VPA=$0.96), low costs (EOC=$0.01), high HFT profits ($12.08)

### Model Comparison
- **Control**: No protection (baseline)
- **Heuristic**: Moderate protection with 100% precision
- **Logistic**: Tunable protection based on risk threshold

### Economic Trade-offs
- **Non-linear relationships** between threshold and economic outcomes
- **Precision-recall trade-off** following classic ROC behavior
- **Efficient frontier** showing optimal threshold selection

## 📋 Usage

### For Paper Writing
- Use `professional_visualizations.png` for main figures
- Reference `results_summary.md` for key findings
- Use `detailed_results.csv` for statistical analysis

### For Further Analysis
- Load `raw_results.json` for programmatic access
- Use `detailed_results.csv` for statistical calculations
- Reference `benchmark_table.csv` for model comparisons

### For Presentations
- Use individual PNG files for specific analyses
- Reference `results_summary.md` for talking points
- Use confidence intervals from `detailed_results.csv`

## 🎯 Results Highlights

1. **Threshold Sensitivity**: Logistic model shows significant sensitivity to threshold parameter
2. **Non-Linear Behavior**: VPA and HFT P&L show exponential relationships with threshold
3. **Economic Efficiency**: Clear trade-offs between protection level and opportunity costs
4. **Statistical Significance**: All differences between configurations are statistically significant (100 runs each)

## 📊 Data Quality

- **Sample Size**: 100 simulation runs per configuration
- **Confidence Level**: 95% confidence intervals provided
- **Statistical Power**: Sufficient for detecting meaningful differences
- **Reproducibility**: Fixed random seeds ensure reproducible results

## 🔗 Related Files

- **Parent Directory**: Complete simulation framework and source code
- **Data Directory**: Calibrated parameters and sample data
- **Documentation**: README.md and LICENSE files

---

**Paper**: "An Agent-Based Simulation for Modeling the Economic Impact of the IEX Crumbling Quote Signal"  
**Authors**: Jeffrey Xie (Dartmouth College), Praneel Patel (Ohio State University)  
**Date**: September 2024
