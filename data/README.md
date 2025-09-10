# Data Directory

This directory contains calibrated parameters and sample data for the IEX CQS simulation.

## Files

- `calibrated_parameters.json` - Hawkes and Poisson parameters fitted from real IEX DEEP data
- `addition_size_histogram.json` - Distribution of liquidity addition sizes
- `depletion_size_histogram.json` - Distribution of liquidity depletion sizes  
- `data.pcap.gz` - Sample IEX DEEP data file (for calibration/testing)

## Usage

The simulation will automatically load calibrated parameters if available. To recalibrate from new data:

```bash
python preprocess.py data/your_data.pcap.gz
python calibrate_from_real_data.py
```

## Data Format

The IEX DEEP data should be in PCAP format as provided by IEX. The preprocessing pipeline extracts:
- Price Level Updates (PLUs) with atomic event bundling
- Trade Reports 
- Order book state changes
- Venue count dynamics for NBBO calculation
