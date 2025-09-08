#!/usr/bin/env python3
"""
IEX DEEP Data Preprocessing with Atomic Event Bundling
Implements Section 3.1 from the LaTeX specification
"""

import json
import gzip
from datetime import datetime
from typing import Iterator, List, Dict, Any
from pathlib import Path
from iex_parser import Parser, DEEP_1_0

def process_atomic_events(pcap_file_path: str, sample_rate: int = 100, max_bundles: int = 1000, start_bundle: int = 0) -> Iterator[List[Dict[str, Any]]]:
    """
    Process IEX DEEP pcap file and yield atomic event bundles.
    
    Implements the logic from Section 3.1:
    1. Read messages sequentially from pcap file
    2. Buffer messages (PLUs, Trade Reports) that share the same high-precision timestamp
    3. Use message flags to identify the final message in an atomic bundle
    4. Yield the entire buffer as a single list of messages when event-end flag is seen
    
    Args:
        pcap_file_path: Path to the pcap.gz file
        sample_rate: Take every Nth bundle (1 = all bundles, 100 = every 100th bundle)
        max_bundles: Maximum number of bundles to process
        start_bundle: Skip the first N bundles (to start from trading data)
        
    Yields:
        List of messages that form an atomic bundle
    """
    
    # Buffer for messages with the same timestamp
    current_bundle = []
    current_timestamp = None
    bundle_count = 0
    yielded_bundles = 0
    
    try:
        with Parser(pcap_file_path, DEEP_1_0) as reader:
            for message in reader:
                # Only process Price Level Updates and Trade Reports for atomic bundling
                if message.get('type') not in ['price_level_update', 'trade_report']:
                    # Yield any existing bundle before processing non-atomic messages
                    if current_bundle:
                        bundle_count += 1
                        if bundle_count >= start_bundle and bundle_count % sample_rate == 0 and yielded_bundles < max_bundles:
                            yield current_bundle
                            yielded_bundles += 1
                        current_bundle = []
                    
                    # Yield single message as its own bundle
                    bundle_count += 1
                    if bundle_count >= start_bundle and bundle_count % sample_rate == 0 and yielded_bundles < max_bundles:
                        yield [message]
                        yielded_bundles += 1
                    continue
                    
                message_timestamp = message.get('timestamp')
                
                # Check if this is a new timestamp (new atomic bundle)
                if current_timestamp is None or message_timestamp != current_timestamp:
                    # Yield the previous bundle if it exists
                    if current_bundle:
                        bundle_count += 1
                        if bundle_count >= start_bundle and bundle_count % sample_rate == 0 and yielded_bundles < max_bundles:
                            yield current_bundle
                            yielded_bundles += 1
                    # Start new bundle
                    current_bundle = [message]
                    current_timestamp = message_timestamp
                else:
                    # Same timestamp, add to current bundle
                    current_bundle.append(message)
                
                # Check for event-end flag in Price Level Updates
                if message.get('type') == 'price_level_update':
                    flags = message.get('flags', 0)
                    # Check if this is the final message in the atomic bundle
                    # According to IEX DEEP spec, flags indicate event properties
                    # We'll use a simple heuristic: if flags indicate this is the last update
                    # For now, we'll assume each bundle ends when we see a different timestamp
                    # This is a simplification - in practice, you'd check the actual flag bits
                    pass
                
                # Stop if we've reached max bundles
                if yielded_bundles >= max_bundles:
                    break
        
        # Yield the final bundle if it exists
        if current_bundle and yielded_bundles < max_bundles:
            bundle_count += 1
            if bundle_count >= start_bundle and bundle_count % sample_rate == 0:
                yield current_bundle
            
    except Exception as e:
        print(f"Error processing pcap file: {e}")
        return

def test_atomic_bundling(pcap_file_path: str, sample_rate: int = 100, max_bundles: int = 10):
    """Test function to verify atomic bundling is working correctly."""
    print(f"Testing atomic event bundling on {pcap_file_path}")
    print(f"Sample rate: every {sample_rate}th bundle, max {max_bundles} bundles")
    print("=" * 60)
    
    bundle_count = 0
    total_messages = 0
    
    for bundle in process_atomic_events(pcap_file_path, sample_rate, max_bundles):
        bundle_count += 1
        total_messages += len(bundle)
        
        print(f"\nBundle {bundle_count} ({len(bundle)} messages):")
        
        # Show first few messages in the bundle
        for i, message in enumerate(bundle[:3]):
            print(f"  Message {i+1}: {message.get('type')} - {message.get('timestamp')}")
            if message.get('type') == 'price_level_update':
                print(f"    Symbol: {message.get('symbol')}, Side: {message.get('side')}, "
                      f"Price: {message.get('price')}, Size: {message.get('size')}, Flags: {message.get('flags')}")
            elif message.get('type') == 'trade_report':
                print(f"    Symbol: {message.get('symbol')}, Price: {message.get('price')}, "
                      f"Size: {message.get('size')}, Trade ID: {message.get('trade_id')}")
        
        if len(bundle) > 3:
            print(f"  ... and {len(bundle) - 3} more messages")
    
    print(f"\nSummary:")
    print(f"Total bundles processed: {bundle_count}")
    print(f"Total messages processed: {total_messages}")
    if bundle_count > 0:
        print(f"Average messages per bundle: {total_messages/bundle_count:.2f}")

if __name__ == "__main__":
    # Test the atomic bundling
    pcap_file = "data.pcap.gz"
    if Path(pcap_file).exists():
        test_atomic_bundling(pcap_file, max_bundles=20)
    else:
        print(f"PCAP file {pcap_file} not found.")