#!/usr/bin/env python3
"""
IEX DEEP Data Preprocessing with Atomic Event Bundling
Implements Section 3.1 from the LaTeX specification

Supports both direct PCAP processing and iex-to-json CLI pre-conversion pathway
"""

import json
import gzip
from datetime import datetime
from typing import Iterator, List, Dict, Any
from pathlib import Path
from iex_parser import Parser, DEEP_1_0
import subprocess
import os

def convert_pcap_to_json(pcap_file_path: str, json_file_path: str = None) -> str:
    """
    Convert PCAP file to JSON using iex-to-json CLI tool for better performance.
    
    Per LaTeX Section 3.1: "Use iex-parser's iex-to-json tool as a first step to convert 
    the raw PCAP file to a stream of JSON objects for better performance."
    
    Args:
        pcap_file_path: Path to the input .pcap.gz file
        json_file_path: Path for output .jsonl file (auto-generated if None)
        
    Returns:
        Path to the created JSON file
    """
    if json_file_path is None:
        json_file_path = pcap_file_path.replace('.pcap.gz', '.jsonl')
    
    try:
        # Check if iex-to-json command is available
        result = subprocess.run(['iex-to-json', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Use the CLI tool for conversion
            print(f"Converting {pcap_file_path} to {json_file_path} using iex-to-json...")
            result = subprocess.run(['iex-to-json', pcap_file_path, json_file_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully converted to {json_file_path}")
                return json_file_path
            else:
                print(f"iex-to-json failed: {result.stderr}")
                print("Falling back to Python parsing...")
        else:
            print("iex-to-json not available, using Python parsing...")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("iex-to-json CLI tool not found, using Python parsing...")
    
    # Fallback: convert using Python iex-parser
    print(f"Converting {pcap_file_path} to {json_file_path} using Python iex-parser...")
    _convert_pcap_to_json_python(pcap_file_path, json_file_path)
    return json_file_path

def _convert_pcap_to_json_python(pcap_file_path: str, json_file_path: str):
    """Convert PCAP to JSON using Python iex-parser as fallback."""
    from decimal import Decimal
    
    def json_serializer(obj):
        """Custom JSON serializer for special types."""
        if isinstance(obj, Decimal):
            return float(obj)
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(json_file_path, 'w') as f:
        with Parser(pcap_file_path, DEEP_1_0) as reader:
            for message in reader:
                # Convert special types for JSON serialization
                for key, value in message.items():
                    if isinstance(value, bytes):
                        message[key] = value.decode('utf-8', errors='ignore')
                    elif isinstance(value, Decimal):
                        message[key] = float(value)
                
                f.write(json.dumps(message, default=json_serializer) + '\n')

def process_atomic_events_from_json(json_file_path: str, sample_rate: int = 100, max_bundles: int = 1000, start_bundle: int = 0) -> Iterator[List[Dict[str, Any]]]:
    """
    Process atomic events from JSON file (converted from PCAP).
    
    Args:
        json_file_path: Path to the JSON lines file
        sample_rate: Take every Nth bundle (1 = all bundles)
        max_bundles: Maximum number of bundles to process
        start_bundle: Skip the first N bundles
        
    Yields:
        List of messages that form an atomic bundle
    """
    # Buffer for messages with the same timestamp
    current_bundle = []
    current_timestamp = None
    bundle_count = 0
    yielded_bundles = 0
    
    try:
        with open(json_file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                message = json.loads(line.strip())
                
                # Convert timestamp back to datetime if needed
                if isinstance(message.get('timestamp'), str):
                    try:
                        message['timestamp'] = datetime.fromisoformat(message['timestamp'].replace('Z', '+00:00'))
                    except:
                        pass  # Keep as string if conversion fails
                
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
                
                # Per LaTeX Section 3.1: Buffer messages that share the same high-precision timestamp
                # and use message flags to identify the final message in an atomic bundle
                if current_timestamp is None or message_timestamp != current_timestamp:
                    # Different timestamp - this could be start of new bundle or isolated message
                    if current_bundle:
                        # We have a pending bundle - handle gracefully by yielding the previous incomplete bundle
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
                
                # Check for event-end flag in Price Level Updates and Trade Reports
                is_atomic_message = message.get('type') in ['price_level_update', 'trade_report']
                if is_atomic_message:
                    flags = message.get('flags', 0)
                    # According to IEX DEEP spec, bit 0 (LSB) indicates "event end" 
                    # Per LaTeX Section 3.1: "Use message flags to identify the final message in an atomic bundle"
                    is_event_end = (flags & 0x1) != 0  # Check if event-end flag is set
                    
                    if is_event_end:
                        # This message completes the atomic bundle - yield it
                        bundle_count += 1
                        if bundle_count >= start_bundle and bundle_count % sample_rate == 0 and yielded_bundles < max_bundles:
                            yield current_bundle
                            yielded_bundles += 1
                        # Clear bundle for next atomic event
                        current_bundle = []
                        current_timestamp = None
                        continue
                
                # Stop if we've reached max bundles (unless unlimited)
                if max_bundles != float('inf') and yielded_bundles >= max_bundles:
                    break
        
        # Yield the final bundle if it exists
        if current_bundle and (max_bundles == float('inf') or yielded_bundles < max_bundles):
            bundle_count += 1
            if bundle_count >= start_bundle and bundle_count % sample_rate == 0:
                yield current_bundle
            
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        return

def process_atomic_events(pcap_file_path: str, sample_rate: int = 100, max_bundles: int = 1000, start_bundle: int = 0, use_json: bool = True) -> Iterator[List[Dict[str, Any]]]:
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
        use_json: If True, try to use iex-to-json CLI conversion for better performance
        
    Yields:
        List of messages that form an atomic bundle
    """
    
    # Per LaTeX Section 3.1: Use iex-to-json tool as first step for better performance
    if use_json:
        json_file_path = pcap_file_path.replace('.pcap.gz', '.jsonl')
        
        # Check if JSON file already exists and is newer than PCAP
        if (os.path.exists(json_file_path) and 
            os.path.getmtime(json_file_path) > os.path.getmtime(pcap_file_path)):
            print(f"Using existing JSON file: {json_file_path}")
        else:
            # Convert PCAP to JSON
            json_file_path = convert_pcap_to_json(pcap_file_path, json_file_path)
        
        # Process from JSON file
        yield from process_atomic_events_from_json(json_file_path, sample_rate, max_bundles, start_bundle)
        return
    
    # Fallback: process directly from PCAP (original implementation)
    print(f"Processing directly from PCAP: {pcap_file_path}")
    
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
                
                # Per LaTeX Section 3.1: Buffer messages that share the same high-precision timestamp
                # and use message flags to identify the final message in an atomic bundle
                if current_timestamp is None or message_timestamp != current_timestamp:
                    # Different timestamp - this could be start of new bundle or isolated message
                    if current_bundle:
                        # We have a pending bundle - this shouldn't happen with proper flag handling
                        # But handle gracefully by yielding the previous incomplete bundle
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
                
                # Check for event-end flag in Price Level Updates and Trade Reports
                is_atomic_message = message.get('type') in ['price_level_update', 'trade_report']
                if is_atomic_message:
                    flags = message.get('flags', 0)
                    # According to IEX DEEP spec, bit 0 (LSB) indicates "event end" 
                    # Per LaTeX Section 3.1: "Use message flags to identify the final message in an atomic bundle"
                    is_event_end = (flags & 0x1) != 0  # Check if event-end flag is set
                    
                    if is_event_end:
                        # This message completes the atomic bundle - yield it
                        bundle_count += 1
                        if bundle_count >= start_bundle and bundle_count % sample_rate == 0 and yielded_bundles < max_bundles:
                            yield current_bundle
                            yielded_bundles += 1
                        # Clear bundle for next atomic event
                        current_bundle = []
                        current_timestamp = None
                        continue
                
                # Stop if we've reached max bundles (unless unlimited)
                if max_bundles != float('inf') and yielded_bundles >= max_bundles:
                    break
        
        # Yield the final bundle if it exists
        if current_bundle and (max_bundles == float('inf') or yielded_bundles < max_bundles):
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