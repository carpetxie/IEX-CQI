#!/usr/bin/env python3
"""
Book Reconstruction Script
Implements Checkpoint 1.2 from the LaTeX specification

Iterates through atomic bundles and maintains order book state.
Per Section 3.1, apply trades *before* applying the final price level updates in the bundle.
Outputs time-series CSV book_states.csv with columns:
[timestamp, best_bid_price, best_bid_size, best_ask_price, best_ask_size]
"""

import csv
from typing import Dict, List, Any, Iterator
from datetime import datetime
from decimal import Decimal
from preprocess import process_atomic_events
from orderbook import OrderBook

def reconstruct_book_states(pcap_file_path: str, output_file: str = "book_states.csv", 
                          sample_rate: int = 1, max_bundles: int = None) -> None:
    """
    Reconstruct order book states from IEX DEEP data.
    
    Args:
        pcap_file_path: Path to the .pcap.gz file
        output_file: Output CSV file path
        sample_rate: Sampling rate (1 = all data)
        max_bundles: Maximum number of bundles to process (None = all)
    """
    print("=" * 60)
    print("RECONSTRUCTING ORDER BOOK STATES")
    print("=" * 60)
    print(f"Input file: {pcap_file_path}")
    print(f"Output file: {output_file}")
    
    if max_bundles is None:
        print(f"Sample rate: {sample_rate}, Processing ALL bundles")
    else:
        print(f"Sample rate: {sample_rate}, Max bundles: {max_bundles}")
    
    # Track order book states per symbol
    symbol_books = {}
    book_states = []
    
    bundle_count = 0
    processed_bundles = 0
    
    try:
        # Process with unlimited bundles if max_bundles is None
        bundle_limit = max_bundles if max_bundles is not None else float('inf')
        
        for bundle in process_atomic_events(pcap_file_path, sample_rate=sample_rate, max_bundles=bundle_limit):
            bundle_count += 1
            if bundle_count % 1000 == 0:
                print(f"Processed {bundle_count} bundles...")
            
            # Group messages by symbol
            symbol_messages = {}
            for message in bundle:
                symbol = message.get('symbol', b'').decode('utf-8') if isinstance(message.get('symbol'), bytes) else str(message.get('symbol', ''))
                if not symbol or symbol == '':
                    continue
                
                if symbol not in symbol_messages:
                    symbol_messages[symbol] = []
                symbol_messages[symbol].append(message)
            
            # Process each symbol's messages in the bundle
            for symbol, messages in symbol_messages.items():
                if symbol not in symbol_books:
                    symbol_books[symbol] = OrderBook()
                
                order_book = symbol_books[symbol]
                
                # Separate trades and PLUs - apply trades first as per spec
                trades = []
                plus = []
                final_plu = None
                
                for message in messages:
                    if message['type'] == 'trade_report':
                        trades.append(message)
                    elif message['type'] == 'price_level_update':
                        flags = message.get('flags', 0)
                        if flags == 1:  # Event end flag - this is the final PLU
                            final_plu = message
                        else:
                            plus.append(message)
                
                # Apply trades first (per Section 3.1)
                for trade in trades:
                    side = trade['side'].decode('utf-8') if isinstance(trade['side'], bytes) else str(trade['side'])
                    price = Decimal(str(trade['price']))
                    size = trade['size']
                    order_book.apply_trade(side, price, size)
                
                # Apply intermediate PLUs
                for plu in plus:
                    side = plu['side'].decode('utf-8') if isinstance(plu['side'], bytes) else str(plu['side'])
                    price = Decimal(str(plu['price']))
                    size = plu['size']
                    order_book.add_order(side, price, size)
                
                # Apply final PLU and record state
                if final_plu:
                    side = final_plu['side'].decode('utf-8') if isinstance(final_plu['side'], bytes) else str(final_plu['side'])
                    price = Decimal(str(final_plu['price']))
                    size = final_plu['size']
                    order_book.add_order(side, price, size)
                    
                    # Record book state after completing the atomic bundle
                    bbo = order_book.get_best_bid_offer()
                    if bbo[0] is not None and bbo[2] is not None:  # Valid BBO
                        book_states.append({
                            'timestamp': final_plu['timestamp'].timestamp(),
                            'symbol': symbol,
                            'best_bid_price': float(bbo[0]),
                            'best_bid_size': bbo[1],
                            'best_ask_price': float(bbo[2]),
                            'best_ask_size': bbo[3]
                        })
            
            processed_bundles += 1
            if max_bundles is not None and processed_bundles >= max_bundles:
                break
                
    except Exception as e:
        print(f"Error processing data: {e}")
        print(f"Processed {bundle_count} bundles before error")
    
    print(f"\nBook Reconstruction Complete:")
    print(f"  Bundles processed: {processed_bundles}")
    print(f"  Book states recorded: {len(book_states)}")
    print(f"  Symbols found: {len(symbol_books)}")
    
    # Save book states to CSV
    save_book_states_csv(book_states, output_file)
    
    print(f"\nBook states saved to: {output_file}")

def save_book_states_csv(book_states: List[Dict], filename: str):
    """Save book states to CSV file."""
    if not book_states:
        print("Warning: No book states to save")
        return
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'symbol', 'best_bid_price', 'best_bid_size', 'best_ask_price', 'best_ask_size'])
        
        for state in book_states:
            writer.writerow([
                state['timestamp'],
                state['symbol'],
                state['best_bid_price'],
                state['best_bid_size'],
                state['best_ask_price'],
                state['best_ask_size']
            ])
    
    print(f"Saved {len(book_states)} book states to {filename}")

if __name__ == "__main__":
    # Reconstruct book states from the full dataset
    reconstruct_book_states('data.pcap.gz', sample_rate=1, max_bundles=None)
