# options_data_downloader.py
# Description: A script to download daily historical OHLCV data for crypto options
# from Deribit or Binance using the CoinAPI.io service with the requests library.

import os
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import sqlite3
from typing import List, Dict, Optional, Tuple
import logging

from dotenv import load_dotenv

load_dotenv()

class OptionSymbolDB:
    """
    A class to manage SQLite database operations for historical option symbols metadata.
    """
    
    def __init__(self, db_path: str = "option_symbols.db"):
        """
        Initialize the database client.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize the database and create tables if they don't exist.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create option_symbols table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS option_symbols (
                        symbol_id TEXT PRIMARY KEY,
                        exchange TEXT NOT NULL,
                        underlying TEXT NOT NULL,
                        base_currency TEXT NOT NULL,
                        expiration_date TEXT NOT NULL,
                        expiration_year TEXT,
                        expiration_month TEXT,
                        strike_price TEXT,
                        option_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_symbol_id ON option_symbols(symbol_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_exchange ON option_symbols(exchange)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_underlying ON option_symbols(underlying)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_expiration ON option_symbols(expiration_date)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_expiration ON option_symbols(expiration_month)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_expiration ON option_symbols(expiration_year)
                ''')
                
                conn.commit()
                print(f"Database initialized successfully at {self.db_path}")
                
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")
            raise
    
    def insert_symbol(self, symbol_data: Dict) -> bool:
        """
        Insert a single option symbol into the database.
        
        Args:
            symbol_data (Dict): Dictionary containing symbol metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Extract data from symbol_data
                symbol_id = symbol_data.get('symbol_id', '')
                exchange = symbol_data.get('exchange', '')
                underlying = symbol_data.get('underlying', '')
                base_currency = symbol_data.get('base_currency', '')
                expiration_date = symbol_data.get('expiration_date', '')
                if expiration_date:
                    expiration_year = expiration_date[:2]
                    expiration_month = expiration_date[2:4]
                strike_price = symbol_data.get('strike_price')
                option_type = symbol_data.get('option_type', '')
                
                cursor.execute('''
                    INSERT OR REPLACE INTO option_symbols 
                    (symbol_id, exchange, underlying, base_currency, expiration_date, expiration_year, expiration_month,
                     strike_price, option_type, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol_id, exchange, underlying, base_currency, expiration_date, expiration_year,
                                expiration_month, strike_price, option_type, datetime.utcnow()))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            print(f"Error inserting symbol {symbol_data.get('symbol_id', 'unknown')}: {e}")
            return False
    
    def insert_symbols_batch(self, symbols_data: List[Dict]) -> Tuple[int, int]:
        """
        Insert multiple option symbols into the database in batch.
        
        Args:
            symbols_data (List[Dict]): List of symbol metadata dictionaries
            
        Returns:
            Tuple[int, int]: (successful_inserts, failed_inserts)
        """
        successful = 0
        failed = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for symbol_data in symbols_data:
                    try:
                        symbol_id = symbol_data.get('symbol_id', '')
                        exchange = symbol_data.get('exchange', '')
                        underlying = symbol_data.get('underlying', '')
                        base_currency = symbol_data.get('base_currency', '')
                        expiration_date = symbol_data.get('expiration_date', '')
                        if expiration_date:
                            expiration_year = expiration_date[:2]
                            expiration_month = expiration_date[2:4]
                        strike_price = str(symbol_data.get('strike_price'))
                        option_type = symbol_data.get('option_type', '')

                        cursor.execute(
                    '''
                           INSERT OR REPLACE INTO option_symbols 
                           (symbol_id, exchange, underlying, base_currency, expiration_date, expiration_year, expiration_month,
                            strike_price, option_type, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (symbol_id, exchange, underlying, base_currency, expiration_date,
                             expiration_year,
                             expiration_month, strike_price, option_type, datetime.utcnow())
                        )
                        
                        successful += 1
                        
                    except sqlite3.Error as e:
                        print(f"Error inserting symbol {symbol_data.get('symbol_id', 'unknown')}: {e}")
                        failed += 1
                
                conn.commit()
                
        except sqlite3.Error as e:
            print(f"Error in batch insert: {e}")
            failed += len(symbols_data)
            successful = 0
        
        return successful, failed
    
    def get_symbols_by_exchange(self, exchange: str) -> List[Dict]:
        """
        Get all symbols for a specific exchange.
        
        Args:
            exchange (str): Exchange identifier
            
        Returns:
            List[Dict]: List of symbol dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM option_symbols 
                    WHERE exchange = ? 
                    ORDER BY symbol_id
                ''', (exchange,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            print(f"Error querying symbols for exchange {exchange}: {e}")
            return []
    
    def get_symbols_by_underlying(self, underlying: str) -> List[Dict]:
        """
        Get all symbols for a specific underlying asset.
        
        Args:
            underlying (str): Underlying asset identifier
            
        Returns:
            List[Dict]: List of symbol dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM option_symbols 
                    WHERE underlying = ? 
                    ORDER BY symbol_id
                ''', (underlying,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            print(f"Error querying symbols for underlying {underlying}: {e}")
            return []
    
    def get_symbols_by_expiration(self, expiration_date: str) -> List[Dict]:
        """
        Get all symbols expiring on a specific date.
        
        Args:
            expiration_date (str): Expiration date in YYYYMMDD format
            
        Returns:
            List[Dict]: List of symbol dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM option_symbols 
                    WHERE expiration_date = ? 
                    ORDER BY symbol_id
                ''', (expiration_date,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            print(f"Error querying symbols for expiration {expiration_date}: {e}")
            return []
    
    def get_symbols_by_filters(self, exchange: str = None, underlying: str = None, 
                              expiration_date: str = None, expiration_year: str = None, expiration_month: str = None,
                               base_currency: str = None, option_type: str = None) -> List[Dict]:
        """
        Get symbols filtered by multiple criteria.
        
        Args:
            exchange (str, optional): Exchange identifier
            underlying (str, optional): Underlying asset identifier
            expiration_date (str, optional): Expiration date in YYYYMMDD format
            base_currency (str, optional): Base currency identifier
            
        Returns:
            List[Dict]: List of symbol dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Build dynamic query
                conditions = []
                params = []
                
                if exchange:
                    conditions.append("exchange = ?")
                    params.append(exchange)
                
                if underlying:
                    conditions.append("underlying = ?")
                    params.append(underlying)
                
                if expiration_date:
                    conditions.append("expiration_date = ?")
                    params.append(expiration_date)

                if expiration_year:
                    conditions.append("expiration_year = ?")
                    params.append(expiration_year)

                if expiration_month:
                    conditions.append("expiration_month = ?")
                    params.append(expiration_month)
                
                if base_currency:
                    conditions.append("base_currency = ?")
                    params.append(base_currency)

                if option_type:
                    conditions.append("option_type = ?")
                    params.append(option_type)
                
                query = "SELECT * FROM option_symbols"
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY symbol_id"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            print(f"Error querying symbols with filters: {e}")
            return []
    
    def get_all_symbols(self) -> List[Dict]:
        """
        Get all symbols from the database.
        
        Returns:
            List[Dict]: List of all symbol dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM option_symbols ORDER BY symbol_id')
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            print(f"Error querying all symbols: {e}")
            return []
    
    def get_symbol_count(self) -> int:
        """
        Get the total number of symbols in the database.
        
        Returns:
            int: Total number of symbols
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM option_symbols')
                return cursor.fetchone()[0]
                
        except sqlite3.Error as e:
            print(f"Error getting symbol count: {e}")
            return 0
    
    def delete_symbol(self, symbol_id: str) -> bool:
        """
        Delete a symbol from the database.
        
        Args:
            symbol_id (str): Symbol ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM option_symbols WHERE symbol_id = ?', (symbol_id,))
                conn.commit()
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            print(f"Error deleting symbol {symbol_id}: {e}")
            return False
    
    def clear_database(self) -> bool:
        """
        Clear all data from the database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM option_symbols')
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            print(f"Error clearing database: {e}")
            return False


class OptionsDataDownloader:
    """
    A class to download historical options data from CoinAPI using the requests library.
    """
    def __init__(self, api_key, db_path: str = None):
        """
        Initializes the downloader with a CoinAPI key.

        Args:
            api_key (str): Your CoinAPI API key.
            db_path (str, optional): Path to SQLite database for storing symbol metadata
        """
        if not api_key:
            raise ValueError("API key cannot be empty. Please set the COIN_API_KEY environment variable.")
        self.api_key = api_key
        self.base_url = 'https://rest.coinapi.io/v1'
        
        # Initialize database client if path provided
        self.db = OptionSymbolDB(db_path) if db_path else None
    
    def get_symbols_from_db(self, exchange: str = None, underlying: str = None, 
                           expiration_date: str = None, base_currency: str = None,
                            option_type: str = None,
                            expiration_year: str = None,
                            expiration_month: str = None) -> List[Dict]:
        """
        Get option symbols from the database using filters.
        
        Args:
            exchange (str, optional): Exchange identifier
            underlying (str, optional): Underlying asset identifier
            expiration_date (str, optional): Expiration date in YYYYMMDD format
            base_currency (str, optional): Base currency identifier
            
        Returns:
            List[Dict]: List of symbol dictionaries from database
        """
        if not self.db:
            print("Database not initialized. Please provide db_path when initializing OptionsDataDownloader.")
            return []
        
        db_list = self.db.get_symbols_by_filters(exchange=exchange, underlying=underlying, expiration_date=expiration_date,
                                              expiration_year=expiration_year, expiration_month=expiration_month,
                                              option_type=option_type, base_currency=base_currency)
        output = dict()
        for symbol in db_list:
            if not output.get(symbol.get('expiration_date')):
                output[symbol.get('expiration_date')] = []
            output[symbol.get('expiration_date')].append(symbol.get('symbol_id'))
        return output


    def get_symbol_count_from_db(self) -> int:
        """
        Get the total number of symbols in the database.
        
        Returns:
            int: Total number of symbols
        """
        if not self.db:
            print("Database not initialized. Please provide db_path when initializing OptionsDataDownloader.")
            return 0
        
        return self.db.get_symbol_count()

    def get_historical_data(self, symbol_id, start_date_str, end_date_str, output_filename):
        """
        Fetches and saves daily historical OHLCV data for a specific option symbol.

        Args:
            symbol_id (str): The full CoinAPI symbol ID for the option.
                               e.g., 'DERIBIT_OPT_BTC_USD_20251226_100000_C'
                               e.g., 'BINANCE_OPT_BTCUSDT_251226_80000_C'
            start_date_str (str): The start date in 'YYYY-MM-DD' format.
            end_date_str (str): The end date in 'YYYY-MM-DD' format.
            output_filename (str): The path to save the output JSON file.
        """
        try:
            print(f"Fetching data for {symbol_id} from {start_date_str} to {end_date_str}...")

            # Convert string dates to datetime objects in ISO 8601 format
            start_time = datetime.strptime(start_date_str, '%Y-%m-%d').isoformat()
            end_time = datetime.strptime(end_date_str, '%Y-%m-%d').isoformat()

            # Construct the API request URL
            url = f"{self.base_url}/ohlcv/{symbol_id}/history"

            # Set up parameters and headers
            params = {
                'period_id': '1DAY',
                'time_start': start_time,
                'time_end': end_time,
                'limit': 100000 # Set a high limit to ensure all data in range is fetched
            }
            headers = {
                'X-CoinAPI-Key': self.api_key
            }

            # Make the API request
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            historical_data = response.json()

            if not historical_data:
                print("No data found for the specified symbol and date range.")
                return

            print(f"Successfully fetched {len(historical_data)} data points.")

            # Save data to a JSON file
            with open(output_filename, 'w') as f:
                json.dump(historical_data, f, indent=4)

            print(f"Data saved to {output_filename}")

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"Response content: {response.text}")
            print("Please check your symbol ID, API key, and subscription plan.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_historical_option_symbols(self, exchange_id, underlying_asset, base_currency, expiration_date_filter):
        """
        Fetches all available option symbols for a given exchange, underlying asset, and expiration date.
        
        Args:
            exchange_id (str): The exchange identifier (e.g., 'DERIBIT', 'BINANCE')
            underlying_asset (str): The underlying asset (e.g., 'BTC', 'ETH', 'BTCUSDT', 'ETHUSDT')
            expiration_date_filter (str): The expiration date in 'YYYMDD' format (e.g., '251226')
            
        Returns:
            list: List of option symbol dictionaries containing symbol metadata
        """

        try:
            print(f"Fetching option symbols for {exchange_id} {underlying_asset} expiring {expiration_date_filter}...")
            

            filter_symbol = f"{exchange_id.upper()}_OPT_{underlying_asset}_{base_currency}_{expiration_date_filter}"
            limit = 10000
            page = 1
            page_data = self.get_hist_symbols_page(exchange_id, page, limit)
            filtered_symbols = []
            while isinstance(page_data, list) and len(page_data) >0:
                # Build filter based on exchange and underlying asset
                option_symbols = [symbol.get('symbol_id') for symbol in page_data if filter_symbol in symbol.get('symbol_id', '')]
                filtered_symbols.extend(option_symbols)
                page += 1
                page_data = self.get_hist_symbols_page(exchange_id, page, limit)
                print(f"Found {len(option_symbols)} option symbols out of {len(page_data)} assets in page {page}")
            return filtered_symbols
            
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            if hasattr(http_err, 'response') and http_err.response:
                print(f"Response content: {http_err.response.text}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

    def load_all_historical_option_symbols(self, exchange_id: str, db_path: str = "option_symbols.db", 
                                         save_to_db: bool = True) -> List[Dict]:
        """
        Load all historical option symbols for a given exchange and optionally save to database.
        Similar to get_historical_option_symbols but returns full symbol metadata and can save to DB.
        
        Args:
            exchange_id (str): The exchange identifier (e.g., 'DERIBIT', 'BINANCE')
            db_path (str): Path to SQLite database file
            save_to_db (bool): Whether to save symbols to database
            
        Returns:
            List[Dict]: List of option symbol dictionaries containing full metadata
        """
        try:
            print(f"Loading all historical option symbols for {exchange_id}...")
            
            # Initialize database if saving to DB
            db = None
            if save_to_db:
                db = OptionSymbolDB(db_path)
            
            limit = 1000
            page = 1
            all_symbols = []
            processed_symbols = 0
            
            # Get first page
            page_data = self.get_hist_symbols_page(exchange_id, page, limit)
            
            while isinstance(page_data, list) and len(page_data) > 0:
                # Filter for option symbols
                option_symbols = []
                for symbol in page_data:
                    symbol_id = symbol.get('symbol_id', '')
                    if '_OPT_' in symbol_id:
                        # Parse symbol metadata
                        symbol_metadata = self._parse_option_symbol(symbol_id, symbol)
                        if symbol_metadata:
                            option_symbols.append(symbol_metadata)
                
                all_symbols.extend(option_symbols)
                processed_symbols += len(page_data)
                
                print(f"Page {page}: Found {len(option_symbols)} option symbols out of {len(page_data)} total symbols")
                
                # Save to database if requested
                if save_to_db and option_symbols:
                    successful, failed = db.insert_symbols_batch(option_symbols)
                    print(f"  Saved {successful} symbols to database ({failed} failed)")
                
                # Get next page
                page += 1
                page_data = self.get_hist_symbols_page(exchange_id, page, limit)
            
            print(f"Total processed: {processed_symbols} symbols")
            print(f"Total option symbols found: {len(all_symbols)}")
            
            if save_to_db:
                total_in_db = db.get_symbol_count()
                print(f"Total symbols in database: {total_in_db}")
            
            return all_symbols
            
        except Exception as e:
            print(f"An unexpected error occurred while loading historical symbols: {e}")
            return []
    
    def _parse_option_symbol(self, symbol_id: str, symbol_data: Dict) -> Optional[Dict]:
        """
        Parse option symbol ID to extract metadata.
        
        Args:
            symbol_id (str): Full symbol ID from CoinAPI
            symbol_data (Dict): Additional symbol data from API
            
        Returns:
            Optional[Dict]: Parsed symbol metadata or None if parsing fails
        """
        try:
            # Parse symbol ID format: EXCHANGE_OPT_UNDERLYING_BASE_EXPIRATION_STRIKE_TYPE
            # Examples:
            # DERIBIT_OPT_BTC_USD_250904_4800_C
            # BINANCE_OPT_BTCUSDT_251226_80000_C
            
            parts = symbol_id.split('_')
            if len(parts) < 6 or parts[1] != 'OPT':
                return None
            
            exchange = parts[0]
            underlying = parts[2]
            base_currency = parts[3]
            expiration_date = parts[4]
            strike_price = float(parts[5]) if parts[5].replace('.', '').isdigit() else None
            option_type = parts[6] if len(parts) > 6 else ''
            
            return {
                'symbol_id': symbol_id,
                'exchange': exchange,
                'underlying': underlying,
                'base_currency': base_currency,
                'expiration_date': expiration_date,
                'strike_price': strike_price,
                'option_type': option_type,
                'raw_data': symbol_data
            }
            
        except Exception as e:
            print(f"Error parsing symbol {symbol_id}: {e}")
            return None


    def get_hist_symbols_page(self, exchange_id, page, limit):
        url = f"{self.base_url}/symbols/{exchange_id.lower()}/history"
        headers = {'Authorization': self.api_key}
        params = {'page': page, 'limit': limit}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        return response.json()

    def download_all_option_price_data(self, exchange_id, underlying_asset, expiration_date,
                                       base_currency, start_date_str, end_date_str, period_id='1DAY',
                                       output_dir=None, delay_seconds=0.1):
        """
        Downloads historical OHLCV data for all available option strikes for given parameters
        and saves each to separate files in a new directory.
        
        Args:
            exchange_id (str): The exchange identifier (e.g., 'DERIBIT', 'BINANCE')
            underlying_asset (str): The underlying asset (e.g., 'BTC', 'ETH', 'BTCUSDT')
            expiration_date (str): The expiration date in 'YYYYMMDD' format (e.g., '20251226')
            start_date_str (str): The start date in 'YYYY-MM-DD' format
            end_date_str (str): The end date in 'YYYY-MM-DD' format
            period_id (str): The period for OHLCV data (default: '1DAY')
            output_dir (str): Output directory path (default: auto-generated)
            delay_seconds (float): Delay between API calls to avoid rate limiting (default: 0.1)
            
        Returns:
            dict: Dictionary containing download results and metadata
        """
        try:
            # Get all option symbols for the given parameters
            symbols = self.get_historical_option_symbols(exchange_id=exchange_id,
                                                         underlying_asset=underlying_asset,
                                                         base_currency=base_currency,
                                                         expiration_date_filter=expiration_date)
            
            if not symbols:
                print("No option symbols found for the given parameters.")
                return {'success': False, 'downloaded_files': [], 'failed_symbols': []}
            
            # Create output directory if not specified
            if output_dir is None:
                output_dir = f"data/options/{exchange_id.lower()}_{underlying_asset.lower()}_{expiration_date}"
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            print(f"Starting download of {len(symbols)} option symbols to {output_dir}")
            
            # Convert dates to ISO format
            start_time = datetime.strptime(start_date_str, '%Y-%m-%d').isoformat()
            end_time = datetime.strptime(end_date_str, '%Y-%m-%d').isoformat()
            
            downloaded_files = []
            failed_symbols = []
            
            for i, symbol in enumerate(symbols, 1):
                symbol_id = symbol['symbol_id']
                print(f"[{i}/{len(symbols)}] Downloading {symbol_id}...")
                
                try:
                    # Construct API request
                    url = f"{self.base_url}/ohlcv/{symbol_id}/history"
                    params = {
                        'period_id': period_id,
                        'time_start': start_time,
                        'time_end': end_time,
                        'limit': 100000
                    }
                    headers = {'X-CoinAPI-Key': self.api_key}
                    
                    response = requests.get(url, headers=headers, params=params)
                    response.raise_for_status()
                    
                    historical_data = response.json()
                    
                    if historical_data:
                        # Save data to file
                        output_filename = Path(output_dir) / f"{symbol_id}.json"
                        with open(output_filename, 'w') as f:
                            json.dump({
                                'symbol_metadata': symbol,
                                'historical_data': historical_data,
                                'download_info': {
                                    'start_date': start_date_str,
                                    'end_date': end_date_str,
                                    'period_id': period_id,
                                    'download_timestamp': datetime.now().isoformat()
                                }
                            }, f, indent=2)
                        
                        downloaded_files.append(str(output_filename))
                        print(f"  ✓ Saved {len(historical_data)} data points to {output_filename}")
                    else:
                        print(f"  ⚠ No data found for {symbol_id}")
                        failed_symbols.append(symbol_id)
                
                except requests.exceptions.HTTPError as http_err:
                    print(f"  ✗ HTTP error for {symbol_id}: {http_err}")
                    failed_symbols.append(symbol_id)
                except Exception as e:
                    print(f"  ✗ Error downloading {symbol_id}: {e}")
                    failed_symbols.append(symbol_id)
                
                # Add delay to avoid rate limiting
                if delay_seconds > 0 and i < len(symbols):
                    time.sleep(delay_seconds)
            
            result = {
                'success': True,
                'output_directory': output_dir,
                'total_symbols': len(symbols),
                'downloaded_files': downloaded_files,
                'failed_symbols': failed_symbols,
                'download_summary': {
                    'successful': len(downloaded_files),
                    'failed': len(failed_symbols)
                }
            }
            
            print(f"\nDownload complete!")
            print(f"Successfully downloaded: {len(downloaded_files)} files")
            print(f"Failed downloads: {len(failed_symbols)} symbols")
            print(f"Output directory: {output_dir}")
            
            return result
            
        except Exception as e:
            print(f"An unexpected error occurred during bulk download: {e}")
            return {'success': False, 'error': str(e)}




    def load_option_data_to_dataframes(self, data_directory):
        """
        Loads all option data files from a directory into pandas DataFrames.
        
        Args:
            data_directory (str): Path to directory containing option data JSON files
            
        Returns:
            dict: Dictionary where keys are symbol IDs and values are pandas DataFrames
        """
        try:
            data_dir = Path(data_directory)
            if not data_dir.exists():
                print(f"Directory {data_directory} does not exist.")
                return {}
            
            json_files = list(data_dir.glob("*.json"))
            if not json_files:
                print(f"No JSON files found in {data_directory}")
                return {}
            
            print(f"Loading {len(json_files)} option data files from {data_directory}")
            
            dataframes = {}
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'historical_data' in data and data['historical_data']:
                        # Extract symbol ID from filename or metadata
                        symbol_id = data.get('symbol_metadata', {}).get('symbol_id')
                        if not symbol_id:
                            symbol_id = json_file.stem
                        
                        # Convert historical data to DataFrame
                        df = pd.DataFrame(data['historical_data'])
                        
                        # Convert time columns to datetime
                        if 'time_period_start' in df.columns:
                            df['time_period_start'] = pd.to_datetime(df['time_period_start'])
                        if 'time_period_end' in df.columns:
                            df['time_period_end'] = pd.to_datetime(df['time_period_end'])
                        if 'time_open' in df.columns:
                            df['time_open'] = pd.to_datetime(df['time_open'])
                        if 'time_close' in df.columns:
                            df['time_close'] = pd.to_datetime(df['time_close'])
                        
                        # Add metadata as attributes
                        df.attrs = {
                            'symbol_metadata': data.get('symbol_metadata', {}),
                            'download_info': data.get('download_info', {})
                        }
                        
                        dataframes[symbol_id] = df
                        print(f"  ✓ Loaded {len(df)} rows for {symbol_id}")
                    else:
                        print(f"  ⚠ No historical data found in {json_file}")
                        
                except Exception as e:
                    print(f"  ✗ Error loading {json_file}: {e}")
            
            print(f"\nSuccessfully loaded {len(dataframes)} DataFrames")
            return dataframes
            
        except Exception as e:
            print(f"An unexpected error occurred while loading data: {e}")
            return {}


def test_download_option_data():
    coinapi_key = os.getenv('COIN_API_KEY')

    # Initialize the downloader
    downloader = OptionsDataDownloader(api_key=coinapi_key)

    # --- --- --- --- --- --- ---
    #       EXAMPLE USAGE
    # --- --- --- --- --- --- ---

    # --- Example 1: Download all Deribit BTC options for specific expiration ---
    print("Example 1: Downloading all Deribit BTC options for expiration 20251226")
    result = downloader.download_all_option_price_data(
        exchange_id='DERIBIT',
        underlying_asset='BTC',
        expiration_date='2509',
        start_date_str='2024-01-01',
        end_date_str='2024-12-01',
        period_id='1DAY',
        delay_seconds=0.1
    )
    print(f"Download result: {result['download_summary'] if result['success'] else result}")

    print("\n" + "=" * 50 + "\n")

    # --- Example 2: Load downloaded data into DataFrames ---
    if result['success']:
        print("Example 2: Loading downloaded data into DataFrames")
        dataframes = downloader.load_option_data_to_dataframes(result['output_directory'])

        if dataframes:
            print(f"Loaded {len(dataframes)} DataFrames")
            # Show info for first DataFrame as example
            first_symbol = list(dataframes.keys())[0]
            df = dataframes[first_symbol]
            print(f"\nExample DataFrame for {first_symbol}:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['time_period_start'].min()} to {df['time_period_start'].max()}")
            print(f"Symbol metadata: {df.attrs['symbol_metadata']['symbol_id']}")

    print("\n" + "=" * 50 + "\n")

def test_load_from_file():
    coinapi_key = os.getenv('COIN_API_KEY')

    # Initialize the downloader
    downloader = OptionsDataDownloader(api_key=coinapi_key)
    print("Single option download")
    deribit_symbol = 'DERIBIT_OPT_ETH_USD_250904_4800_C'
    deribit_start = '2024-01-01'
    deribit_end = '2025-09-10'
    deribit_output_file = f"data/coinapi/{deribit_symbol}_data.json"
    downloader.get_historical_data(deribit_symbol, deribit_start, deribit_end, deribit_output_file)


def load_all_options_functionality():
    """
    Test the new database functionality for option symbols.
    """
    coinapi_key = os.getenv('COIN_API_KEY')
    exchange_id = 'BINANCE'
    if not coinapi_key:
        print("COIN_API_KEY not found in environment variables. Please set it to test database functionality.")
        return
    
    print("Testing Database Functionality")
    print("=" * 50)
    
    # Initialize downloader with database
    db_path = "data/db/binance_option_symbols.db"
    downloader = OptionsDataDownloader(api_key=coinapi_key, db_path=db_path)
    
    # Test 1: Load all historical option symbols for DERIBIT
    print(f"\n1. Loading all historical option symbols for {exchange_id}..")
    symbols = downloader.load_all_historical_option_symbols(
        exchange_id=exchange_id,
        db_path=db_path, 
        save_to_db=True
    )
    print(f"Found {len(symbols)} option symbols")
    
    # Test 2: Query symbols from database
    print("\n2. Querying symbols from database...")
    db_symbols = downloader.get_symbols_from_db(exchange=exchange_id)
    print(f"Found {len(db_symbols)} symbols in database for {exchange_id}")
    
    # Test 3: Query by underlying asset
    print("\n3. Querying symbols by underlying asset (BTC)...")
    btc_symbols = downloader.get_symbols_from_db(exchange=exchange_id, underlying='BTC')
    print(f"Found {len(btc_symbols)} BTC option symbols")
    
    # Test 4: Query by expiration date
    print("\n4. Querying symbols by expiration date...")
    if btc_symbols:
        # Get a sample expiration date from the first symbol
        sample_expiration = btc_symbols[0]['expiration_date']
        exp_symbols = downloader.get_symbols_from_db(expiration_date=sample_expiration)
        print(f"Found {len(exp_symbols)} symbols expiring on {sample_expiration}")
    
    # Test 5: Get total count
    print("\n5. Getting total symbol count...")
    total_count = downloader.get_symbol_count_from_db()
    print(f"Total symbols in database: {total_count}")
    
    # Test 6: Show sample symbol data
    print("\n6. Sample symbol data:")
    if db_symbols:
        sample_symbol = db_symbols[0]
        print(f"Symbol ID: {sample_symbol['symbol_id']}")
        print(f"Exchange: {sample_symbol['exchange']}")
        print(f"Underlying: {sample_symbol['underlying']}")
        print(f"Base Currency: {sample_symbol['base_currency']}")
        print(f"Expiration: {sample_symbol['expiration_date']}")
        print(f"Strike: {sample_symbol['strike_price']}")
        print(f"Type: {sample_symbol['option_type']}")
    
    print("\nDatabase functionality test completed!")
    print(f"Database file: {db_path}")
    
    # # Clean up test database
    # try:
    #     os.remove(db_path)
    #     print(f"Test database {db_path} cleaned up.")
    # except OSError:
    #     print(f"Could not remove test database {db_path}")




def test_option_price_data_download():
    # --- Example 2: Single option download (original functionality) ---
    print("Running original functionality test...")
    downloader = OptionsDataDownloader(api_key=coinapi_key)
    option_ids = downloader.get_historical_option_symbols(
        exchange_id='DERIBIT',
        underlying_asset='BTC',
        base_currency="USDC",
        expiration_date_filter='2509'
    )
    print(f"Option IDs: {option_ids}")

    print("\n" + "=" * 80 + "\n")

    # --- Example 3: Load all historical symbols with database ---
    print("Loading all historical option symbols with database...")
    downloader_with_db = OptionsDataDownloader(api_key=coinapi_key, db_path="option_symbols.db")
    all_symbols = downloader_with_db.load_all_historical_option_symbols(
        exchange_id='DERIBIT',
        db_path="option_symbols.db",
        save_to_db=True
    )
    print(f"Total symbols loaded: {len(all_symbols)}")

    # Query some examples from database
    btc_symbols = downloader_with_db.get_symbols_from_db(exchange='DERIBIT', underlying='BTC')
    print(f"BTC symbols in database: {len(btc_symbols)}")

    if btc_symbols:
        print("Sample BTC symbol:")
        sample = btc_symbols[0]
        print(f"  {sample['symbol_id']} - Strike: {sample['strike_price']}, Exp: {sample['expiration_date']}")


if __name__ == '__main__':
    # Get API Key from environment variable
    coinapi_key = os.getenv('COIN_API_KEY')

    if not coinapi_key:
        print("COIN_API_KEY not found in environment variables. Please set it to run examples.")
        exit(1)

    # --- Example 1: Test database functionality ---
    print("Running database functionality test...")
    load_all_options_functionality()
    print("\n" + "=" * 80 + "\n")

    # db_path = "data/db/deribit_option_symbols.db"
    # downloader = OptionsDataDownloader(api_key=coinapi_key, db_path=db_path)
    # btc_symbols = downloader.get_symbols_from_db(exchange='DERIBIT', underlying='BTC', expiration_year='25', expiration_month='08', option_type='C')
    # print(f"BTC symbols in database: {len(btc_symbols)}")
    #
    # if btc_symbols:
    #     print("Sample BTC symbol:")
    #     sample = btc_symbols[0]
    #     print(f"  {sample['symbol_id']} - Strike: {sample['strike_price']}, Exp: {sample['expiration_date']}")
