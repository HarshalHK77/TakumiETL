import pandas as pd
import numpy as np
import random
import os
import fastavro
from datetime import datetime, timedelta,date
import pyarrow.parquet as pq
import pyarrow as pa
import yaml
import time
import yfinance as yf
import csv
from collections import Counter
# Load configurations from yaml
with open("configurations/configuration.yaml", "r") as file:
    config = yaml.safe_load(file)
    

# Assign values from yaml
NUM_TRANSACTIONS = config["NUM_TRANSACTIONS"]
TRADERS = [f"T{i}" for i in range(config["TRADERS_RANGE"])]
BROKERS = [f"B{i}" for i in range(config["BROKERS_RANGE"])]
PRICE_VARIATION = config["PRICE_VARIATION"]
OUTPUT_FORMAT = config["OUTPUT_FORMAT"]
TIME_DISTRIBUTION = config["TIME_DISTRIBUTION"]
INPUT_BATCH_SIZE = config["INPUT_BATCH_SIZE"]
SYMBOLS_RANGE = config["SYMBOLS_RANGE"]
VOLUME_MULTIPLIER = config["VOLUME_MULTIPLIER"]

REF_CURRENCIES = config["REF_CURRENCIES"]
REF_EXCHANGES = config["REF_EXCHANGES"]
REF_SIDES = config["REF_SIDES"]
REF_TRANSACTION_TYPES = config["REF_TRANSACTION_TYPES"]
REF_ORDER_STATUSES = config["REF_ORDER_STATUSES"]
REF_MARKET_TIMING = config["REF_MARKET_TIMING"]
REF_MICS = config["REF_MICS"]
REF_ORDER_TYPES = config["REF_ORDER_TYPES"]
NO_OF_DAYS = config["NO_OF_DAYS"]

# Ensure directories exist      
os.makedirs("reference_data", exist_ok=True)
os.makedirs("reference_market_data", exist_ok=True)
symbol_df = pd.read_csv("ref_symbol_data.csv")
symbol_df = symbol_df.sample(frac=1, random_state=None).reset_index(drop=True)


def generate_ref_market_data(date):
   
    threshold = NUM_TRANSACTIONS // SYMBOLS_RANGE
    threshold *=1.2
    
    global symbol_df
    stock_data = []
    symbol_count = 0

    # Iterate over unique random rows
    for i, row in symbol_df.iterrows():
        if symbol_count >= SYMBOLS_RANGE:
            break  # Stop once we reach the required number of symbols

        try:
            symbol = row["symbol"]      
            stock = yf.Ticker(symbol)
            history = stock.history(period="30d")

            if history.empty:
                # print(f"No historical data found for {symbol}")
                continue

            if date not in history.index.strftime("%Y-%m-%d").values:
                # print(f"No data available for {symbol} on {date}")
                continue

            day_data = history.loc[
                history.index.strftime("%Y-%m-%d") == date
            ].iloc[0]

            open_price = day_data["Open"]
            high_price = day_data["High"]
            low_price = day_data["Low"]
            close_price = day_data["Close"]
            volume = day_data["Volume"]
            adv30 = history["Volume"].mean()
            
            if volume < threshold:
                 continue  # Skip if volume is less than the threshold
            
            
            # print("Simple Volume: ",volume)
            volume*=VOLUME_MULTIPLIER
            adv30*=VOLUME_MULTIPLIER
            
            
            isin = "US5007541060"
            listing_date = row["listing_date"]
            exchange = row["exchange"]
            current_time_ns = get_current_timestamp_ns()
            stock_data.append(
                {
                    "listing_internal_id": symbol_count + 1,  # Unique index
                    "symbol": symbol,
                    "isin": isin,
                    "listing_date": listing_date,
                    "date": date,
                    "exchange": exchange,
                    "open_price": open_price,
                    "high_price": high_price,
                    "low_price": low_price,
                    "close_price": close_price,
                    "volume": volume,
                    "adv30": adv30,
                    "creation_time": current_time_ns,
                    "last_updated_time": current_time_ns,
                }
            )

            symbol_count += 1  # Increment symbol count
            print(f"Data fetched for {symbol} on {date}")

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    df_output = pd.DataFrame(stock_data)
    output_file = f"reference_market_data/ref_market_data_{date}"

    if OUTPUT_FORMAT == "parquet":
        table = pa.Table.from_pandas(df_output)
        pq.write_table(table, f"{output_file}.parquet")
    else:
        df_output.to_csv(f"{output_file}.csv", index=False)

    return df_output.set_index("symbol").to_dict(orient="index")


def generate_reference_data():
    def generate_data(ref_data, data_name, index_key):
        data_list = []
        current_timestamp = get_current_timestamp_ns()
        for item in ref_data:
            data_list.append(
                {**item, "creation_time": current_timestamp, "last_update_time": current_timestamp}
            )
        save_reference_data(data_list, data_name)
        return pd.DataFrame(data_list).set_index(index_key).to_dict(orient="index")

    return [
        generate_data(REF_MARKET_TIMING, "ref_market_timing_data", "exchange_id"),
        generate_data(REF_CURRENCIES, "ref_currency_data", "currency_id"),
        generate_data(REF_EXCHANGES, "ref_exchange_data", "exchange_id"),
        generate_data(REF_ORDER_TYPES, "ref_order_types_data", "order_type_id"),
        generate_ref_sides_data(),
        generate_ref_transaction_types_data(),
        generate_ref_order_statuses_data(),
        generate_data(REF_MICS, "ref_mics_data", "mic_id"),
        generate_data(REF_MARKET_TIMING, "ref_market_timing_data", "exchange_id")
    ]

# Function to generate ref_sides_data
def generate_ref_sides_data():
    sides_data = []
    current_timestamp = get_current_timestamp_ns()
    for item in REF_SIDES:
        sides_data.append(
            {
                "side_id": item["side_id"],
                "side": item["side"],
                "creation_time": current_timestamp,
                "last_update_time": current_timestamp,
            }
        )
    save_reference_data(sides_data, "ref_sides_data")
    return pd.DataFrame(sides_data).set_index("side_id").to_dict(orient="index")

# Function to generate ref_order_statuses_data
def generate_ref_order_statuses_data():
    ref_order_statuses = []
    current_timestamp = get_current_timestamp_ns()
    for item in REF_ORDER_STATUSES:
        ref_order_statuses.append(
            {
                "order_status_id": item["order_status_id"],
                "order_status": item["order_status"],
                "creation_time": current_timestamp,
                "last_update_time": current_timestamp,
            }
        )
    save_reference_data(ref_order_statuses, "ref_order_statuses_data")
    return (
        pd.DataFrame(ref_order_statuses)
        .set_index("order_status_id")
        .to_dict(orient="index"))

# Function to ref_transaction_types_data
def generate_ref_transaction_types_data():
    transaction_types_data = []
    current_timestamp = get_current_timestamp_ns()
    for item in REF_TRANSACTION_TYPES:
        transaction_types_data.append(
            {
                "transaction_type_id" : item["transaction_type_id"],
                "transaction_type" : item["transaction_type"],
                "creation_time": current_timestamp,
                "last_update_time": current_timestamp,
            }
        )
    save_reference_data(transaction_types_data, "ref_transaction_types_data")
    return (
        pd.DataFrame(transaction_types_data)
        .set_index("transaction_type_id")
        .to_dict(orient="index"))

def generate_input_data(NUM_TRANSACTIONS,symbol_data,mics_data,exchange_data,date):
    today = date
    start_time = datetime.strptime("09:00:00", "%H:%M:%S").replace(
        year=today.year, month=today.month, day=today.day
    )
    end_time = datetime.strptime("17:00:00", "%H:%M:%S").replace(
        year=today.year, month=today.month, day=today.day
    )
    total_seconds = int((end_time - start_time).total_seconds())

    # Distribute transactions based on time
    first_hour = int(NUM_TRANSACTIONS * TIME_DISTRIBUTION[0])
    mid_hours = int(NUM_TRANSACTIONS * TIME_DISTRIBUTION[1])
    last_hours = NUM_TRANSACTIONS - first_hour - mid_hours

    time_intervals = (
        [
            start_time + timedelta(seconds=random.randint(0, 3600))
            for _ in range(first_hour)
        ]
        + [
            start_time + timedelta(seconds=random.randint(3600, 14400))
            for _ in range(mid_hours)
        ]
        + [
            start_time + timedelta(seconds=random.randint(14400, total_seconds))
            for _ in range(last_hours)
        ]
    )

    time_intervals_ns = [int(t.timestamp() * 1e9) + random.randint(0, 999999999) for t in time_intervals]
    time_intervals_ns.sort()

    symbol_ids = np.random.choice(list(symbol_data.keys()), size=NUM_TRANSACTIONS)
    
    # Count occurrences of each symbol
    symbol_transactions_count = dict(Counter(symbol_ids))
    symbol_remaining_volume = {symbol_id: symbol_data[symbol_id]["volume"] for symbol_id in symbol_data.keys()}
    
    
    order_type = np.random.choice([order_types['order_type_name'] for order_types in REF_ORDER_TYPES], size=NUM_TRANSACTIONS)
    currency_name = np.random.choice([currency['currency_name'] for currency in REF_CURRENCIES], size=NUM_TRANSACTIONS)
    transaction_type = np.random.choice(
        [transaction_type['transaction_type'] for transaction_type in REF_TRANSACTION_TYPES],
        size=NUM_TRANSACTIONS,
        p=[transaction_type['probability'] for transaction_type in REF_TRANSACTION_TYPES]
        )
    
    order_status = np.random.choice(
        [status['order_status'] for status in REF_ORDER_STATUSES], 
        size=NUM_TRANSACTIONS, 
        p=[status['probability'] for status in REF_ORDER_STATUSES]
    )
    side = np.random.choice(
        [status['side'] for status in REF_SIDES], 
        size=NUM_TRANSACTIONS, 
        p=[status['probability'] for status in REF_SIDES]
    )
    
    
    # traders = np.random.choice(TRADERS, size=NUM_TRANSACTIONS)
    traders = np.where(np.random.rand(NUM_TRANSACTIONS) < 0.1, None, np.random.choice(TRADERS, size=NUM_TRANSACTIONS)).astype(object)
    # Create a dictionary to track remaining volume for each symbol
    brokers = np.random.choice(BROKERS, size=NUM_TRANSACTIONS)
    input_data = []
    
    last_index = 0

    os.makedirs("input_data", exist_ok=True)  # Ensure output directory exists
    os.makedirs(f"input_data/input_data_{date}", exist_ok=True)  # Ensure output directory exists
    
    
    while last_index < NUM_TRANSACTIONS:  # Process in batches
        input_data = []
        batch_end = min(last_index + INPUT_BATCH_SIZE, NUM_TRANSACTIONS)  # Stop at total limit
        
        for i in range(last_index, batch_end):
            # Generate timestamp
            seconds_part = time_intervals_ns[i] // 10**9
            nanoseconds_part = time_intervals_ns[i] % 10**9
            dt = datetime.fromtimestamp(seconds_part)
            timestamp = f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{nanoseconds_part:09d}"
            # timestamp = pd.to_datetime(timestamp_str)

            # Extract symbol details
            symbol = symbol_data.get(symbol_ids[i], None)
            if not symbol:
                continue

            symbol_id = symbol_ids[i]
            symbol_exchange = symbol["exchange"]
            symbol_adv30 = symbol["adv30"]
            high_price, low_price = symbol["high_price"], symbol["low_price"]
            open_price, close_price = symbol["open_price"], symbol["close_price"]

            # Ensure we do not allocate more than the remaining volume
            max_possible_quantity = symbol_remaining_volume.get(symbol_id, 0)
            remaining_txns = symbol_transactions_count.get(symbol_id, 1)  # Remaining transactions

            if max_possible_quantity <= 0 or remaining_txns <= 0:
                quantity = 0  # No shares left to allocate
            else:
                if remaining_txns == 1:
                    quantity = int(max_possible_quantity)  # Allocate the remaining volume in the last transaction
                else:
                    # Ensure quantity is not too small
                    min_allocation = max_possible_quantity // remaining_txns // 2  # Ensure fair distribution
                    max_allocation = max_possible_quantity // remaining_txns * 2  # Avoid small trailing values
                    quantity = int(np.random.randint(int(min_allocation), int(min(max_allocation, max_possible_quantity)) + 1))

                symbol_remaining_volume[symbol_id] -= quantity  # Deduct from remaining volume
                symbol_transactions_count[symbol_id] -= 1  # Reduce the remaining transactions

            # Simulate price
            price = open_price
            if np.random.rand() < 0.1:
                price = np.nan
            else:
                rand_val = np.random.rand()
                if rand_val < 0.3:
                    price = np.random.uniform(open_price, high_price)  # Moves towards high
                elif rand_val < 0.6:
                    price = np.random.uniform(low_price, high_price)  # Fluctuates between high & low
                else:
                    price = np.random.uniform(low_price, close_price)  # Moves towards close
            price *= np.random.choice([0.98, 1, 1.02])

            # Get exchange ID
            symbol_exchange_id = next(
                (exchange_id for exchange_id, values in exchange_data.items() if values["exchange_code"] == symbol_exchange),
                None
            )
            if not symbol_exchange_id:
                continue

            # Get MIC codes
            mic_codes = [values["mic_code"] for values in mics_data.values() if values["exchange_id"] == symbol_exchange_id]
            mic_code = random.choice(mic_codes) if mic_codes else None

            transaction_id = f"trx_{date}_{i+1}"
            file_path = f"input_data/input_data_{date}_part_{i+1}.{OUTPUT_FORMAT}"  # Single output file

            input_data.append({
                "transaction_id": transaction_id,
                "transaction_parent_id": transaction_id,
                "transaction_timestamp": timestamp,
                "transaction_type": transaction_type[i],
                "order_status": order_status[i],
                "order_type": order_type[i],
                "mic_code": mic_code,
                "exchange_code": symbol_exchange,
                "side": side[i],
                "symbol": None if symbol_ids[i] == "NULL" else symbol_ids[i],
                "isin": symbol["isin"],
                "price": price,
                "quantity": quantity,
                "adv30": symbol_adv30,
                "trader_id": traders[i],
                "broker_id": brokers[i],
                "currency_name": currency_name[i],
            })

        # Convert batch to DataFrame
        df = pd.DataFrame(input_data)
        input_data.clear()
        if not df.empty:
            file_path = f"input_data/input_data_{date}/input_data_{date}_batch_{last_index // INPUT_BATCH_SIZE + 1}.{OUTPUT_FORMAT}"  
            if OUTPUT_FORMAT == "parquet":
                table = pa.Table.from_pandas(df)
                pq.write_table(table, file_path)
            else:   
                df.to_csv(file_path, index=False)

        last_index = batch_end
        print(f"{last_index} input data generated")


def save_reference_data(data, filename):
    df = pd.DataFrame(data)
    if OUTPUT_FORMAT == "parquet":
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f"reference_data/{filename}.parquet")
    elif OUTPUT_FORMAT == "avro":
        schema = {
            "type": "record",
            "name": "RefData",
            "fields": [
                {"name": key, "type": ["string", "float", "int"]}
                for key in data[0].keys()
            ],
        }
        with open(f"reference_data/{filename}.avro", "wb") as out:
            fastavro.writer(out, schema, data)
    else:
        df.to_csv(f"reference_data/{filename}.csv", index=False)

def get_last_n_weekdays(num_batches):
    today = date.today()
    # today = datetime.strptime("2025-03-03", "%Y-%m-%d").date()
    days_count = 0
    current_date = today - timedelta(days=1)
    dates = []
    
    while days_count < num_batches:
        if current_date.weekday() < 5:  
            dates.append(current_date)
            days_count += 1
        current_date -= timedelta(days=1)
    
    return dates[::-1]

# Functions to generate current timestamp
def get_current_timestamp_ns():
    current_time_ns = time.time_ns()
    timestamp_sec = current_time_ns / 1_000_000_000
    dt = datetime.fromtimestamp(timestamp_sec)
    formatted_dt = (
        dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{current_time_ns % 1_000_000_000:09d}"
    )
    return formatted_dt

# Function to read avro data into dataframe
def read_avro_to_dataframe(file):
    with open(file, "rb") as file:
        reader = fastavro.reader(file)
        data = list(reader)  # Convert to a list of dictionaries

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    return df

# Main function to generate dataset
def generate_dataset():
    dates = get_last_n_weekdays(NO_OF_DAYS)
    ref_data = generate_reference_data()
    (
        market_timing_data,
        currency_data,
        exchange_data,
        order_types_data,
        sides_data,
        transaction_types_data,
        order_statuses_data,
        mics_data,
        timing_data
    ) = ref_data

    for i in range(NO_OF_DAYS):
        print(f"Generating batch {i + 1}/{NO_OF_DAYS}")
        print("\nFetching Reference Market Data")
        start_time_execution = time.time()
        
        symbol_data = generate_ref_market_data(dates[i].strftime('%Y-%m-%d'))
        end_time_execution = time.time()
        execution_time = (end_time_execution - start_time_execution)
        print(f"Execution Time for ref_market_data Generation: {execution_time:.6f} seconds")

        print("Reference Market Data Generated")
        print("\nGenerating Input Data")
        start_time_execution = time.time()
        generate_input_data(NUM_TRANSACTIONS,symbol_data,mics_data,exchange_data,dates[i])

        end_time_execution = time.time()
        execution_time = (end_time_execution - start_time_execution)
        print(f"Execution Time for input_data Generation: {execution_time:.6f} seconds")
        
        print("Input Data Generated")

    print("Dataset generation complete.")


if __name__ == "__main__":
    start_time_execution = time.time()

    generate_dataset()

    end_time_execution = time.time()
    execution_time = (end_time_execution - start_time_execution)  # Calculate execution time
    print(f"Execution Time: {execution_time:.6f} seconds")