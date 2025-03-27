import pyspark
from pyspark.sql import SparkSession
import pandas as pd
import random
import threading
import queue
import time
import json
import uuid
import os
import shutil
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    def __init__(self, 
                 bucket="your-bucket",
                 catalog="glue",
                 namespace="benchmark",
                 table_name="test_table",
                 n_threads=4,
                 k_rows=30,
                 batches_per_thread=5,
                 l_batch_size=10000,
                 use_buffer=False,
                 is_partitioned=True):
        self.bucket = bucket
        self.catalog = catalog
        self.namespace = namespace
        self.table_name = table_name
        self.n_threads = n_threads
        self.k_rows = k_rows
        self.batches_per_thread = batches_per_thread
        self.l_batch_size = l_batch_size
        self.use_buffer = use_buffer
        self.is_partitioned = is_partitioned

# Spark session singleton
def init_spark(bucket, temp_dir):
    # Create log directory and file
    os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
    
    # Single log file for all benchmarks
    log_file = os.path.join(temp_dir, "spark-iceberg-benchmark.log")
    
    # Create log4j.properties content
    log4j_content = f"""
# Set root logger level and appender
log4j.rootLogger=INFO, FILE, console

# Console appender
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.layout=org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%d{{yyyy-MM-dd HH:mm:ss}} %p %c: %m%n

# File appender
log4j.appender.FILE=org.apache.log4j.FileAppender
log4j.appender.FILE.File={log_file}
log4j.appender.FILE.ImmediateFlush=true
log4j.appender.FILE.Append=true
log4j.appender.FILE.layout=org.apache.log4j.PatternLayout
log4j.appender.FILE.layout.ConversionPattern=%d{{yyyy-MM-dd HH:mm:ss}} %-5p [%t] %c{{1}}: %m%n

# Set iceberg package logging to INFO
log4j.logger.org.apache.iceberg=INFO
"""
    
    # Write log4j properties file
    log4j_file = os.path.join(temp_dir, "log4j.properties")
    with open(log4j_file, "w") as f:
        f.write(log4j_content)
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Iceberg Benchmark") \
        .master("local[*]") \
        .config("spark.driver.extraJavaOptions", f"-Dlog4j.configuration=file:{log4j_file}") \
        .config("spark.executor.extraJavaOptions", f"-Dlog4j.configuration=file:{log4j_file}") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.glue", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.glue.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog") \
        .config("spark.sql.catalog.glue.warehouse", f"s3://{bucket}/warehouse") \
        .config("spark.sql.catalog.glue.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
        .config("spark.local.dir", temp_dir) \
        .config("spark.jars.packages", 
               "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.0," +
               "software.amazon.awssdk:bundle:2.23.9," +
               "software.amazon.awssdk:url-connection-client:2.23.9") \
        .getOrCreate()
        
    # Set log level
    spark.sparkContext.setLogLevel("INFO")
    
    return spark, log_file

# Generate random data with thread-specific partition
def generate_random_data(rows, thread_idx, is_partitioned):
    data = []
    
    # Each thread writes to a unique partition in partitioned mode
    partition_value = f"partition_{thread_idx}" if is_partitioned else f"partition_{random.randint(1, 5)}"
    
    for _ in range(rows):
        data.append({
            "id": str(uuid.uuid4()),
            "val": random.randint(1, 1000),
            "partition_col": partition_value
        })
    
    return data

# Direct write to Iceberg
def write_directly_to_iceberg(spark, catalog, namespace, table_name, k_rows, batches, thread_idx, is_partitioned):
    thread_id = threading.current_thread().name
    total_rows = 0
    total_time = 0
    
    for batch in range(batches):
        data = generate_random_data(k_rows, thread_idx, is_partitioned)
        pdf = pd.DataFrame(data)
        
        start_time = time.time()
        spark_df = spark.createDataFrame(pdf)
        spark_df.write.format("iceberg").mode("append").saveAsTable(f"{catalog}.{namespace}.{table_name}")
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        total_rows += len(data)
        
        logger.info(f"Thread {thread_id} (idx {thread_idx}) wrote batch {batch+1}/{batches} with {len(data)} rows in {batch_time:.2f} seconds to partition: {f'partition_{thread_idx}' if is_partitioned else 'various'}")
    
    logger.info(f"Thread {thread_id} (idx {thread_idx}) completed {batches} batches with {total_rows} total rows in {total_time:.2f} seconds")
    return total_rows

# Write to buffer
def write_to_buffer(buffer, k_rows, batches, thread_idx, is_partitioned):
    thread_id = threading.current_thread().name
    total_rows = 0
    total_time = 0
    
    for batch in range(batches):
        data = generate_random_data(k_rows, thread_idx, is_partitioned)
        
        start_time = time.time()
        for item in data:
            buffer.put(json.dumps(item))
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        total_rows += len(data)
        
        logger.info(f"Thread {thread_id} (idx {thread_idx}) added batch {batch+1}/{batches} with {len(data)} rows to buffer in {batch_time:.2f} seconds from partition: {f'partition_{thread_idx}' if is_partitioned else 'various'}")
    
    logger.info(f"Thread {thread_id} (idx {thread_idx}) completed {batches} batches with {total_rows} total rows in {total_time:.2f} seconds")
    return total_rows

# Consumer thread to process buffer
def consumer_thread(spark, buffer, catalog, namespace, table_name, l_batch_size, stop_event):
    total_processed = 0
    batch = []
    
    logger.info(f"Consumer started, batch size: {l_batch_size}")
    
    while not (stop_event.is_set() and buffer.empty()):
        try:
            item = buffer.get(timeout=1)
            batch.append(json.loads(item))
            buffer.task_done()
            
            if len(batch) >= l_batch_size:
                start_time = time.time()
                pdf = pd.DataFrame(batch)
                spark_df = spark.createDataFrame(pdf)
                spark_df.write.format("iceberg").mode("append").saveAsTable(f"{catalog}.{namespace}.{table_name}")
                end_time = time.time()
                
                logger.info(f"Consumer wrote batch of {len(batch)} rows in {end_time - start_time:.2f} seconds")
                total_processed += len(batch)
                batch = []
        except queue.Empty:
            # Short sleep to avoid CPU spinning
            time.sleep(0.1)
    
    # Process any remaining items
    if batch:
        start_time = time.time()
        pdf = pd.DataFrame(batch)
        spark_df = spark.createDataFrame(pdf)
        spark_df.write.format("iceberg").mode("append").saveAsTable(f"{catalog}.{namespace}.{table_name}")
        end_time = time.time()
        
        logger.info(f"Consumer wrote final batch of {len(batch)} rows in {end_time - start_time:.2f} seconds")
        total_processed += len(batch)
    
    logger.info(f"Consumer finished, processed {total_processed} records total")
    return total_processed

# Count retries in log file
def count_retries_in_log(log_file, start_position=0):
    """Count the number of retries in the log file starting from a given position"""
    try:
        with open(log_file, 'r') as f:
            log_text = f.read()
            
        # Look for retry patterns in logs
        retry_patterns = [
            r"Retrying.*commit", 
            r"CommitFailedException", 
            r"Retrying.*write", 
            r"Committing.*attempt",
            r"Retrying.*after.*CommitFailedException"
        ]
        
        total_retries = 0
        for pattern in retry_patterns:
            matches = re.findall(pattern, log_text)
            total_retries += len(matches)
            logger.info(f"Found {len(matches)} matches for pattern: {pattern}")
            
        logger.info(f"Total commit retries detected in this segment: {total_retries}")
        return total_retries
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return 0, start_position

# Benchmark function
def run_benchmark(config, spark, log_file):
    # Add a marker in the log file to make it easier to identify this benchmark
    logger.info(f"=" * 80)
    logger.info(f"BENCHMARK START: {'Buffered' if config.use_buffer else 'Direct'} - {'Partitioned' if config.is_partitioned else 'Non-partitioned'}")
    logger.info(f"=" * 80)
    
    # Setup environment
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {config.catalog}.{config.namespace}")
    spark.sql(f"DROP TABLE IF EXISTS {config.catalog}.{config.namespace}.{config.table_name} PURGE")
    
    # Create table with appropriate schema and partitioning
    if config.is_partitioned:
        spark.sql(f"""
            CREATE TABLE {config.catalog}.{config.namespace}.{config.table_name} (
                id STRING,
                val INT,
                partition_col STRING
            )
            USING iceberg
            PARTITIONED BY (partition_col)
        """)
    else:
        spark.sql(f"""
            CREATE TABLE {config.catalog}.{config.namespace}.{config.table_name} (
                id STRING,
                val INT,
                partition_col STRING
            )
            USING iceberg
        """)
    
    # Set the commit retry configuration
    spark.sql(f"""
        ALTER TABLE {config.catalog}.{config.namespace}.{config.table_name} 
        SET TBLPROPERTIES ('commit.retry.num-retries' = '32')
    """)
    retries_before = count_retries_in_log(log_file)
    
    logger.info(f"Table created with retry configuration: {config.catalog}.{config.namespace}.{config.table_name}")
    
    # Create buffer if needed
    buffer = queue.Queue() if config.use_buffer else None
    stop_event = threading.Event() if config.use_buffer else None
    
    total_rows = 0
    start_time = time.time()
    
    # Start consumer if using buffer
    consumer = None
    if config.use_buffer:
        consumer = threading.Thread(
            target=consumer_thread, 
            args=(spark, buffer, config.catalog, config.namespace, config.table_name, 
                  config.l_batch_size, stop_event),
            name="Consumer"
        )
        consumer.start()
    
    # Create and execute producer threads
    with ThreadPoolExecutor(max_workers=config.n_threads) as executor:
        futures = []
        
        for i in range(config.n_threads):
            if config.use_buffer:
                future = executor.submit(
                    write_to_buffer, 
                    buffer, 
                    config.k_rows, 
                    config.batches_per_thread, 
                    i,  # thread index used for partition
                    config.is_partitioned
                )
            else:
                future = executor.submit(
                    write_directly_to_iceberg, 
                    spark, 
                    config.catalog, 
                    config.namespace, 
                    config.table_name, 
                    config.k_rows, 
                    config.batches_per_thread, 
                    i,  # thread index used for partition
                    config.is_partitioned
                )
            futures.append(future)
        
        # Wait for all futures to complete and add up the results
        for future in as_completed(futures):
            try:
                total_rows += future.result()
            except Exception as e:
                logger.error(f"Thread error: {str(e)}")
    
    # Signal consumer to stop and wait for it to finish
    if config.use_buffer:
        logger.info("All producers finished, waiting for consumer to finish...")
        stop_event.set()
        consumer.join()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Make sure all logs are flushed before counting retries
    time.sleep(2)
    
    # Count retries in log file section for this benchmark
    retries_after = count_retries_in_log(log_file)
    retries = retries_after - retries_before
    
    # Query the table to see the actual row count
    row_count = spark.sql(f"SELECT COUNT(*) FROM {config.catalog}.{config.namespace}.{config.table_name}").collect()[0][0]
    
    if config.is_partitioned:
        # Get counts by partition
        partition_counts = spark.sql(f"""
            SELECT partition_col, COUNT(*) as count 
            FROM {config.catalog}.{config.namespace}.{config.table_name}
            GROUP BY partition_col
            ORDER BY partition_col
        """).collect()
        partition_info = ", ".join([f"{row['partition_col']}: {row['count']}" for row in partition_counts])
    else:
        partition_info = "Not partitioned"
    
    # Print benchmark results
    print(f"\nBenchmark completed:")
    print(f"  Mode: {'Buffered' if config.use_buffer else 'Direct'}")
    print(f"  Threads: {config.n_threads}")
    print(f"  Rows per batch: {config.k_rows}")
    print(f"  Batches per thread: {config.batches_per_thread}")
    print(f"  Total rows expected: {total_rows}")
    print(f"  Table partitioned: {config.is_partitioned}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Throughput: {total_rows / duration:.2f} rows/second")
    print(f"  Actual row count in table: {row_count}")
    print(f"  Commit retries detected: {retries}")
    print(f"  Partition distribution: {partition_info}")
    
    # Add a marker to indicate the end of this benchmark
    logger.info(f"=" * 80)
    logger.info(f"BENCHMARK END: {'Buffered' if config.use_buffer else 'Direct'} - {'Partitioned' if config.is_partitioned else 'Non-partitioned'}")
    logger.info(f"=" * 80)
    
    return {
        "duration": duration, 
        "rows": total_rows,
        "actual_rows": row_count,
        "retries": retries,
        "throughput": total_rows / duration
    }

# Main execution
def main():
    # Set your S3 bucket name here
    bucket = "test_bucket"
    
    # Create common temp directory
    temp_dir = "/tmp/spark-iceberg-benchmark"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize a single Spark session for all benchmarks
    spark, log_file = init_spark(bucket, temp_dir)
    
    # Configs to run
    configs = [
        # Test 1: Direct write, partitioned table
        Config(bucket=bucket, use_buffer=False, is_partitioned=True, batches_per_thread=5),
        
        # Test 2: Buffered write, partitioned table
        Config(bucket=bucket, use_buffer=True, is_partitioned=True, batches_per_thread=5),
        
        # Test 3: Direct write, non-partitioned table
        Config(bucket=bucket, use_buffer=False, is_partitioned=False, batches_per_thread=5),
        
        # Test 4: Buffered write, non-partitioned table
        Config(bucket=bucket, use_buffer=True, is_partitioned=False, batches_per_thread=5)
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"\n{'=' * 80}")
        print(f"Running Test {i+1}: {'Buffered' if config.use_buffer else 'Direct'} write, " +
              f"{'partitioned' if config.is_partitioned else 'non-partitioned'} table")
        print(f"{'=' * 80}")
        
        # Run the benchmark and get log position for next run
        result = run_benchmark(config, spark, log_file)
        
        results.append({
            "test": i+1,
            "mode": "Buffered" if config.use_buffer else "Direct",
            "is_partitioned": config.is_partitioned,
            "duration": result["duration"],
            "rows": result["rows"],
            "actual_rows": result["actual_rows"],
            "retries": result["retries"],
            "throughput": result["throughput"]
        })
    
    # Print summary
    print("\n\nBenchmark Summary:")
    print(f"{'Test':^5} | {'Mode':^10} | {'Partitioned':^12} | {'Duration':^10} | {'Throughput':^15} | {'Retries':^8} | {'Expected/Actual':^20}")
    print("-" * 100)
    for r in results:
        print(f"{r['test']:^5} | {r['mode']:^10} | {str(r['is_partitioned']):^12} | {r['duration']:.2f}s | {r['throughput']:.2f} rows/s | {r['retries']:^8} | {r['rows']}/{r['actual_rows']:^10}")
    
    print(f"\nLog file is available at: {log_file}")
    
    # Stop Spark session at the end
    spark.stop()

if __name__ == "__main__":
    main()