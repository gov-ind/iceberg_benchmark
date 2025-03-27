from pyspark.sql import SparkSession
import pyarrow.dataset as ds
import time
import os
import shutil
import logging

from conf import bucket, catalog, namespace, table_name

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

total_rows = 1_000_000
temp_dir = "/tmp/spark-iceberg-benchmark"
local_parquet_path = "/tmp/iceberg_data.parquet"

if __name__ == "__main__":
    results = {}
    os.makedirs(temp_dir, exist_ok=True)
    
    spark = SparkSession.builder \
        .appName("Iceberg Read Benchmark") \
        .master("local[*]") \
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

    
    partitions_df = spark.sql(f"""
        SELECT DISTINCT partition_col
        FROM {catalog}.{namespace}.{table_name}
        ORDER BY partition_col
    """)
    partitions = [row["partition_col"] for row in partitions_df.collect()]
    
    # Benchmark 1: PySpark reads
    logger.info("Starting PySpark read benchmark")
    
    start_time = time.time()
    n_rows_read = 0
    for partition in partitions:
        start = time.time()
        df = spark.sql(f"""
            SELECT * 
            FROM {catalog}.{namespace}.{table_name} 
            WHERE partition_col = '{partition}'
        """)
        n_rows_read += df.toPandas().shape[0]
        logger.info(f"Spark took {time.time() - start}s to read partition {partition}.")
        
    results["spark"] = time.time() - start_time
    
    # PyArrow benchmark
    logger.info("Starting PyArrow read benchmark")
    if os.path.exists(local_parquet_path):
        shutil.rmtree(local_parquet_path)
    os.makedirs(os.path.dirname(local_parquet_path), exist_ok=True)
    
    start_time = time.time()
    spark.sql(f"SELECT * FROM {catalog}.{namespace}.{table_name}") \
        .write.mode("overwrite").parquet(local_parquet_path)
    logger.info(f"Spark took {time.time() - start_time}s to do a full table scan.")
    
    dataset = ds.dataset(local_parquet_path, format="parquet")
    n_rows_read_pyarrow = 0
    for partition in partitions:
        filtered_table = dataset.to_table(
            filter=ds.field("partition_col") == partition
        )
        n_rows_read_pyarrow += filtered_table.to_pandas().shape[0]
    
    results["pyarrow"] = time.time() - start_time
    
    assert n_rows_read == n_rows_read_pyarrow, f"Mismatch in number of rows read by Spark and PyArrow!"
    shutil.rmtree(local_parquet_path, ignore_errors=True)
    spark.stop()
    
    print("\nSummary:")
    print(f"{'Metric':<25} | {'PySpark':<17} | {'PyArrow':<15}")
    print("-" * 70)
    print(f"{'Total time (s)':<25} | {results['spark']:.2f}s{'':<11} | {results['pyarrow']:.2f}s")
    print(f"{'Avg time per read (s)':<25} | {results['spark'] / len(partitions):.2f}s{'':<12} | {results['pyarrow'] / len(partitions):.2f}s")
    print(f"{'Avg rows/second':<25} | {n_rows_read / results['spark']:.2f} rows/s{'':<2} | {n_rows_read / results['pyarrow']:.2f} rows/s")