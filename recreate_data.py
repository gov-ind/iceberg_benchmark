from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

temp_dir = "/tmp/spark-iceberg-benchmark"
bucket = "test_bucket"
catalog = "glue"
namespace = "benchmark"
table_name = "test_table"

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
        
spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{namespace}")
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{namespace}.{table_name} PURGE")
# Create partitioned table
spark.sql(f"""
    CREATE TABLE {catalog}.{namespace}.{table_name} (
        id STRING,
        val INT,
        partition_col STRING
    )
    USING iceberg
    PARTITIONED BY (partition_col)
""")

df = (
    spark.range(1_000_000)
    .withColumn("id", expr("CAST(id AS STRING)"))
    .withColumn("val", expr("id % 100"))
    .withColumn("partition_col", expr("CONCAT('partition_', id % 10)"))
)

df.write.format("iceberg").mode("append").saveAsTable(f"{catalog}.{namespace}.{table_name}")
