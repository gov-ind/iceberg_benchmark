# iceberg_benchmark

This repo benchmarks:

1. Iceberg reads and compares Spark reads vs PyArrow disk-based reads.
2. Iceberg writes and compares direct writes to Iceberg with indirect writes via a buffer.