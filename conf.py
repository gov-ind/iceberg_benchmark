import yaml

try:
    with open("config.yml", "r") as f:
        conf = yaml.safe_load(f.read())
        bucket = conf["bucket"]
        catalog = conf["catalog"]
        namespace = conf["namespace"]
        table_name = conf["table_name"]
except FileNotFoundError:
    bucket = "test_bucket"
    catalog = "glue"
    namespace = "benchmark"
    table_name = "test_table"