from pyspark.sql import SparkSession

def get_config():
    return (SparkSession.builder.master("local")
    .appName("LANL")
    .config("spark.some.config.option", "some-value")
    .config('spark.executor.memory', '50G')
    .config('spark.driver.memory', '50G')
    .config('spark.driver.maxResultSize', '50G')
    # .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC')
    # .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC')
    # .config('spark.sql.autoBroadcastJoinThreshold' , '-1')
    .config('spark.local.dir' , '/root/tmp')
    .config('spark.executor.extraJavaOptions' , '-Djava.io.tmpdir=/root/tmp')
    .config('spark.driver.extraJavaOptions' , '-Djava.io.tmpdir=/root/tmp')
    .getOrCreate())
