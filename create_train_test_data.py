from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql.functions import monotonically_increasing_id, row_number, lit
from pyspark.sql.window import Window
import pyspark.sql.functions as fn
import os
import pandas as pd
import shutil
import config

spark = config.get_config()

schema = StructType([
    StructField("x", DoubleType(), True),
    StructField("y", DoubleType(), True)])


def create_train_data():

    w1 = Window.orderBy("uid")
    w2 = Window.partitionBy("seg").orderBy("uid")
    df_train = spark.read.csv(os.path.join("datasets","train.csv"), header=True,schema=schema).withColumn(
        "uid", monotonically_increasing_id()).withColumn(
        "idx", row_number().over(w1).cast(IntegerType())).withColumn(
        "seg", fn.floor(((fn.col("idx")-1)/150000)).cast(IntegerType())).withColumn(
        "no", row_number().over(w2).cast(IntegerType())).withColumn(
        "name", fn.concat(lit("raw_"),fn.lpad(fn.col("seg"),4,"0").cast(StringType()))).withColumn(
        "set", lit(0))

    df_train.createOrReplaceTempView("data")
    df_train_f = spark.sql("""
    SELECT uid, set, seg, no, name, x, y FROM data 
    ORDER BY set, seg, no, uid
    """)

    df_train_f = df_train_f.repartition(1)
    df_train_f.write.mode("overwrite").parquet(os.path.join("datasets", "train.parquet"))

def create_test_data():
    df_train = spark.read.parquet(os.path.join("datasets", "train.parquet"))
    df_train.createOrReplaceTempView("data")
    max_id = spark.sql("""
    SELECT max(uid) as m FROM data
    """).first().m
    print("max_id", max_id)

    df_result = pd.read_csv(os.path.join("datasets", "sample_submission.csv"))
    files = list(df_result["seg_id"].values)

    schema = StructType([StructField("x", DoubleType(), True)])

    seg = 0
    for file in files:
        sep = "."
        if seg % 200 == 0: sep = "|"    
        if seg % 20 == 0: print(sep, end="", flush=True)
        seg += 1
    print("", end="\n", flush=True)

    seg = 0
    df_test = None
    for file in files:
    #     print(file)
        if seg % 20 == 0: print("|", end="", flush=True)

        w1 = Window.orderBy("uid")
        w2 = Window.partitionBy("seg").orderBy("uid")
        df_temp = spark.read.csv(os.path.join("datasets", "test", file+".csv"), header=True,schema=schema).withColumn(
            "y", lit(None).cast(DoubleType())).withColumn(
            "uid", lit(max_id+1)+monotonically_increasing_id()).withColumn(
            "idx", row_number().over(w1).cast(IntegerType())).withColumn(
            "seg", lit(seg).cast(IntegerType())).withColumn(
            "no", row_number().over(w2).cast(IntegerType())).withColumn(
            "name", (lit(file.split(".")[0])).cast(StringType())).withColumn(
            "set", lit(1))

        df_temp.createOrReplaceTempView("data")
        df_temp_f = spark.sql("""
        SELECT uid, set, seg, no, name, x, y FROM data
        ORDER BY set, seg, no, uid
        """)

        max_id = spark.sql("""
        SELECT max(uid) as m FROM data
        """).first().m

        seg += 1

        if df_test == None : df_test = df_temp_f
        else: df_test = df_test.union(df_temp_f)

        # create 1 file per 20 = I had issue when processing all in one go
        if seg % 20 == 0: 
            file_name = "test_1_{:04}.parquet".format(seg)
            df_test = df_test.repartition(1)
            df_test.write.parquet(os.path.join("datasets", file_name))
            df_test = None
    #     if seg == 4 : break 

    print("(", end="", flush=True)
    # left under 20 batch
    if df_test != None : 
        file_name = "test_1_{:04}.parquet".format(seg)
        df_test = df_test.repartition(1)
        df_test.write.parquet(os.path.join("datasets", file_name))
        df_test = None
    print("x)", end="\n", flush=True)
        
    print("max_id", max_id)


    df_result = pd.read_csv(os.path.join("datasets", "sample_submission.csv"))
    files = list(df_result["seg_id"].values)

    seg = 0
    for file in files:
        sep = "."
        if seg % 200 == 0: sep = "|"    
        if seg % 20 == 0: print(sep, end="", flush=True)
        seg += 1
    print("", end="\n", flush=True)

    seg = 0
    mode = "overwrite"
    for file in files:
        if seg % 20 == 0: print("|", end="", flush=True)
        seg += 1

        if seg % 20 == 0: 
            file_name = "test_1_{:04}.parquet".format(seg)
            df_test = spark.read.parquet(os.path.join("datasets", file_name))
            df_test.write.mode(mode).parquet(os.path.join("datasets", "test.parquet"))
            mode = "append"

    print("(", end="", flush=True)

    # left under 20 batch
    if seg % 20 != 0 : 
        file_name = "test_1_{:04}.parquet".format(seg)
        df_test = spark.read.parquet(os.path.join("datasets", file_name))
        df_test.write.mode(mode).parquet(os.path.join("datasets", "test.parquet"))
        mode = "append"

    print("x", end="", flush=True)

    print(")", end="\n", flush=True)


    seg = 0
    for file in files:
        sep = "."
        if seg % 200 == 0: sep = "|"    
        if seg % 20 == 0: print(sep, end="", flush=True)
        seg += 1
    print("", end="\n", flush=True)

    seg = 0
    for file in files:
        if seg % 20 == 0: print("|", end="", flush=True)
        seg += 1

        if seg % 20 == 0: 
            file_name = "test_1_{:04}.parquet".format(seg)
            shutil.rmtree(os.path.join("datasets", file_name))

    print("(", end="", flush=True)
    # left under 20 batch
    if seg % 20 != 0 : 
        file_name = "test_1_{:04}.parquet".format(seg)
        shutil.rmtree(os.path.join("datasets", file_name))
    print("x", end="", flush=True)
    print(")", end="\n", flush=True)


create_train_data()
print("create train data success!!!!!")
create_test_data()
print("create test data success!!!!!")


