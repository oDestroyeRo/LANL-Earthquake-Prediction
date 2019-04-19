from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql.functions import monotonically_increasing_id, row_number, lit
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
import pyspark.sql.functions as fn
import os
import pandas as pd
import shutil
from pyspark import SparkContext, SparkConf
import config

spark = config.get_config()
    
sc = SparkContext.getOrCreate()

def ventor_train():

    df = spark.read.parquet(os.path.join("datasets", "train.parquet"))
    df.createOrReplaceTempView("data")

    df_agg = sc.parallelize([])
    for i in range(10):
        print("Tr: ", i)
        sql = """
        SELECT seg, x FROM data WHERE set = 0 AND no BETWEEN {0} AND {1} ORDER BY uid
        """.format(i*15000, (i+1)*15000)
        df_temp = spark.sql(sql)
        rdd_temp = df_temp.rdd.map(lambda row:(row.seg,row.x))      \
            .map(lambda data: (data[0], [ data[1] ]))               \
            .reduceByKey(lambda a, b: a + b)                        \
            .map(lambda row: Row(seg=row[0],vx=Vectors.dense(row[1])))      
        if df_agg.count() == 0: 
            df_agg = rdd_temp.toDF(["seg","vx"+str(i)])
        else: 
            df_temp = rdd_temp.toDF(["seg0","vx"+str(i)])
            df_agg = df_agg.join(df_temp, df_agg.seg == df_temp.seg0).drop("seg0")

    df_agg.write.mode("overwrite").parquet(os.path.join("datasets", "train.vector.parquet"))

def vector_test():
    df = spark.read.parquet(os.path.join("datasets", "test.parquet"))
    df.createOrReplaceTempView("data")

    df_agg = sc.parallelize([])
    for i in range(10):
        print("Te: ", i)
        sql = """
        SELECT seg, x FROM data WHERE set = 1 AND no BETWEEN {0} AND {1} ORDER BY uid
        """.format(i*15000, (i+1)*15000)
        df_temp = spark.sql(sql)
        rdd_temp = df_temp.rdd.map(lambda row:(row.seg,row.x))      \
            .map(lambda data: (data[0], [ data[1] ]))               \
            .reduceByKey(lambda a, b: a + b)                        \
            .map(lambda row: Row(seg=row[0],vx=Vectors.dense(row[1])))      
        if df_agg.count() == 0: 
            df_agg = rdd_temp.toDF(["seg","vx"+str(i)])
        else: 
            df_temp = rdd_temp.toDF(["seg0","vx"+str(i)])
            df_agg = df_agg.join(df_temp, df_agg.seg == df_temp.seg0).drop("seg0")

    df_agg.write.mode("overwrite").parquet(os.path.join("datasets", "test.vector.parquet"))



ventor_train()
print('vector_train finish!!!!')

vector_test()
print('vector_test finish!!!!')
