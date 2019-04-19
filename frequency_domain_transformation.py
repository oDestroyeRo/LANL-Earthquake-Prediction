from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import DCT
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.feature import StandardScaler
import os 
import config

spark = config.get_config()

def df_train_test():
    df_train = spark.read.parquet(os.path.join("datasets", "train.vector.parquet"))
    df_test = spark.read.parquet(os.path.join("datasets", "test.vector.parquet"))
    cols = ["vx"+str(i) for i in range(10)]
    assembler = VectorAssembler(inputCols=cols, outputCol="vx_t1")
    dct = DCT(inverse=False, inputCol="vx_t1", outputCol="vx_t2")
    slicer = VectorSlicer(inputCol="vx_t2", outputCol="vx_t3", indices=[i for i in range(40000)])
    scaler = StandardScaler(inputCol="vx_t3", outputCol="vx", withStd=True, withMean=False)

    pipeline = Pipeline(stages=[assembler, dct, slicer, scaler])
    p_model = pipeline.fit(df_train)

    df_train = p_model.transform(df_train)
    df_train = df_train.drop("vx0").drop("vx1").drop("vx2").drop("vx3").drop("vx4")
    df_train = df_train.drop("vx5").drop("vx6").drop("vx7").drop("vx8").drop("vx9")
    df_train = df_train.drop("vx_t1").drop("vx_t2").drop("vx_t3")

    df_test = p_model.transform(df_test)
    df_test = df_test.drop("vx0").drop("vx1").drop("vx2").drop("vx3").drop("vx4")
    df_test = df_test.drop("vx5").drop("vx6").drop("vx7").drop("vx8").drop("vx9")
    df_test = df_test.drop("vx_t1").drop("vx_t2").drop("vx_t3")

    df_train.write.mode("overwrite").parquet(os.path.join("datasets", "train.vector.dct.parquet"))
    df_test.write.mode("overwrite").parquet(os.path.join("datasets", "test.vector.dct.parquet"))

    df_train.printSchema()
    df_test.printSchema()

def bin_source_to(source, destination):
    df_v = spark.read.parquet(os.path.join("datasets", source))
    rdd_v = df_v.rdd.map(lambda row:(row.seg, row.vx, []))
    for i in range(0,99):
        rdd_v = rdd_v.map(lambda row, i=i:(row[0], row[1], row[2] + [float(sum(abs(row[1].toArray()[i*400:i*400+800]))/800)]))
    rdd_v = rdd_v.map(lambda row: Row(seg=row[0],f=Vectors.dense(row[2])))      
    df_v = rdd_v.toDF()
    df_v.write.mode("overwrite").parquet(os.path.join("datasets", destination))


def slice_source_to(source, destination):
    slicer = VectorSlicer(inputCol="f_t1", outputCol="f", indices=[i for i in range(50,76)])
    df_v = spark.read.parquet(os.path.join("datasets", source))
    df_v = df_v.selectExpr("*", "f AS f_t1").drop("_c0").drop("f")
    df_v = slicer.transform(df_v).drop("f_t1")
    df_v.write.mode("overwrite").parquet(os.path.join("datasets", destination))

def create_train_stats_feature():
    df_train = spark.read.parquet(os.path.join("datasets", "train.parquet"))
    df_train.createOrReplaceTempView("data")
    df_stat = spark.sql("""
    SELECT seg,
        AVG(x) AS x_avg,
        STD(x)AS x_std, 
        SKEWNESS(x) AS x_skew, 
        KURTOSIS(x) AS x_kurt, 
        MAX(x) AS x_max,
        PERCENTILE(x, 0.05) AS x_p1,
        PERCENTILE(x, 0.20) AS x_p2,
        PERCENTILE(x, 0.50) AS x_p5,
        PERCENTILE(x, 0.80) AS x_p8,
        PERCENTILE(x, 0.95) AS x_p9,
        AVG(ABS(x)) AS xa_avg,
        STD(ABS(x))AS xa_std, 
        SKEWNESS(ABS(x)) AS xa_skew, 
        KURTOSIS(ABS(x)) AS xa_kurt, 
        MAX(ABS(x)) AS xa_max,
        PERCENTILE(ABS(x), 0.05) AS xa_p1,
        PERCENTILE(ABS(x), 0.20) AS xa_p2,
        PERCENTILE(ABS(x), 0.50) AS xa_p5,
        PERCENTILE(ABS(x), 0.80) AS xa_p8,
        PERCENTILE(ABS(x), 0.95) AS xa_p9
    FROM data 
    GROUP BY seg
    ORDER BY seg
    """ )
    cols = ["x_avg", "x_std", "x_skew", "x_kurt", "x_max", "x_p1", "x_p2", "x_p5", "x_p8", "x_p9", 
            "xa_avg", "xa_std", "xa_skew", "xa_kurt", "xa_max", "xa_p1", "xa_p2", "xa_p5", "xa_p8", "xa_p9"]
    assembler = VectorAssembler(inputCols=cols, outputCol="stat_t1")
    df_stat = assembler.transform(df_stat)
    for col in cols: df_stat = df_stat.drop(col) 

    scaler = StandardScaler(inputCol="stat_t1", outputCol="stat", withStd=True, withMean=False)
    scalerModel = scaler.fit(df_stat)
    df_stat = scalerModel.transform(df_stat).drop("stat_t1")
    df_stat.write.mode("overwrite").parquet(os.path.join("datasets", "train.stat.parquet"))

    df_train = spark.read.parquet(os.path.join("datasets", "train.parquet"))
    df_train.createOrReplaceTempView("data")
    df_stat = spark.sql("""
    SELECT seg,
        AVG(x) AS x_avg,
        PERCENTILE(x, 0.01) AS x_p01,
        PERCENTILE(x, 0.02) AS x_p02,
        PERCENTILE(x, 0.05) AS x_p05,
        PERCENTILE(x, 0.10) AS x_p10,
        PERCENTILE(x, 0.90) AS x_p90,
        PERCENTILE(x, 0.95) AS x_p95,
        PERCENTILE(x, 0.98) AS x_p98,
        PERCENTILE(x, 0.99) AS x_p99,
        PERCENTILE(ABS(x), 0.90) AS xa_p90,
        PERCENTILE(ABS(x), 0.92) AS xa_p92,
        PERCENTILE(ABS(x), 0.95) AS xa_p95,
        PERCENTILE(ABS(x), 0.98) AS xa_p98,
        PERCENTILE(ABS(x), 0.99) AS xa_p99
    FROM data 
    GROUP BY seg
    ORDER BY seg
    """ )

    cols = ["x_avg", "x_p01","x_p02","x_p05","x_p10","x_p90","x_p95","x_p98","x_p99",
            "xa_p90","xa_p92","xa_p95","xa_p98","xa_p99"]
    assembler = VectorAssembler(inputCols=cols, outputCol="stat_t1")
    df_stat = assembler.transform(df_stat)
    for col in cols: df_stat = df_stat.drop(col) 

    scaler = StandardScaler(inputCol="stat_t1", outputCol="stat", withStd=True, withMean=False)
    scalerModel = scaler.fit(df_stat)
    df_stat = scalerModel.transform(df_stat).drop("stat_t1")

    df_stat.write.mode("overwrite").parquet(os.path.join("datasets", "train.stat.2.parquet"))

def create_test_stats_feature():
    df_test = spark.read.parquet(os.path.join("datasets", "test.parquet"))
    df_test.createOrReplaceTempView("data")
    df_stat = spark.sql("""
    SELECT seg,
        AVG(x) AS x_avg,
        STD(x)AS x_std, 
        SKEWNESS(x) AS x_skew, 
        KURTOSIS(x) AS x_kurt, 
        MAX(x) AS x_max,
        PERCENTILE(x, 0.05) AS x_p1,
        PERCENTILE(x, 0.20) AS x_p2,
        PERCENTILE(x, 0.50) AS x_p5,
        PERCENTILE(x, 0.80) AS x_p8,
        PERCENTILE(x, 0.95) AS x_p9,
        AVG(ABS(x)) AS xa_avg,
        STD(ABS(x))AS xa_std, 
        SKEWNESS(ABS(x)) AS xa_skew, 
        KURTOSIS(ABS(x)) AS xa_kurt, 
        MAX(ABS(x)) AS xa_max,
        PERCENTILE(ABS(x), 0.05) AS xa_p1,
        PERCENTILE(ABS(x), 0.20) AS xa_p2,
        PERCENTILE(ABS(x), 0.50) AS xa_p5,
        PERCENTILE(ABS(x), 0.80) AS xa_p8,
        PERCENTILE(ABS(x), 0.95) AS xa_p9
    FROM data 
    GROUP BY seg
    ORDER BY seg
    """ )
    cols = ["x_avg", "x_std", "x_skew", "x_kurt", "x_max", "x_p1", "x_p2", "x_p5", "x_p8", "x_p9", 
            "xa_avg", "xa_std", "xa_skew", "xa_kurt", "xa_max", "xa_p1", "xa_p2", "xa_p5", "xa_p8", "xa_p9"]
    assembler = VectorAssembler(inputCols=cols, outputCol="stat_t1")
    df_stat = assembler.transform(df_stat)
    for col in cols: df_stat = df_stat.drop(col)
    scaler = StandardScaler(inputCol="stat_t1", outputCol="stat", withStd=True, withMean=False)
    scalerModel = scaler.fit(df_stat)
    df_stat = scalerModel.transform(df_stat).drop("stat_t1")

    df_stat.write.mode("overwrite").parquet(os.path.join("datasets", "test.stat.parquet"))

def create_label(type_data):
    df_data = spark.read.parquet(os.path.join("datasets", type_data + ".parquet"))
    df_data.createOrReplaceTempView("data")
    df_target = spark.sql("""
    SELECT seg, y FROM data 
    WHERE no = 150000
    ORDER BY seg
    """ )
    df_target.write.mode("overwrite").parquet(os.path.join("datasets", type_data + ".target.parquet"))


df_train_test()
print('Frequency domain transformation finish!!!!')
bin_source_to("train.vector.dct.parquet", "train.vector.fbin.parquet") 
bin_source_to("test.vector.dct.parquet", "test.vector.fbin.parquet")
print('bin_source_to finish!!!!')
slice_source_to("train.vector.fbin.parquet", "train.vector.fbin.2.parquet")
slice_source_to("test.vector.fbin.parquet", "test.vector.fbin.2.parquet")
print('slice vector finish!!!!')
create_train_stats_feature()
print('create train stats feature finish!!!!!!')
create_test_stats_feature()
print('create test stats feature finish!!!!!!')
create_label('train')
print('create train label finish!!!!!!')
create_label('test')
print('create test label finish!!!!!!')







