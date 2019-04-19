from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import DCT
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.feature import StandardScaler
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
import os 
import config

spark = config.get_config()

def vec_dct_to_win():
    df_train = spark.read.parquet(os.path.join("datasets", "train.vector.parquet"))
    df_test = spark.read.parquet(os.path.join("datasets", "test.vector.parquet"))
    for j in range(8):
    cols = ["vx"+str(j+i) for i in range(3)]
    assembler = VectorAssembler(inputCols=cols, outputCol="vx_w"+str(j))
    dct = DCT(inverse=False, inputCol="vx_w"+str(j), outputCol="fr_w"+str(j))
    slicer = VectorSlicer(inputCol="fr_w"+str(j), outputCol="fs_w"+str(j), indices=[i for i in range(12000)])
    scaler = StandardScaler(inputCol="fs_w"+str(j), outputCol="fn_w"+str(j), withStd=True, withMean=False)

    pipeline = Pipeline(stages=[assembler, dct, slicer, scaler])
    pw_model = pipeline.fit(df_train)    

    df_train = pw_model.transform(df_train).drop("vx"+str(j)).drop("vx_w"+str(j)).drop("fr_w"+str(j)).drop("fs_w"+str(j)) 
    df_test = pw_model.transform(df_test).drop("vx"+str(j)).drop("vx_w"+str(j)).drop("fr_w"+str(j)).drop("fs_w"+str(j))  

    df_train.write.mode("overwrite").parquet(os.path.join("datasets", "train.win.vector.dct.parquet"))
    df_test.write.mode("overwrite").parquet(os.path.join("datasets", "test.win.vector.dct.parquet"))

    df_train.printSchema()
    df_test.printSchema()

def bin_win_source_to(source, destination):
    df_w = spark.read.parquet(os.path.join("datasets", source))
    for j in range(8):
        rdd_w = df_w.rdd.map(lambda row, j=j:(row.seg, row["fn_w"+str(j)], []))
        for i in range(0,99):
            rdd_w = rdd_w.map(lambda row, i=i:(row[0], row[1], row[2] + [float(sum(abs(row[1].toArray()[i*120:i*120+240]))/120)]))
        rdd_w = rdd_w.map(lambda row: Row(seg=row[0],f=Vectors.dense(row[2])))
        df_tmp = rdd_w.toDF()
        df_tmp = df_tmp.selectExpr("seg AS seg2", "f AS f"+str(j)).drop("seg").drop("_c0")
        df_w = df_w.join(df_tmp, df_w.seg.cast(IntegerType()) == df_tmp.seg2.cast(IntegerType())).drop("seg2").drop("fn_w"+str(j))
    df_w = df_w.drop("vx8").drop("vx9")
    df_w.write.mode("overwrite").parquet(os.path.join("datasets", destination))

def slice_win_source_to(source, destination):
    df_w = spark.read.parquet(os.path.join("datasets", source))
    for j in range(8):
        slicer = VectorSlicer(inputCol="f"+str(j), outputCol="f_sl"+str(j), indices=[i for i in range(50,76)])
        df_w = slicer.transform(df_w).drop("f"+str(j))
    cols = ["f_sl"+str(i) for i in range(8)]
    assembler = VectorAssembler(inputCols=cols, outputCol="f")
    df_w = assembler.transform(df_w) 
    df_w.write.mode("overwrite").parquet(os.path.join("datasets", destination))
    df_w.printSchema()

def label_win_to():
    df_train = spark.read.parquet(os.path.join("datasets", "train.parquet"))
    df_train.createOrReplaceTempView("data")
    df_target = spark.sql("""
    SELECT d0.seg, d0.y AS y0, d1.y AS y1, d2.y AS y2, d3.y AS y3, 
                d4.y AS y4, d5.y AS y5, d6.y AS y6, d7.y AS y7
    FROM        data AS d0 
    INNER JOIN  data AS d1 ON d1.no =  60000 AND d1.seg = d0.seg
    INNER JOIN  data AS d2 ON d2.no =  75000 AND d2.seg = d0.seg
    INNER JOIN  data AS d3 ON d3.no =  90000 AND d3.seg = d0.seg
    INNER JOIN  data AS d4 ON d4.no = 105000 AND d4.seg = d0.seg
    INNER JOIN  data AS d5 ON d5.no = 120000 AND d5.seg = d0.seg
    INNER JOIN  data AS d6 ON d6.no = 135000 AND d6.seg = d0.seg
    INNER JOIN  data AS d7 ON d7.no = 150000 AND d7.seg = d0.seg
    WHERE d0.no = 45000
    ORDER BY d0.seg
    """ )
    df_target.write.mode("overwrite").parquet(os.path.join("datasets", "train.win.target.parquet"))
    df_target.show()

def stats_feature_win_to():
    df_train = spark.read.parquet(os.path.join("datasets", "train.parquet"))
    df_train.createOrReplaceTempView("data")
    df_stat = spark.sql("""
    SELECT seg,
        INT(no/1000) AS seq,
        AVG(x) AS x_avg,
        PERCENTILE(x, 0.02) AS x_p02,
        PERCENTILE(x, 0.98) AS x_p98,
        PERCENTILE(ABS(x), 0.95) AS xa_p95
    FROM data
    GROUP BY seg, INT(no/1000)
    ORDER BY seg, INT(no/1000)
    """ )
    df_agg = sc.parallelize([])
    rdd_temp = df_stat.rdd.map(lambda row:(row.seg, row.x_avg, row.x_p02, row.x_p98, row.xa_p95))      \
        .map(lambda data: (data[0], ([ data[1] ], [ data[2] ] , [ data[3] ] , [ data[4] ] )))          \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])  )              \
        .map(lambda row: Row(seg=row[0],
                            stx1=Vectors.dense(row[1][0]),
                            stx2=Vectors.dense(row[1][1]),
                            stx3=Vectors.dense(row[1][2]),
                            stx4=Vectors.dense(row[1][3])))      
    # if df_agg.count() == 0: 
    df_agg = rdd_temp.toDF(["seg","stx1","stx2","stx3","stx4"])

    # df_agg.show()
    df_agg = df_agg.select("*").where("seg != 4194")
    scaler = StandardScaler(inputCol="stx1", outputCol="stxn1", withStd=True, withMean=False)
    scalerModel = scaler.fit(df_agg)
    df_agg = scalerModel.transform(df_agg).drop("stx1")

    scaler = StandardScaler(inputCol="stx2", outputCol="stxn2", withStd=True, withMean=False)
    scalerModel = scaler.fit(df_agg)
    df_agg = scalerModel.transform(df_agg).drop("stx2")

    scaler = StandardScaler(inputCol="stx3", outputCol="stxn3", withStd=True, withMean=False)
    scalerModel = scaler.fit(df_agg)
    df_agg = scalerModel.transform(df_agg).drop("stx3")

    scaler = StandardScaler(inputCol="stx4", outputCol="stxn4", withStd=True, withMean=False)
    scalerModel = scaler.fit(df_agg)
    df_agg = scalerModel.transform(df_agg).drop("stx4")

    df_agg.write.mode("overwrite").parquet(os.path.join("datasets", "train.win.stat.parquet"))


vec_dct_to_win()
print("vec_dct_to_win finish!!!!!!!!!!")
bin_win_source_to("train.win.vector.dct.parquet", "train.win.vector.fbin.parquet")
print("bin_win_source_to train finish!!!!")
bin_win_source_to("test.win.vector.dct.parquet", "test.win.vector.fbin.parquet")
print("bin_win_source_to test finish!!!!")
slice_win_source_to("train.win.vector.fbin.parquet", "train.win.vector.fbin.2.parquet")
print("slice_win_source_to train finish!!!!!!!")
slice_win_source_to("test.win.vector.fbin.parquet", "test.win.vector.fbin.2.parquet")
print("slice_win_source_to test finish!!!!!!!")
label_win_to()
print("label_win_to finish!!!!!!!")
stats_feature_win_to()
print("stats_feature_win_to finish!!!!!!!!!!!")