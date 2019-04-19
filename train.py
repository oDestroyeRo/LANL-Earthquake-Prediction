import os 
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import keras, math
from keras.utils.training_utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint   
import matplotlib.pyplot as plt
import config
from sklearn.metrics import mean_absolute_error
import pandas
import csv

spark = config.get_config()

df_f = spark.read.parquet(os.path.join("datasets", "train.vector.fbin.2.parquet"))
df_y = spark.read.parquet(os.path.join("datasets", "train.target.parquet"))

df_f = df_f.selectExpr("*").drop("_c0")
df_y = df_y.selectExpr("seg AS seg2", "y AS label")

df_train = df_f
df_train = df_train.join(df_y, df_train.seg.cast(IntegerType()) == df_y.seg2.cast(IntegerType())).drop("seg2")

df_train.printSchema()

n_dim = 26 #26 99 119 99+14 20+26 14+26

vect_cols = ["f"]
vectorAssembler = VectorAssembler(inputCols=vect_cols, outputCol="features")

df_train = vectorAssembler.transform(df_train)


trainingData = df_train.selectExpr("*").where("seg < 3145")
testData = df_train.selectExpr("*").where("seg >= 3145")

x_train = trainingData.select("features").orderBy("seg").collect()
x_test = testData.select("features").orderBy("seg").collect()
y_train = trainingData.select("label").orderBy("seg").collect()
y_test = testData.select("label").orderBy("seg").collect()


x_train = np.array(x_train).astype('float32').reshape(-1,n_dim)
x_test = np.array(x_test).astype('float32').reshape(-1,n_dim)
y_train = np.array(y_train).astype('float32').reshape(-1,1)
y_test = np.array(y_test).astype('float32').reshape(-1,1)

p_dropout = 0.25
activation = 'relu'
model =  Sequential()
model.add(Dense(32, activation=activation, input_shape=(n_dim,)))
model.add(Dropout(p_dropout))
model.add(Dense(32, activation=activation))
model.add(Dropout(p_dropout))
model.add(Dense(64, activation=activation))
model.add(Dropout(p_dropout))
model.add(Dense(1))

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='mae', optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True))
model.summary()
# checkpointer = ModelCheckpoint(filepath="model/model.2.0.keras.mlp.hdf5", verbose=1, save_best_only=True)
parallel_model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), verbose=1)
model.save('model/model.2.0.keras.mlp.hdf5')
model.load_weights("model/model.2.0.keras.mlp.hdf5")



df_f = spark.read.parquet(os.path.join("datasets", "test.vector.fbin.2.parquet"))
df_y = spark.read.parquet(os.path.join("datasets", "test.target.parquet"))

df_f = df_f.selectExpr("*").drop("_c0")
df_y = df_y.selectExpr("seg AS seg2", "y AS label")

df_train = df_f
df_train = df_train.join(df_y, df_train.seg.cast(IntegerType()) == df_y.seg2.cast(IntegerType())).drop("seg2")

df_train.printSchema()

n_dim = 26 #26 99 119 99+14 20+26 14+26

vect_cols = ["f"]
vectorAssembler = VectorAssembler(inputCols=vect_cols, outputCol="features")

df_train = vectorAssembler.transform(df_train)


trainingData = df_train.selectExpr("*")
testData = df_train.selectExpr("*")

x_train = trainingData.select("features").orderBy("seg").collect()
x_test = testData.select("features").orderBy("seg").collect()
y_train = trainingData.select("label").orderBy("seg").collect()
y_test = testData.select("label").orderBy("seg").collect()


x_train = np.array(x_train).astype('float32').reshape(-1,n_dim)
x_test = np.array(x_test).astype('float32').reshape(-1,n_dim)
y_train = np.array(y_train).astype('float32').reshape(-1,1)
y_test = np.array(y_test).astype('float32').reshape(-1,1)



y_hat = model.predict(x_test)

# print(x_test)
# print(testData.collect())
print(y_hat)
# print(mean_absolute_error(y_test, y_hat))


h = [['seg_id', 'time_to_failure']]

f = open('result.csv','w')
writer = csv.writer(f)
writer.writerows(h)




df = pandas.read_csv('datasets/sample_submission.csv')

for index, x in enumerate(y_hat):
    # d=[[df['seg_id'][x],random.uniform(1.469100, 9.759796)]]
    d = [[df['seg_id'][index], y_hat[index][0]]]
    writer.writerows(d)




