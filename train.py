import os 
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import numpy as np
import keras, math
from keras.utils.training_utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, CuDNNGRU, RNN, ConvLSTM2D, Conv1D, Reshape, MaxPooling1D, SimpleRNNCell, Flatten
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint   
import matplotlib.pyplot as plt
import config
from sklearn.metrics import mean_absolute_error
import pandas
import csv
from pyspark.ml.feature import VectorSlicer

spark = config.get_config()

df_f = spark.read.parquet(os.path.join("datasets", "train.vector.fbin.2.parquet"))
slicer = VectorSlicer(inputCol='f', outputCol="fsl", indices=[i for i in range(0,26,2)])
df_f = slicer.transform(df_f).drop('f')
df_y = spark.read.parquet(os.path.join("datasets", "train.target.parquet"))

df_f = df_f.selectExpr("*").drop("_c0")
df_y = df_y.selectExpr("seg AS seg2", "y as label").drop("seg")

df_train = df_f
df_train = df_train.join(df_y, df_train.seg.cast(IntegerType()) == df_y.seg2.cast(IntegerType())).drop("seg2")

df_train.printSchema()

n_dim = 13 #26 99 119 99+14 20+26 14+26

# vect_cols = ["f"]
# vectorAssembler = VectorAssembler(inputCols=vect_cols, outputCol="features")
# df_train = vectorAssembler.transform(df_train)


trainingData = df_train.selectExpr("*")

# x_train = trainingData.select("features").orderBy("seg").collect()
x_train = trainingData.select("fsl").orderBy("seg").collect()
y_train = trainingData.select("label").orderBy("seg").collect()


x_train = np.array(x_train).astype('float32').reshape(-1,1,n_dim)
y_train = np.array(y_train).astype('float32')


Y_max = np.max(y_train)
y_train = y_train/Y_max

def mlp_model():
    p_dropout = 0.25
    activation = 'relu'
    model = Sequential()
    model.add(Dense(32, activation=activation, input_shape=(n_dim,)))
    model.add(Dropout(p_dropout))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(p_dropout))
    model.add(Dense(64, activation=activation))
    model.add(Dropout(p_dropout))
    return model

def cnn_rnn_model():
    p_dropout = 0.25
    activation = 'relu'
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu', input_shape=(1,n_dim,)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(LSTM(64, return_sequences=True, activation=activation))
    model.add(Dropout(p_dropout))
    model.add(LSTM(64, activation=activation))
    model.add(Dropout(p_dropout))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(p_dropout))
    return model

def rnn_model():
    model = Sequential()
    model.add(GRU(16, input_shape=(1,n_dim,), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_regularizer=keras.regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    return model

model = cnn_rnn_model()
model.add(Dense(1, activation='relu'))

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='mae', optimizer='adam', metrics=['mse', 'accuracy'])
model.summary()
# checkpointer = ModelCheckpoint(filepath="model/model.2.0.keras.mlp.hdf5", verbose=1, save_best_only=True)
parallel_model.fit(x_train, y_train ,epochs=3000, verbose=1)
model.save('model/model.2.0.keras.cnn_rnn.hdf5')
model.load_weights("model/model.2.0.keras.cnn_rnn.hdf5")

df_f = spark.read.parquet(os.path.join("datasets", "test.vector.fbin.2.parquet"))
slicer = VectorSlicer(inputCol='f', outputCol="fsl", indices=[i for i in range(0,26,2)])
df_f = slicer.transform(df_f).drop('f')
df_y = spark.read.parquet(os.path.join("datasets", "test.target.parquet"))
df_f = df_f.selectExpr("*").drop("_c0")
df_y = df_y.selectExpr("seg AS seg2", "y AS label")
df_test = df_f
df_test = df_test.join(df_y, df_test.seg.cast(IntegerType()) == df_y.seg2.cast(IntegerType())).drop("seg2")

# vect_cols = ["f"]
# vectorAssembler = VectorAssembler(inputCols=vect_cols, outputCol="features")
# df_test = vectorAssembler.transform(df_test)
testData = df_test.selectExpr("*")
# x_test = testData.select("features").orderBy("seg").collect()
x_test = testData.select("fsl").orderBy("seg").collect()
x_test = np.array(x_test).astype('float32').reshape(-1,1,n_dim)

y_hat = model.predict(x_train)

y_hat = y_hat * Y_max
y_train = y_train * Y_max
print("mae : ", "{0:.20f}".format(mean_absolute_error(y_train, y_hat)))

fig, ax = plt.subplots(figsize=(20,8))
plt.plot(y_hat, alpha=0.8, label="predictions")
plt.plot(y_train, label="labels")
title = ax.set_title("Prediction vs. ground truth", loc="left")
legend = plt.legend(loc="upper left")
plt.savefig("result/result.png")

y_hat = model.predict(x_test)

y_hat = y_hat * Y_max

h = [['seg_id', 'time_to_failure']]
f = open('result/result.csv','w')
writer = csv.writer(f)
writer.writerows(h)
df = pandas.read_csv('datasets/sample_submission.csv')
for index, x in enumerate(y_hat):

    d = [[df['seg_id'][index], "{0:.20f}".format(y_hat[index][0])]]
    writer.writerows(d)

