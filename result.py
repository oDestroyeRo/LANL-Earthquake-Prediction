import numpy as np
import keras, math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import os 
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import mean_absolute_error
import config

spark = config.get_config()

df_y = spark.read.parquet(os.path.join("datasets", "train.target.parquet"))
df_y = df_y.selectExpr("seg AS seg2", "y AS label")
testData = df_train.selectExpr("*").where("seg >= 3145")
y_test = testData.select("label").orderBy("seg").collect()
y_test = np.array(y_test).astype('float32').reshape(-1,1)
model.load_weights("model.2.0.keras.mlp.hdf5")
y_hat = model.predict(x_test)
fig, ax = plt.subplots(figsize=(20,8))
plt.plot(y_hat, alpha=0.8, label="predictions")
plt.plot(y_test, label="labels")
title = ax.set_title("Prediction vs. ground truth", loc="left")
legend = plt.legend(loc="upper left")
mean_absolute_error(y_test, y_hat)