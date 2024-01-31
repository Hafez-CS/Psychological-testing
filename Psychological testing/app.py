import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import math

df = pd.read_csv('ph.csv')

plt.subplots(figsize=(9, 9))
sns.heatmap(df.corr(), annot=True)


x = df.drop("dep", axis=1)
y = df.dep

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


model = Sequential()
model.add(Dense(64, input_dim=11, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='sigmoid')) 
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=200, batch_size=10)


scores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

# model.save()


##############################################################


man = np.array([[1,3,3,33,0,0,2,1,2,2,2]])

out=model.predict(man)
h=np.array(out[0])
bars = ('low', 'medium', 'high') 
  
plt.bar(bars, h)
 
plt.show()