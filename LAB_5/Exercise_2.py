import pandas as pd
import numpy as np
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


path_to_dataset = 'voting_complete.csv' # change the PATH
pd_dataset = pd.read_csv(path_to_dataset)

# define a function for train and test split

def train_test_split(pd_data: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    pd_dataset = pd_data.copy()
    pd_dataset = pd_dataset[pd_dataset.columns[1:]]
    index = np.arange(len(pd_dataset))
    index = np.random.permutation(index)
    train_ammount = int(len(index)*test_ratio)
    train_ids = index[train_ammount:]
    test_ids = index[:train_ammount]

    train_dataset = pd_dataset[pd_dataset.index.isin(train_ids)].reset_index()
    test_dataset = pd_dataset[pd_dataset.index.isin(test_ids)].reset_index()

    train_dataset = train_dataset[train_dataset.columns[1:]]
    test_dataset = test_dataset[test_dataset.columns[1:]]

    return train_dataset[train_dataset.columns[1:]], train_dataset[train_dataset.columns[0]], test_dataset[test_dataset.columns[1:]], test_dataset[test_dataset.columns[0]]

x_train, y_train, x_test, y_test = train_test_split(pd_dataset)

# Fill NaN values with the mode of each column (the most frequent value)
# Replace '?' with NaN
x_train.replace('?', np.nan, inplace=True)
x_train.fillna(x_train.mode().iloc[0], inplace=True)
print(x_train)
# Perform one-hot encoding
x=pd.get_dummies(x_train)
print(x)
y = y_train.replace({'republican':1, 'democrat':0})
print(y)
model = Sequential()
model.add(Dense(2, input_shape=(x.shape[1],), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

optimizer = rmsprop_v2.RMSprop(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Split the dataset into training and validation datasets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

hystory = model.fit(X_train, y_train, epochs=40, batch_size=4, verbose=1, validation_data=(X_val, y_val))

x_test.replace('?', np.nan, inplace=True)
x_test.fillna(x_test.mode().iloc[0], inplace=True)
print(x_test)
x_te=pd.get_dummies(x_test)
#x_te
y_te = y_test.replace({'republican':1, 'democrat':0})
print(y_te)

loss, accuracy = model.evaluate(x_te, y_te, verbose=0)
print('Accuracy: {:.2f}'.format(accuracy*100))
print('Loss: {:.2f}'.format(loss*100))

#Plot the loss and accuracy history
fig, ax = plt.subplots()
ax.plot(hystory.history['loss'], label='Training Loss')
ax.plot(hystory.history['val_loss'], label='Validation_loss')
ax.set_ylabel('Value')
ax.set_xlabel('Epoch')
ax.set_title('Model Loss over Epochs')
ax.legend()
plt.show()
fig, ay = plt.subplots()
ay.plot(hystory.history['accuracy'], label='Training Accuracy')
ay.plot(hystory.history['val_accuracy'], label='Validation_Accuracy')
ay.set_ylabel('Value')
ay.set_xlabel('Epoch')
ay.set_title('Model Accuracy over Epochs')
ay.legend()
plt.show()

