001 Understanding the Perceptron

\# Import the libraries and dataset

import numpy as np 

from sklearn.model\_selection import train\_test\_split

import matplotlib.pyplot as plt

\# We will be using the Iris Plants Database

from sklearn.datasets import load\_iris

SEED = 2017

\# The first two classes (Iris-Setosa and Iris-Versicolour) are linear separable

iris = load\_iris()

idxs = np.where(iris.target<2)

X = iris.data[idxs]

y = iris.target[idxs]

\# Let's plot the data for two of the four variables 

plt.scatter(X[y==0][:,0],X[y==0][:,2], color='green', label='Iris-Setosa')

plt.scatter(X[y==1][:,0],X[y==1][:,2], color='red', label='Iris-Versicolour')

plt.title('Iris Plants Database')

plt.xlabel('sepal length in cm')

plt.ylabel('sepal width in cm')

plt.legend()

plt.show()

\# In the following graph, we've plotted the distribution of the two classes4

\# To validate our results, we split the data into training and validation sets

X\_train, X\_val, y\_train, y\_val = train\_test\_split(X, y, test\_size=0.2, random\_state=SEED)

print(X\_train)

print(X\_val)

print(y\_train)

print(y\_val)

\# Next, we initialize the weights and the bias for the perceptron

weights = np.random.normal(size=X\_train.shape[1])

bias = 1

\# Before training, we need to define the hyperparameters

learning\_rate = 0.1

n\_epochs = 15

np.zeros(weights.shape)

\# Now, we can start training our perceptron with a for loop

del\_w = np.zeros(weights.shape)

hist\_loss = []

hist\_accuracy = []

for i in range(n\_epochs):

`    `# We apply a simple step function, if the output is > 0.5 we predict 1, else 0

`    `output = np.where((X\_train.dot(weights)+bias)>0.5, 1, 0)

`    `print(output)



`    `# Compute MSE

`    `error = np.mean((y\_train-output)\*\*2)

`    `print("Error: ", error)

`    `# Update weights and bias

`    `weights-= learning\_rate \* np.dot((output-y\_train), X\_train)

`    `bias += learning\_rate \* np.sum(np.dot((output-y\_train), X\_train))

`    `print("Weights:", weights)

`    `print("bias:", bias)

`    `# Calculate MSE

`    `loss = np.mean((output - y\_train) \*\* 2)

`    `hist\_loss.append(loss)

`    `# Determine validation accuracy

`    `output\_val = np.where(X\_val.dot(weights)>0.5, 1, 0)

`    `accuracy = np.mean(np.where(y\_val==output\_val, 1, 0))

`    `hist\_accuracy.append(accuracy)

002 Implementing a single-layer neural network

\# Import libraries and dataset

import numpy as np 

from sklearn.model\_selection import train\_test\_split

import matplotlib.pyplot as plt

\# We will be using make\_circles from scikit-learn

from sklearn.datasets import make\_circles

SEED = 2017

\# First, we need to create the training data

\# We create an inner and outer circle

X, y = make\_circles(n\_samples=400, factor=.3, noise=.05, random\_state=2017)

outer = y == 0

inner = y == 1

\# Let's plot the data to show the two classes

plt.title("Two Circles")

plt.plot(X[outer, 0], X[outer, 1], "ro")

plt.plot(X[inner, 0], X[inner, 1], "bo")

plt.show()

\# Example of non-linearly separable data

\# We normalize the data to make sure the center of both circles is (1,1)

X = X+1

\# To determine the performance of our algorithm we split our data

X\_train, X\_val, y\_train, y\_val = train\_test\_split(X, y, test\_size=0.2, random\_state=SEED)

\# A linear activation function won't work in this case, so we'll be using a sigmoid function

def sigmoid(x):

`    `return 1 / (1 + np.exp(-x))

\# Next, we define the hyperparameters

n\_hidden = 50 # number of hidden units

n\_epochs = 1000

learning\_rate = 1

\# Initialize the weights and other variables

\# Initialise weights

weights\_hidden = np.random.normal(0.0, size=(X\_train.shape[1], n\_hidden))

weights\_output = np.random.normal(0.0, size=(n\_hidden))

hist\_loss = []

hist\_accuracy = []

print(weights\_hidden)

print(weights\_output)

\# Run the single-layer neural network and output the statistics

for e in range(n\_epochs):

`    `del\_w\_hidden = np.zeros(weights\_hidden.shape)

`    `del\_w\_output = np.zeros(weights\_output.shape)

`    `# Loop through training data in batches of 1

`    `for x\_, y\_ in zip(X\_train, y\_train):

`        `# Forward computations

`        `hidden\_input = np.dot(x\_, weights\_hidden)

`        `hidden\_output = sigmoid(hidden\_input)

`        `output = sigmoid(np.dot(hidden\_output, weights\_output))

`        `# Backward computations

`        `error = y\_ - output

`        `output\_error = error \* output \* (1 - output)

`        `hidden\_error = np.dot(output\_error, weights\_output) \* hidden\_output \* (1 - hidden\_output)

`        `del\_w\_output += output\_error \* hidden\_output

`        `del\_w\_hidden += hidden\_error \* x\_[:, None]

`    `# Update weights

`    `weights\_hidden += learning\_rate \* del\_w\_hidden / X\_train.shape[0]

`    `weights\_output += learning\_rate \* del\_w\_output / X\_train.shape[0]

`    `# Print stats (validation loss and accuracy)

`    `if e % 100 == 0:

`        `hidden\_output = sigmoid(np.dot(X\_val, weights\_hidden))

`        `out = sigmoid(np.dot(hidden\_output, weights\_output))

`        `loss = np.mean((out - y\_val) \*\* 2)

`        `# Final prediction is based on a threshold of 0.5

`        `predictions = out > 0.5

`        `accuracy = np.mean(predictions == y\_val)

`        `print("Epoch: ", '{:>4}'.format(e), 

`            `"; Validation loss: ", '{:>6}'.format(loss.round(4)), 

`            `"; Validation accuracy: ", '{:>6}'.format(accuracy.round(4)))

003 Building a multi-layer neural network

\# We start by import the libraries

import numpy as np 

import pandas as pd

from sklearn.model\_selection import train\_test\_split

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

SEED = 2017

\# Data can be downloaded at <https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv>

\# Load the wine data set

data = pd.read\_csv('C:\\Users\\ifsrk\\Documents\\01 Deep Learning\\winequality-red.csv', sep=';')

y = data['quality']

X = data.drop(['quality'], axis=1)

\# Split data for training and testing

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=SEED)

\# Print average quality and first rows of training set

print('Average quality training set: {:.4f}'.format(y\_train.mean()))

X\_train.head()

\# An important next step is to normalize the input data

scaler = StandardScaler().fit(X\_train)

X\_train = pd.DataFrame(scaler.transform(X\_train))

X\_test = pd.DataFrame(scaler.transform(X\_test))

\# Determine baseline predictions

\# Predict the mean quality of the training data for each validation input

print('MSE:', np.mean((y\_test - ([y\_train.mean()] \* y\_test.shape[0])) \*\* 2))

print('MSE:', np.mean((y\_test - ([y\_train.mean()] \* y\_test.shape[0])) \*\* 2))

\# Now, let's build our neural network by defining the network architecture

model = Sequential()

\# First hidden layer with 100 hidden units

model.add(Dense(200, input\_dim=X\_train.shape[1], activation='relu')) 

\# Second hidden layer with 50 hidden units

model.add(Dense(25, activation='relu'))

\# Output layer

model.add(Dense(1, activation='linear'))

\# Set optimizer

opt = Adam()

\# Compile model

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

\# Define the callback for early stopping and saving the best model

callbacks = [

`             `EarlyStopping(monitor='val\_accuracy', patience=30, verbose=2),

`             `ModelCheckpoint('checkpoints/multi\_layer\_best\_model.h5', monitor='val\_accuracy', save\_best\_only=True, verbose=0)

`            `]

\# Run the model with a batch size of 64, 5,000 epochs, and a validation split of 20%

batch\_size = 64

n\_epochs = 5000

model.fit(X\_train.values, y\_train, batch\_size=64, epochs=n\_epochs, validation\_split=0.2,     

`             `verbose=2,

`              `validation\_data=(X\_test.values, y\_test),

`             `callbacks=callbacks)

model.summary()

\# We can now print the performance on the test set after loading the optimal weights:

best\_model = model

best\_model.load\_weights('checkpoints/multi\_layer\_best\_model.h5')

best\_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

\# Evaluate on test set

score = best\_model.evaluate(X\_test.values, y\_test, verbose=0)

print('Test accuracy: %.2f%%' % (score[1]\*100))

\# Test accuracy: 65.62% 

\# Benchmark accuracy on dataset 62.4%

***Getting started with activation function***

\# Import the libraries as follows

import numpy as np 

import pandas as pd

from sklearn.model\_selection import train\_test\_split

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.utils import to\_categorical

from keras.callbacks import Callback

from keras.datasets import mnist

SEED = 2022

\# Load the MNIST dataset

\# Need Internet Connection to download dataset

(X\_train, y\_train), (X\_val, y\_val) = mnist.load\_data()

\# Show an example of each label and print the count per label

\# Plot first image of each label

unique\_labels = set(y\_train)

plt.figure(figsize=(12, 12))

i = 1

for label in unique\_labels:

`    `image = X\_train[y\_train.tolist().index(label)]

`    `plt.subplot(10, 10, i)

`    `plt.axis('off')

`    `plt.title("{0}: ({1})".format(label, y\_train.tolist().count(label)))

`    `i += 1

`    `\_ = plt.imshow(image, cmap='gray')

plt.show()

print(X\_val)

print(y\_val)

\# Preprocess the data

\# Normalize data

X\_train = X\_train.astype('float32')/255.

X\_val = X\_val.astype('float32')/255.

X\_val

\# One-Hot-Encode labels

n\_classes = 10

y\_train = to\_categorical(y\_train, n\_classes)

y\_val = to\_categorical(y\_val, n\_classes)

print(y\_train)

\# Flatten data - we treat the image as a sequential array of values

X\_train = np.reshape(X\_train, (60000, 784))

X\_val = np.reshape(X\_val, (10000, 784))

X\_train

\# Define the model with the sigmoid activation function

model\_sigmoid = Sequential()

model\_sigmoid.add(Dense(700, input\_dim=784, activation='sigmoid'))

model\_sigmoid.add(Dense(700, activation='sigmoid'))

model\_sigmoid.add(Dense(700, activation='sigmoid'))

model\_sigmoid.add(Dense(700, activation='sigmoid'))

model\_sigmoid.add(Dense(700, activation='sigmoid')) 

model\_sigmoid.add(Dense(350, activation='sigmoid')) 

model\_sigmoid.add(Dense(100, activation='sigmoid')) 

model\_sigmoid.add(Dense(10, activation='softmax'))

\# Compile model with SGD

model\_sigmoid.compile(loss='categorical\_crossentropy',

optimizer='sgd', metrics=['accuracy'])

\# Define the model with the ReLU activation function

model\_relu = Sequential()

model\_relu.add(Dense(700, input\_dim=784, activation='relu'))

model\_relu.add(Dense(700, activation='relu'))

model\_relu.add(Dense(700, activation='relu'))

model\_relu.add(Dense(700, activation='relu'))

model\_relu.add(Dense(700, activation='relu')) 

model\_relu.add(Dense(350, activation='relu')) 

model\_relu.add(Dense(100, activation='relu')) 

model\_relu.add(Dense(10, activation='softmax'))

\# Compile model with SGD

model\_relu.compile(loss='categorical\_crossentropy', 

optimizer='sgd', metrics=['accuracy'])

\# Create a callback function to store the loss values per batch

class history\_loss(Callback):

`    `def on\_train\_begin(self, logs={}):

`        `self.losses = []

`    `def on\_batch\_end(self, batch, logs={}):

`        `batch\_loss = logs.get('loss')

`        `self.losses.append(batch\_loss)

n\_epochs = 10

batch\_size = 256

validation\_split = 0.2

history\_sigmoid = history\_loss()

model\_sigmoid.fit(X\_train, y\_train, epochs=n\_epochs,

batch\_size=batch\_size,

callbacks=[history\_sigmoid],

validation\_split=validation\_split, verbose=2)

history\_relu = history\_loss()

model\_relu.fit(X\_train, y\_train, epochs=n\_epochs,

batch\_size=batch\_size,

callbacks=[history\_relu],

validation\_split=validation\_split, verbose=2)

plt.plot(np.arange(len(history\_sigmoid.losses)), 

`         `history\_sigmoid.losses, label='sigmoid')

plt.plot(np.arange(len(history\_relu.losses)), 

`         `history\_relu.losses, label='relu')

plt.title('Losses for sigmoid and ReLU model')

plt.xlabel('number of batches')

plt.ylabel('loss')

plt.legend(loc=1)

plt.show()

\# Losses for sigmoid and ReLU model

##### **005*Experiment with hidden layers and hidden units***
#####
\# Import libraries as follows

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model\_selection import train\_test\_split

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.optimizers import SGD

SEED = 2017

\# Data can be downloaded at <https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv>

\# Load the dataset

data = pd.read\_csv('C:\\Users\\ifsrk\\Documents\\01 Deep Learning\\winequality-red.csv', sep=';')

y = data['quality']

X = data.drop(['quality'], axis=1)

\# Split the dataset into training and testing

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=SEED)

\# Normalize the input data

scaler = StandardScaler().fit(X\_train)

X\_train = pd.DataFrame(scaler.transform(X\_train))

X\_test = pd.DataFrame(scaler.transform(X\_test))

print("Input Layers :", X\_train.shape[1])

\# Define the model and optimizer and compile

model = Sequential()

model.add(Dense(1024, input\_dim=X\_train.shape[1], activation='relu')) 

model.add(Dense(1024, activation='relu')) 

model.add(Dense(1024, activation='relu')) 

model.add(Dense(1024, activation='relu')) 

\# Output layer

model.add(Dense(1, activation='linear'))

\# Set optimizer

opt = SGD(learning\_rate=0.01)

\# Compile model

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

! pip install pydot

! pip install pydotplus

! pip install graphviz

\# Visualize network architecture

from IPython.display import SVG

from keras.utils.vis\_utils import model\_to\_dot

from keras.utils.vis\_utils import plot\_model

SVG(model\_to\_dot(model, show\_shapes=True).create(prog="dot", format="svg"))

\# Save the visualization as a file

plot\_model(model, show\_shapes=True, to\_file="network\_model.png")

\# Set the hyperparameters and train the model

n\_epochs = 500

batch\_size = 256

history = model.fit(X\_train.values, y\_train, batch\_size=batch\_size, epochs=n\_epochs, validation\_split=0.2, verbose=1)

\# Predict on the test set

predictions = model.predict(X\_test.values)

print('Test accuracy: {:f>2}%'.format(np.round(np.sum([y\_test==predictions.flatten().round()])/y\_test.shape[0]\*100, 2)))

\# list all data in history

print(history.history.keys())

np.arange(len(history.history['val\_accuracy'])), history.history['val\_accuracy']

\# Plot the training and validation accuracy

plt.plot(np.arange(len(history.history['accuracy'])), history.history['accuracy'], label='training')

plt.plot(np.arange(len(history.history['val\_accuracy'])), history.history['val\_accuracy'], label='validation')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy ')

plt.legend(loc=0)

plt.show()

##### ***006 Experimenting with different optimizers***
#####
##### ***import numpy as np***
##### ***import pandas as pd***
#####
##### ***from sklearn.model\_selection import train\_test\_split***
##### ***from keras.models import Sequential***
##### ***from keras.layers import Dense, Dropout***
##### ***from keras.callbacks import EarlyStopping, ModelCheckpoint***
##### ***from tensorflow.keras.optimizers import SGD, Adadelta, Adam, RMSprop, Adagrad, Nadam, Adamax***
#####
##### ***SEED = 2022***
##### ***# Data can be downloaded at [https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv***](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)***
##### ***data = pd.read\_csv('C:\\Users\\ifsrk\\Documents\\01 Deep Learning\\winequality-red.csv', sep=';')***
##### ***y = data['quality']***
##### ***X = data.drop(['quality'], axis=1)***
##### ***X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=SEED)***
##### ***X\_train, X\_val, y\_train, y\_val = train\_test\_split(X\_train, y\_train, test\_size=0.2, random\_state=SEED)***
##### ***SEED***
##### ***print(np.any(np.isnan(X\_test)))***
##### ***print(np.any(np.isinf(X\_test)))***
##### ***print(np.any(np.isnan(X\_train)))***
##### ***print(np.any(np.isinf(X\_train)))***
##### ***print(np.any(np.isnan(y\_test)))***
##### ***print(np.any(np.isinf(y\_test)))***
##### ***print(np.any(np.isnan(y\_train)))***
##### ***print(np.any(np.isinf(y\_train)))***
##### ***def create\_model(opt):*** 
##### `    `***model = Sequential()***
##### `    `***model.add(Dense(100, input\_dim=X\_train.shape[1],***
##### `    `***activation='relu'))***
##### `    `***model.add(Dense(50, activation='relu'))***
##### `    `***model.add(Dense(25, activation='relu'))***
##### `    `***model.add(Dense(10, activation='relu'))***
##### `    `***model.add(Dense(1, activation='linear'))***
##### `    `***return model***
##### ***def create\_callbacks(opt):***
##### `    `***callbacks = [***
##### `    `***EarlyStopping(monitor='accuracy', patience=50, verbose=2),***
##### `    `***ModelCheckpoint('checkpoints/optimizers\_best\_' + opt + '.h5', monitor='accuracy', save\_best\_only=True, verbose=1)***
##### `    `***]***
##### `    `***return callbacks***
##### ***opts = dict({***
##### `    `***'sgd': SGD(),***
##### `     `***'sgd-0001': SGD(learning\_rate=0.0001, decay=0.00001),***
##### `     `***'adam': Adam(),***
##### `     `***'adadelta': Adadelta(),***
##### `     `***'rmsprop': RMSprop(),***
##### `     `***'rmsprop-0001': RMSprop(learning\_rate=0.0001),***
##### `     `***'nadam': Nadam(),***
##### `     `***'adamax': Adamax()***
##### `    `***})***
##### ***X\_train.values***
##### ***batch\_size = 128***
##### ***n\_epochs = 1000***
#####
##### ***results = []***
##### ***# Loop through the optimizers***
##### ***for opt in opts:***
##### `    `***model = create\_model(opt)***
##### `    `***callbacks = create\_callbacks(opt)***
##### `    `***model.compile(loss='mse', optimizer=opts[opt], metrics=['accuracy'])***
##### ***#   model.compile(loss='mse', optimizer=opts[opt], metrics=['mean\_squared\_error'])***
##### `    `***hist = model.fit(X\_train.values, y\_train, batch\_size=batch\_size, epochs=n\_epochs, validation\_data=(X\_val.values, y\_val), verbose=1,***
##### `    `***callbacks=callbacks)***
##### `    `***print(hist.history)***
##### `    `***best\_epoch = np.argmax(hist.history['accuracy'])***
##### `    `***print(best\_epoch)***
##### `    `***best\_acc = hist.history['accuracy'][best\_epoch]*** 
##### `    `***print(best\_acc)***
##### `    `***best\_model = create\_model(opt)***
##### `    `***best\_model.summary()***
##### `    `***# Load the model weights with the highest validation accuracy*** 
##### `    `***best\_model.load\_weights('checkpoints/optimizers\_best\_' + opt + '.h5')***
##### `    `***best\_model.compile(loss='mse', optimizer=opts[opt], metrics=['accuracy'])***
##### `    `***score = best\_model.evaluate(X\_test.values, y\_test, verbose=0)***
##### `    `***results.append([opt, best\_epoch, best\_acc, score[1]])***
##### ***res = pd.DataFrame(results)***
##### ***res***
#####
##### ***res.columns = ['optimizer', 'epochs', 'val\_accuracy', 'test\_accuracy']***
##### ***res***
#####
#####
##### ***007 Improving generalization with regularization (1)***
#####
##### ***import numpy as np***
##### ***import pandas as pd***
##### ***from matplotlib import pyplot as plt***
#####
##### ***from keras.models import Sequential***
##### ***from keras.layers import Dense, Dropout***
##### ***from keras import regularizers***
##### ***# Dataset can be downloaded at [https://archive.ics.uci.edu/ml/machine-learning-databases/00275/***](https://archive.ics.uci.edu/ml/machine-learning-databases/00275/)***
##### ***data = pd.read\_csv('C:\\Users\\ifsrk\\Documents\\01 Deep Learning\\001 Handson\\hour.csv')***
##### ***data***
##### ***# Feature engineering***
##### ***ohe\_features = ['season', 'weathersit', 'mnth', 'hr', 'weekday']***
##### ***for feature in ohe\_features:***
##### `    `***dummies = pd.get\_dummies(data[feature], prefix=feature, drop\_first=False)***
##### `    `***data = pd.concat([data, dummies], axis=1)***
#####
##### ***data***
##### ***drop\_features = ['instant', 'dteday', 'season', 'weathersit',*** 
##### `                  `***'weekday', 'atemp', 'mnth', 'workingday', 'hr', 'casual', 'registered']***
##### ***data = data.drop(drop\_features, axis=1)***
##### ***data***
##### ***norm\_features = ['cnt', 'temp', 'hum', 'windspeed']***
##### ***scaled\_features = {}***
##### ***for feature in norm\_features:***
##### `    `***mean, std = data[feature].mean(), data[feature].std()***
##### `    `***scaled\_features[feature] = [mean, std]***
##### `    `***data.loc[:, feature] = (data[feature] - mean)/std***
#####
##### ***data***
##### ***data[-31\*24:]***
##### ***# Save the final month for testing***
##### ***# 744 rows***
##### ***test\_data = data[-31\*24:]***
##### ***data = data[:-31\*24]***
##### ***# Extract the target field***
##### ***target\_fields = ['cnt']***
##### ***features, targets = data.drop(target\_fields, axis=1), data[target\_fields]***
##### ***test\_features, test\_targets = test\_data.drop(target\_fields, axis=1), test\_data[target\_fields]***
##### ***# Create a validation set (based on the last )***
##### ***X\_train, y\_train = features[:-30\*24], targets[:-30\*24]***
##### ***X\_val, y\_val = features[-30\*24:], targets[-30\*24:]***
##### ***model = Sequential()***
##### ***model.add(Dense(250, input\_dim=X\_train.shape[1], activation='relu'))***
##### ***model.add(Dense(150, activation='relu'))***
##### ***model.add(Dense(50, activation='relu'))***
##### ***model.add(Dense(25, activation='relu'))***
##### ***model.add(Dense(1, activation='linear'))***
#####
##### ***# Compile model***
##### ***model.compile(loss='mse', optimizer='sgd', metrics=['mse'])***
##### ***n\_epochs = 4000***
##### ***batch\_size = 1024***
#####
##### ***history = model.fit(X\_train.values, y\_train['cnt'],*** 
##### `                 `***validation\_data=(X\_val.values, y\_val['cnt']),*** 
##### `                 `***batch\_size=batch\_size, epochs=n\_epochs, verbose=0***
##### `                `***)***
##### ***plt.plot(np.arange(len(history.history['loss'])), history.history['loss'], label='training')***
##### ***plt.plot(np.arange(len(history.history['val\_loss'])), history.history['val\_loss'], label='validation')***
##### ***plt.title('Overfit on Bike Sharing dataset')***
##### ***plt.xlabel('epochs')***
##### ***plt.ylabel('loss')***
##### ***plt.legend(loc=0)***
##### ***plt.show()***
##### ***print('Minimum loss: ', min(history.history['val\_loss']),*** 
##### `      `***'\nAfter ', np.argmin(history.history['val\_loss']), ' epochs')***
#####
##### ***# Minimum loss:  0.140975862741*** 
##### ***# After  730  epochs***
##### ***model\_reg = Sequential()***
##### ***model\_reg.add(Dense(250, input\_dim=X\_train.shape[1], activation='relu',***
##### `            `***kernel\_regularizer=reg***
##### ***ularizers.l2(0.005)))***
##### ***model\_reg.add(Dense(150, activation='relu'))***
##### ***model\_reg.add(Dense(50, activation='relu'))***
##### ***model\_reg.add(Dense(25, activation='relu',***
##### `            `***kernel\_regularizer=regularizers.l2(0.005)))***
##### ***model\_reg.add(Dense(1, activation='linear'))***
#####
##### ***# Compile model***
##### ***model\_reg.compile(loss='mse', optimizer='sgd', metrics=['mse'])’***
##### ***history\_reg = model\_reg.fit(X\_train.values, y\_train['cnt'],*** 
##### ***validation\_data=(X\_val.values, y\_val['cnt']),*** 
##### `                 `***batch\_size=batch\_size, epochs=n\_epochs, verbose=1***
##### `                `***)***
##### ***plt.plot(np.arange(len(history\_reg.history['loss'])), history\_reg.history['loss'], label='training')***
##### ***plt.plot(np.arange(len(history\_reg.history['val\_loss'])), history\_reg.history['val\_loss'], label='validation')***
##### ***plt.title('Use regularisation for Bike Sharing dataset')***
##### ***plt.xlabel('epochs')***
##### ***plt.ylabel('loss')***
##### ***plt.legend(loc=0)***
##### ***plt.show()***
##### ***print('Minimum loss: ', min(history\_reg.history['val\_loss']),*** 
##### `      `***'\nAfter ', np.argmin(history\_reg.history['val\_loss']), ' epochs')***
#####
##### ***# Minimum loss:  0.13514482975*** 
##### ***# After  3647  epochs***
#####
#####
##### **008 Adding dropout to prevent overfitting (1)**
#####
##### ***import numpy as np*** 
##### ***import pandas as pd***
##### ***from matplotlib import pyplot as plt***
#####
##### ***from keras.models import Sequential***
##### ***from keras.layers import Dense, Dropout***
#####
##### ***import numpy as np***
##### ***from matplotlib import pyplot as plt***
##### ***# Dataset can be downloaded at [https://archive.ics.uci.edu/ml/machine-learning-databases/00275/***](https://archive.ics.uci.edu/ml/machine-learning-databases/00275/)***
##### ***data = pd.read\_csv('C:\\Users\\ifsrk\\Documents\\01 Deep Learning\\001 Handson\\hour.csv')***
##### ***data***

\# Feature engineering

ohe\_features = ['season', 'weathersit', 'mnth', 'hr', 'weekday']

for feature in ohe\_features:

`    `dummies = pd.get\_dummies(data[feature], prefix=feature, drop\_first=False)

`    `data = pd.concat([data, dummies], axis=1)

data

drop\_features = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr', 'casual', 'registered']

data = data.drop(drop\_features, axis=1)

data

norm\_features = ['cnt', 'temp', 'hum', 'windspeed']

scaled\_features = {}

for feature in norm\_features:

`    `mean, std = data[feature].mean(), data[feature].std()

`    `scaled\_features[feature] = [mean, std]

`    `data.loc[:, feature] = (data[feature] - mean)/std

scaled\_features

\# Save the final month for testing

test\_data = data[-31\*24:]

data = data[:-31\*24]

\# Extract the target field

target\_fields = ['cnt']

features, targets = data.drop(target\_fields, axis=1), data[target\_fields]

test\_features, test\_targets = test\_data.drop(target\_fields, axis=1), test\_data[target\_fields]

\# Create a validation set (based on the last )

X\_train, y\_train = features[:-30\*24], targets[:-30\*24]

X\_val, y\_val = features[-30\*24:], targets[-30\*24:]

model = Sequential()

model.add(Dense(250, input\_dim=X\_train.shape[1], activation='relu'))

model.add(Dense(150, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(1, activation='linear'))

\# Compile model

model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

model.summary()

!pip install pydot

\# Visualize network architecture

import pydot

import pydotplus

import graphviz

from IPython.display import SVG

#from tensorflow.keras.utils.vis\_utils import model\_to\_dot

#from tensorflow.keras.utils.vis\_utils import plot\_model

from tensorflow.keras.utils import model\_to\_dot

from tensorflow.keras.utils import plot\_model

SVG(model\_to\_dot(model, show\_shapes=True).create(prog="dot", format="svg"))

\# Save the visualization as a file

plot\_model(model, show\_shapes=True, to\_file="dropout\_network\_model.png")

n\_epochs = 1000

batch\_size = 1024

history = model.fit(X\_train.values, y\_train['cnt'], 

`    `validation\_data=(X\_val.values, y\_val['cnt']), 

`    `batch\_size=batch\_size, epochs=n\_epochs, verbose=1

`    `)

plt.plot(np.arange(len(history.history['loss'])), history.history['loss'], label='training')

plt.plot(np.arange(len(history.history['val\_loss'])), history.history['val\_loss'], label='validation')

plt.title('Overfit on Bike Sharing dataset')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(loc=0)

plt.show()

print('Minimum loss: ', min(history.history['val\_loss']), 

` `'\nAfter ', np.argmin(history.history['val\_loss']), ' epochs')

\# Minimum loss:  0.129907280207 

\# After  980  epochs

model\_drop = Sequential()

model\_drop.add(Dense(250, input\_dim=X\_train.shape[1], activation='relu'))

model\_drop.add(Dropout(0.20))

model\_drop.add(Dense(150, activation='relu'))

model\_drop.add(Dropout(0.20))

model\_drop.add(Dense(50, activation='relu'))

model\_drop.add(Dropout(0.20))

model\_drop.add(Dense(25, activation='relu'))

model\_drop.add(Dropout(0.20))

model\_drop.add(Dense(1, activation='linear'))

\# Compile model

model\_drop.compile(loss='mse', optimizer='sgd', metrics=['mse'])

history\_drop = model\_drop.fit(X\_train.values, y\_train['cnt'], 

`    `validation\_data=(X\_val.values, y\_val['cnt']), 

`    `batch\_size=batch\_size, epochs=n\_epochs, verbose=1

`    `)

plt.plot(np.arange(len(history\_drop.history['loss'])), history\_drop.history['loss'], label='training')

plt.plot(np.arange(len(history\_drop.history['val\_loss'])), history\_drop.history['val\_loss'], label='validation')

plt.title('Use dropout for Bike Sharing dataset')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend(loc=0)

plt.show()

print('Minimum loss: ', min(history\_drop.history['val\_loss']), 

` `'\nAfter ', np.argmin(history\_drop.history['val\_loss']), ' epochs')

\# Minimum loss:  0.126063346863 

\# After  998  epochs

CNN

import tensorflow as tf

from tensorflow.keras import layers, models

\# Define the CNN model

def create\_cnn\_model():

` `model = models.Sequential()

` `# Convolutional layers

` `model.add(layers.Conv2D(32, (3, 3), activation='relu', input\_shape=(64, 64, 3)))

` `model.add(layers.MaxPooling2D((2, 2)))

` `model.add(layers.Conv2D(64, (3, 3), activation='relu'))

` `model.add(layers.MaxPooling2D((2, 2)))

` `model.add(layers.Conv2D(128, (3, 3), activation='relu'))

` `model.add(layers.Flatten())

` `# Dense layers

` `model.add(layers.Dense(128, activation='relu'))

` `model.add(layers.Dense(10, activation='softmax'))

` `return model

\# Create the CNN model

cnn\_model = create\_cnn\_model()

\# Compile the model

cnn\_model.compile(optimizer='adam',

` `loss='sparse\_categorical\_crossentropy',

` `metrics=['accuracy'])

\# Display the model architecture

cnn\_model.summary()

LSTM

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense

import numpy as np

input\_seq = np.array([[i] for i in range(10)], dtype=np.float32)

target\_seq = np.array([[i\*2] for i in range(10)], dtype=np.float32)

\# Reshape the input data to be 3D (batch\_size, timesteps, input\_dim)

input\_seq = input\_seq.reshape((1, 10, 1))

target\_seq = target\_seq.reshape((1, 10, 1))

\# Define the LSTM model

model = Sequential()

model.add(LSTM(10, activation='relu', input\_shape=(10, 1),

return\_sequences=True))

\# Add a Dense layer with 1 unit (output size)

model.add(Dense(1))

\# Compile the model

model.compile(optimizer='adam', loss='mean\_squared\_error')

\# Train the model

model.fit(input\_seq, target\_seq, epochs=10, batch\_size=1)

\# Predict using the trained model

predictions = model.predict(input\_seq)

\# Display the predictions

print("Predictions:")

for i in range(len(input\_seq[0])):

` `print("Input:", input\_seq[0, i, 0], "Predicted:", predictions[0, i, 0])

rnn

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import SimpleRNN, Dense, TimeDistributed

import numpy as np

\# Generate some sample data for training (sequence of numbers)

input\_seq = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)

target\_seq = np.array([[2], [3], [4], [5], [6]], dtype=np.float32)

\# Define the RNN model

model = Sequential()

model.add(SimpleRNN(10, activation='relu', return\_sequences=True,

input\_shape=(None, 1)))

model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mean\_squared\_error')

\# Train the model

model.fit(input\_seq.reshape((1, 5, 1)), target\_seq.reshape((1, 5, 1)), epochs=10,

batch\_size=1)

\# Predict using the trained model

predictions = model.predict(input\_seq.reshape((1, 5, 1)))

\# Display the predictions

print("Predictions:")

for i in range(len(input\_seq)):

` `print("Input:", input\_seq[i].tolist(), "Predicted:", predictions[0, i, 0])
