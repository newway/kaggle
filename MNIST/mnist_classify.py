import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

def LoadData():
    K = 42000
    Train = pd.read_csv("./input/train.csv", nrows=K)
    Test = pd.read_csv("./input/test.csv")
    target = Train['label']
    Train.drop('label', axis=1, inplace=True)
    return Train[:K], Test, target[:K]

train,test,target = LoadData()
print(train.shape, test.shape, train.info())
print(target.value_counts())
print(train.isnull().any().describe())
print(test.isnull().any().describe())
train = train / 255.0
test = test / 255.0

X_train = train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)
Y_train = to_categorical(target, num_classes=10)

random_seed = 2
n_epoch = 30
batch_size = 50
n_filter1 = 32
n_filter2 = 64

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

model = Sequential()
model.add(Conv2D(filters=n_filter1, kernel_size=(5,5),padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=n_filter2, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

optimizer = RMSprop(lr=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=n_epoch, batch_size=batch_size)
score = model.evaluate(X_val, Y_val, batch_size=batch_size)

results = model.predict(test)
results = np.argmax(results, axis=1)

submission = pd.DataFrame({'ImageId':range(1,28001), 'Label': results})
name = "cnn_mnist_{}_{}_{}.csv".format(n_epoch, n_filter1, n_filter2)
submission.to_csv(name, index=False)
