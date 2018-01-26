import numpy as np
import pandas as pd
import time
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
batch_size = 128
n_filter1 = 32
n_filter2 = 64
n_flat = 256

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

model = Sequential()
model.add(Conv2D(filters=n_filter1, kernel_size=(5,5),padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=n_filter1, kernel_size=(5,5),padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=n_filter2, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=n_filter2, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(n_flat, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

optimizer = RMSprop(lr=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
datagen = ImageDataGenerator(
    zoom_range = 0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1)

datagen.fit(X_train)    #no needed here 只有使用featurewise_center，featurewise_std_normalization或zca_whitening时需要此函数
start = time.time()
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size, seed=random_seed),
                              epochs = n_epoch, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] / batch_size      #datagen will produce 7times of original data, so *2 to train more generated batch
                              , callbacks=[learning_rate_reduction])
#model.fit(X_train, Y_train, epochs=n_epoch, batch_size=batch_size)
end = time.time()
score = model.evaluate(X_val, Y_val, batch_size=batch_size)
print("train time(min):", (end-start)/60)
print("validation score: ", score)
results = model.predict(test)
results = np.argmax(results, axis=1)

submission = pd.DataFrame({'ImageId':range(1,28001), 'Label': results})
name = "cnn_mnist_{}_{}_{}_{}.csv".format(n_epoch, n_filter1, n_filter2, n_flat)
submission.to_csv(name, index=False)
