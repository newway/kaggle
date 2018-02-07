import os, sys, random, warnings
import numpy as np
import pandas as pd
import time
import datetime
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.filters import threshold_otsu
from skimage.util import img_as_bool
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

TRAIN_PATH = './stage1_train/'
TEST_PATH = './stage1_test/'
train_ids = next(os.walk(TRAIN_PATH))[1]    #root, dirs, files
test_ids = next(os.walk(TEST_PATH))[1]    #root, dirs, files
print(len(train_ids), len(test_ids))

def load_data(train_path, test_path, img_size):
    start = time.time()
    X_train = np.zeros((len(train_ids), img_size, img_size, IMG_CHANNELS))  #, dtype=np.uint8
    Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)
    for i, _id in enumerate(train_ids):
        path= train_path + _id
        img = imread(path+'/images/'+_id+'.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (img_size, img_size))     #[0, 255]->[0.0, 1.0]
        X_train[i] = img
        #print(np.unique(img, return_counts=True))
        #print(np.unique(X_train[0], return_counts=True))
        mask = np.zeros((img_size, img_size, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)    # shape (256, 256), values: 0,255
            #print(np.unique(mask_, return_counts=True))
            mask_ = resize(mask_, (img_size, img_size))   # preserve_range=False make (256, 256) 255.0 ->value range [0.1-1.0]
            mask_ = mask_[:, :, np.newaxis]     #== np.expand_dims(mask_, axis=-1)
            mask_ = img_as_bool(mask_)
            #print(np.unique(mask_, return_counts=True))
            mask = np.maximum(mask, mask_)      
        Y_train[i] = mask
    X_test = np.zeros((len(test_ids), img_size, img_size, IMG_CHANNELS))
    shapes_test = []
    for i, _id in enumerate(test_ids):
        path = test_path + _id
        img = imread(path+'/images/'+_id+'.png')[:,:,:IMG_CHANNELS]
        shapes_test.append(img.shape)
        X_test[i] = resize(img, (img_size, img_size))

    end = time.time()
    print("load data time: ", (end-start)/60.0)
    return X_train, Y_train, X_test, shapes_test

def Unet(img_size, cutoff):
    inputs = Input((img_size, img_size, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])

    return model

def generator(xtr, xval, ytr, yval, bt_size):
    # we create two instances with the same arguments
    data_gen_args = dict(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    SEED = 7
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(xtr) 
    mask_datagen.fit(ytr)   #, augment=True, seed=7
    image_generator = image_datagen.flow(xtr, batch_size=bt_size, seed=SEED)
    mask_generator = mask_datagen.flow(ytr, batch_size=bt_size, seed=SEED)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(xval)
    mask_datagen_val.fit(yval)
    image_generator_val = image_datagen_val.flow(xval, batch_size=bt_size, seed=SEED)
    mask_generator_val = mask_datagen_val.flow(yval, batch_size=bt_size, seed=SEED)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator

#y_pred: IOU rate
def mean_iou_val(y_true, y_pred):
    y_true_ = tf.reshape(y_true, [-1])      #bool
    y_true_ = tf.to_float(y_true_)
    y_pred_ = tf.reshape(y_pred, [-1])      # probability
    inter = tf.reduce_sum(tf.multiply(y_true_, y_pred_))
    union = tf.reduce_sum(tf.subtract( tf.add(y_true_, y_pred_), tf.multiply(y_true_, y_pred_) ))

    return tf.div(inter, union)
    #y_true_ = tf.reshape(y_true, [-1])
    #y_true_ = tf.to_int32(y_true_)
    #y_pred_ = tf.to_int32(y_pred > 0.5)
    #y_pred_ = tf.reshape(y_pred_, [-1])
    #
    #return tf.metrics.mean_iou(y_true_, y_pred_, 2)
def mean_iou_loss(y_true, y_pred):
    return tf.subtract(tf.constant(1.0), mean_iou_val(y_true, y_pred))

#calc approximately IOU, input shape batch_size*H*W
def get_piou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

    #iou, upd = batch_iou(y_true, y_pred)
    #return iou

    #return 1.0 - my_loss(y_true, y_pred)

def my_loss(y_true, y_pred):
    y_true_ = tf.to_float(y_true)     #0-255
    #y_true_ = tf.reshape(y_true, [-1])
    #y_pred_ = tf.reshape(y_pred, [-1])
    y_pred_ = tf.to_float(y_pred)     #0-1
    smooth = 1
    intersection = tf.reduce_sum(tf.multiply(y_pred_ , y_true_))
    union = tf.reduce_sum(tf.subtract( tf.add(y_true_, y_pred_), tf.multiply(y_true_, y_pred_) ))  
    #return (smooth + intersection)/(union + smooth)
    return 1.0 - tf.reduce_mean(tf.div(tf.add(1.0,intersection), tf.add(1.0, union)))

    #return tf.subtract(tf.constant(1.0, dtype=tf.float32), get_piou(y_true, y_pred))

def rle_encoding(x):
    dots = np.where(x.T.flatten()==1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    #or rescale to (0,1)
    m = np.amax(x)
    if (cutoff >= m):
        cutoff = m*0.5
    lab_img = label(x > cutoff)
    #while(lab_img.max()<1):
    #    cutoff = cutoff - 0.05
    #    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

if __name__ == '__main__':
    IMG_CHANNELS = 3
    IMG_SIZE = 256
    SEED = 7
    #random.seed = SEED
    #np.random.seed = SEED
    N_EPOCHS = 50
    BATCH_SIZE = 32
    print(IMG_SIZE, SEED, N_EPOCHS, BATCH_SIZE)
    X_train, Y_train, X_test, sizes_test = load_data(TRAIN_PATH, TEST_PATH, IMG_SIZE)
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=7)
    #xtr, xval = X_train[:int(X_train.shape[0]*0.9)], X_train[int(X_train.shape[0]*0.9):]
    #ytr, yval = Y_train[:int(Y_train.shape[0]*0.9)], Y_train[int(Y_train.shape[0]*0.9):]
    print(np.shape(xtr), np.shape(ytr), np.shape(xval), np.shape(yval))
    train_generator,val_generator = generator(xtr, xval, ytr, yval, BATCH_SIZE)
    #print(np.any(Y_train[0]), np.sum(Y_train[0]))
    #print(np.any(ytr[0]),np.any(ytr[1]), np.sum(ytr[1]), np.shape(ytr[0]))

    #model = load_model('final_model-256-30-16-12.18.h5', custom_objects={"mean_iou_loss": mean_iou_loss, "mean_iou_val": mean_iou_val})
    
    model = Unet(IMG_SIZE, 0.5)
    model.compile(optimizer='adam', loss=mean_iou_loss, metrics=[mean_iou_val])
    start = time.time()
    earlystopper = EarlyStopping(patience=5, verbose=1)
    model_name = 'median-{}-{}-{}.h5'.format(IMG_SIZE, N_EPOCHS, BATCH_SIZE)
    checkpointer = ModelCheckpoint(model_name, verbose=1, save_best_only=True)
    #model.fit(xtr, ytr, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(xval,yval), verbose=2, callbacks=[earlystopper, checkpointer])
    model.fit_generator(train_generator, validation_data=val_generator, validation_steps=len(xval)//BATCH_SIZE, steps_per_epoch=len(xtr)//BATCH_SIZE, epochs=N_EPOCHS,  verbose=2, callbacks=[earlystopper, checkpointer])
    end = time.time()
    print("model training total time(min):", (end-start)/60.0)
    now = datetime.datetime.now()
    #print now.year, now.month, now.day, now.hour, now.minute, now.second
    model.save('final_model-{}-{}-{}-{}.{}.h5'.format(IMG_SIZE, N_EPOCHS, BATCH_SIZE, now.hour, now.minute))
    #json_string = model.to_json()

    preds_test = model.predict(X_test, verbose=1)
    #print(tf.count_nonzero(tf.greater(preds_test[0], tf.constant(0.5))))
    #print(tf.shape(preds_test[0]))
    #exit()
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1])))
    #print(tf.shape(preds_test_upsampled))

    new_test_ids = []
    rles = []
    cutoff = 0.5
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n], cutoff=cutoff))  #rle_encoding(preds_test_upsampled[n]) 
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    print(new_test_ids[0], rles[0])
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    name = 'dsbowl2018-{}-{}-{}-{}.csv'.format(IMG_SIZE, N_EPOCHS, BATCH_SIZE, cutoff)
    sub.to_csv(name, index=False)
