import csv
import cv2 
import numpy as np

# https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+filename # data/IMG/
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)



X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import Iterator

IMAGE_ROW = 160
IMAGE_COL = 320
IMAGE_CH = 3

# https://www.kaggle.com/raghakot/ultrasound-nerve-segmentation/easier-keras-imagedatagenerator
class ImageDataGenerator(Iterator):
    def __init__(self, csv_lines, batch_size=32, shuffle=True, seed=None):
        
        self.csv = csv_lines

        super(ImageDataGenerator,self).__init__(len(csv_lines),batch_size,shuffle,seed)

    def next(self):

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros(shape=(current_batch_size, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.uint8)
        batch_y = np.zeros(shape=(current_batch_size))

        return batch_x,batch_y



def model_simple():

    input_shape = (IMAGE_ROW,IMAGE_COL, IMAGE_CH)

    model = Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(1))

    return model

# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
def model_nvidia():
    input_shape = (IMAGE_ROW,IMAGE_COL, IMAGE_CH)
    drop_out = 0.5
    max_pool = (2, 2)

    kernel_size = (3, 3)

    model = Sequential()
    
    # exception while saving the checkpoint!!! 
    #model.add(Lambda(lambda x : (x/255.) -0.5, input_shape = input_shape, name = 'Normalizer'))
    
    model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape = input_shape))
    
    model.add(Convolution2D(32, 5, 5,
                        border_mode='valid',
                        name = 'conv_1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = max_pool))

    model.add(Convolution2D(16, 5, 5,
                        border_mode='valid',
                        name='conv_2'))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3,
                        border_mode='valid',
                        name='conv_3'))
                            
    model.add(Activation('relu'))
    model.add(Convolution2D(4, 3, 3,
                        border_mode='valid',
                        name='conv_4'))
    
    model.add(Activation('relu'))


    model.add(Flatten())
    
    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))
    
    model.add(Dense(1))
    return model


train_generator = ImageDataGenerator(lines)

model = model_nvidia()
model.compile(loss='mse', optimizer='adam')


checkpoint = ModelCheckpoint('model_1.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit_generator(train_generator,samples_per_epoch = len(X_train), callbacks=callbacks_list,nb_epoch=5) #
#model.fit(X_train,y_train,validation_split=0.2, callbacks=callbacks_list,shuffle=True, nb_epoch=5)
