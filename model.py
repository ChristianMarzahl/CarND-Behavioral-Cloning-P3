import csv
import cv2 
import numpy as np
import random as rnd
from enum import Enum

# https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    lines = lines[1:]


image_counter = 0
measurements = []
images = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+filename # data/IMG/
    image = cv2.imread(current_path)
    images.append(image)
    y = measurement = float(line[3])
    measurements.append(measurement)
    #cv2.arrowedLine(image,(160,80),(int(160+50*y),80),(0,0,255),3)
    #cv2.imwrite("temp/test_{0:07d}.png".format(image_counter),image)

    image_counter += 1

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

class DataArgumentationOptions(Enum):
    Unchanged = 1,
    Brightness = 2,
    Flipp = 3,
    BrightnessAndFlipp = 4


class ImageSelector(Enum):
    Center = 0,
    Left = 1,
    Right = 2,

class ImageDataGeneratorMode(Enum):
    Train = 0,
    Validation = 1

# https://www.kaggle.com/raghakot/ultrasound-nerve-segmentation/easier-keras-imagedatagenerator
class ImageDataGenerator(Iterator):
    def __init__(self, csv_lines, mode = ImageDataGeneratorMode.Train, batch_size=128, shuffle=True, seed=None):

        self.mode = mode
        self.csv = csv_lines

        super(ImageDataGenerator,self).__init__(len(csv_lines),batch_size,shuffle,seed)

    def next(self):

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros(shape=(current_batch_size, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.uint8)
        batch_y = np.zeros(shape=(current_batch_size))

        batch_index = 0
        for array_index in index_array:
            choice = rnd.choice([ImageSelector.Left,ImageSelector.Center,ImageSelector.Center]) #ImageSelector.Left,ImageSelector.Center,

            if self.mode == ImageDataGeneratorMode.Validation:
                choice = ImageSelector.Center

            source_path = self.csv[array_index][choice.value[0]]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/'+filename
            image = cv2.imread(current_path)

            angle =  float(self.csv[array_index][3])
                      
            # if the loaded image is a left carmera image increase angle by 0.25
            if choice == ImageSelector.Left:
                angle = angle - 0.25
            # if the loaded image is a right carmera image deincrease angle by 0.25
            elif choice == ImageSelector.Right:
                angle = angle + 0.25

            if self.mode == ImageDataGeneratorMode.Train:
                image, angle = self.select_random_argumentation(image,angle)

            #cv2.arrowedLine(image,(160,80),(int(160+50*angle), 80),(0,0,255),3) # int(80+50*angle)
            #cv2.imwrite("temp/test_{0:07d}.png".format(array_index),image)

            batch_x[batch_index,:,:,:] = image
            batch_y[batch_index] = angle
            batch_index += 1

        return batch_x,batch_y

    def select_random_argumentation(self,image, angle):

        # ,DataArgumentationOptions.Flipp,DataArgumentationOptions.BrightnessAndFlipp
        choice = rnd.choice([DataArgumentationOptions.Unchanged,DataArgumentationOptions.Brightness])

        if choice == DataArgumentationOptions.Brightness:
            return self.random_brightness(image), angle
        elif choice == DataArgumentationOptions.Flipp:
            return self.random_flipp(image,angle)
        elif choice == DataArgumentationOptions.BrightnessAndFlipp:
            image = self.random_brightness(image)
            return self.random_flipp(image,angle)
        return image, angle
    
    def random_brightness(self,image, lower_range = 0.5, upper_range = 0.5):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert image to hsv color space
        hsv[:,:,2] = hsv[:,:,2] * (lower_range + rnd.uniform(0.0,upper_range))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def random_flipp(self,image,angle):
        return np.fliplr(image), -angle

                                                     
import matplotlib.pyplot as plt
def plot_history(history):

    ax1 = plt.plot()
    plt.title('loss')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss' ], loc='upper left')

    plt.show()


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
    
    model.add(Convolution2D(3,1,1,
                        border_mode='valid',
                        name='Color_layer', init='he_normal'))
    
    model.add(Convolution2D(32, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        name = 'conv_2', init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = max_pool))

    model.add(Convolution2D(16, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        name='conv_3', init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        name='conv_4', init='he_normal'))
                            
    model.add(Activation('relu'))
    model.add(Convolution2D(4, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        name='conv_5', init='he_normal'))
    
    model.add(Activation('relu'))


    model.add(Flatten())
    
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    return model

train_generator = ImageDataGenerator(lines)
validation_generator = ImageDataGenerator(lines,ImageDataGeneratorMode.Validation)

model = model_nvidia()
model.compile(loss='mse', optimizer='adam')


checkpoint = ModelCheckpoint('model_1.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


history = model.fit(X_train,y,validation_split=0.2,callbacks=callbacks_list)

#history = model.fit_generator(train_generator,
#                    validation_data= validation_generator,
#                    samples_per_epoch = len(lines), 
#                    callbacks=callbacks_list,
#                    nb_epoch=5, 
#                    nb_val_samples=len(lines)*0.25) 

plot_history(history)