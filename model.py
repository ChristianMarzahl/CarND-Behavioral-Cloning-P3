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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.layers.normalization import BatchNormalization

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

# https://srikanthpagadala.github.io/serve/carnd-behavioral-cloning-p3-report.html
# https://www.kaggle.com/raghakot/ultrasound-nerve-segmentation/easier-keras-imagedatagenerator
class ImageDataGenerator(Iterator):
    def __init__(self, csv_lines, mode = ImageDataGeneratorMode.Train, batch_size=256, shuffle=True, seed=None):

        self.mode = mode

        csv_lines = np.array(csv_lines)

        # calculate histogram  
        hist, bin_edges = np.histogram(np.array(csv_lines[:,3],dtype=float), bins=50, density=False)
        max_index = np.argmax(hist)

        hist_array = np.sort(hist)[::-1]
        # select lines with the values arround zero  np.take(zero_list,[1,2,5],axis=1)
        zero_list = [line for line in lines if float(line[3]) > bin_edges[max_index-1] and float(line[3]) < bin_edges[max_index+1]]
        index_list = np.random.random_integers(0,hist_array[0]-1,3 * hist_array[1])
        # take the same number of near zero lines as the second most angle range
        zero_list = np.take(zero_list,index_list,axis=0)
        non_zero_list = np.array([line for line in lines if float(line[3]) < bin_edges[max_index-1] or float(line[3]) > bin_edges[max_index+1]])
        self.csv = np.concatenate((zero_list,non_zero_list))

        super(ImageDataGenerator,self).__init__(self.csv.shape[0],batch_size,shuffle,seed)

    def next(self):

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros(shape=(current_batch_size, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float32)
        batch_y = np.zeros(shape=(current_batch_size))

        batch_index = 0
        for array_index in index_array:
            choice = rnd.choice([ImageSelector.Left,ImageSelector.Right,ImageSelector.Center]) #ImageSelector.Left,ImageSelector.Center,

            if self.mode == ImageDataGeneratorMode.Validation:
                choice = ImageSelector.Center

            source_path = self.csv[array_index][choice.value[0]]
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/'+filename
            image = cv2.imread(current_path)

            angle =  float(self.csv[array_index][3])
                      
            # if the loaded image is a left carmera image increase angle by 0.25
            if choice == ImageSelector.Left:
                angle = angle + 0.25
            # if the loaded image is a right carmera image deincrease angle by 0.25
            elif choice == ImageSelector.Right:
                angle = angle - 0.25

            if self.mode == ImageDataGeneratorMode.Train:
                image, angle = self.select_random_argumentation(image,angle)

            #cv2.arrowedLine(image,(160,80),(int(160+50*angle), 80),(0,0,255),3) # int(80+50*angle)
            #cv2.imwrite("temp/test_{0:07d}.png".format(array_index),image)

            batch_x[batch_index,:,:,:] = (image / 255.) - 0.5
            batch_y[batch_index] = angle
            batch_index += 1

        return batch_x,batch_y

    def select_random_argumentation(self,image, angle):

        choice = rnd.choice([DataArgumentationOptions.Unchanged,DataArgumentationOptions.Flipp, DataArgumentationOptions.Brightness,DataArgumentationOptions.BrightnessAndFlipp]) 

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
        return cv2.flip(image,1), angle * -1

                                                     
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


def model_udacity():

    input_shape = (IMAGE_ROW,IMAGE_COL, IMAGE_CH)
    model = Sequential()

    model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape = input_shape))
    model.add(Convolution2D(6, 5, 5, activation="relu", name = 'conv_2')) #subsample=(2,2),
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu",  name = 'conv_3')) #subsample=(2,2),
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(120))
    #model.add(Dropout(0.5))
    model.add(Dense(84))
    #model.add(Dropout(0.5))
    model.add(Dense(10))
    
    model.add(Dense(1))
    return model

# https://github.com/commaai/research
def model_commaai():

    input_shape = (IMAGE_ROW,IMAGE_COL, IMAGE_CH)

    model = Sequential()
    model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape = input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
  
    return model



# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
def model_nvidia():
    input_shape = (IMAGE_ROW,IMAGE_COL, IMAGE_CH)
    drop_out = 0.5

    model = Sequential()
    
    # exception while saving the checkpoint!!! 
    #model.add(Lambda(lambda x : (x/255.) -0.5, input_shape = input_shape, name = 'Normalizer'))
    
    model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape = input_shape))
    
    model.add(Convolution2D(3,1,1,activation="relu",name='Color_layer'))
    
    model.add(Convolution2D(24, 5, 5, activation="relu", subsample=(2,2),init = 'he_normal', name = 'conv_2'))
    model.add(BatchNormalization())

    model.add(Convolution2D(36, 5, 5, activation="relu", subsample=(2,2),init = 'he_normal', name = 'conv_3'))
    model.add(BatchNormalization())

    model.add(Convolution2D(48, 5, 5, activation="relu", subsample=(2,2),init = 'he_normal', name = 'conv_4'))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, 3, 3, activation="relu", name = 'conv_5'))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, 3, 3, activation="relu", name = 'conv_6'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(ELU()) #https://keras.io/layers/advanced_activations/ 

    model.add(Dense(100))
    model.add(Dropout(0.2))

    model.add(Dense(50))

    model.add(ELU())
    model.add(Dense(16))
    
    model.add(Dense(1))
    return model

train_generator = ImageDataGenerator(lines,ImageDataGeneratorMode.Train)
validation_generator = ImageDataGenerator(lines,ImageDataGeneratorMode.Validation)


anlges = []
counter = 0
for image_batch, angle_batch in train_generator:
    for angle in angle_batch:
        anlges.append(angle)
    counter += 1

    if(counter == 20):
        break;
    
n, bins, patches = plt.hist(anlges, 50, normed=1, facecolor='green', alpha=0.75)

plt.axis([min(anlges), max(anlges), 0, max(n)])
plt.show()


model = model_commaai()
model.compile(loss='mse', optimizer='adam')


checkpoint = ModelCheckpoint('model_4.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


#history = model.fit(X_train,y_train, validation_split=0.2,callbacks=callbacks_list)
#print(history)

history = model.fit_generator(train_generator,
                    validation_data= validation_generator,
                    samples_per_epoch = len(lines), 
                    callbacks=callbacks_list,
                    nb_epoch=20,
                    nb_val_samples=len(lines)*0.2) 

plot_history(history)