import csv
import cv2 
import numpy as np
import random as rnd
from enum import Enum
import time
import matplotlib.pyplot as plt

# https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

lines = []
for file in [ 'data/christian_track_2.csv', 'data/driving_log.csv', 'data/driving_log_track2_com.csv',]:
             #'data/driving_log_track_sharp_turn.csv', 'data/driving_log_track2_uphill.csv',
             #'data/driving_log_track2_bridgedown.csv', 'data/driving_log_track2_downhill_again.csv', 'data/driving_log_track2_downhill_again2.csv'  #
 
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line) 


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Conv2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
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
    Shadow = 4


class ImageSelector(Enum):
    Center = 0,
    Left = 1,
    Right = 2,

class ImageDataGeneratorMode(Enum):
    Train = 0,
    Validation = 1

class ShadowType(Enum):
    Bright = 0,
    Dark = 1

class BrightnessType(Enum):
    Bright = 0,
    Dark = 1

# https://www.kaggle.com/raghakot/ultrasound-nerve-segmentation/easier-keras-imagedatagenerator
class ImageDataGenerator(Iterator):
    def __init__(self, csv_lines, mode = ImageDataGeneratorMode.Train, batch_size=128, shuffle=True, seed=None):

        self.mode = mode

        self.csv = csv_lines = np.array(csv_lines)

        # calculate histogram  
        hist, bin_edges = np.histogram(np.array(csv_lines[:,3],dtype=float), bins=51, density=False)
        max_index = np.argmax(hist)

        hist_array = np.sort(hist)[::-1]
        # select lines with the values arround zero  np.take(zero_list,[1,2,5],axis=1)
        zero_list = [line for line in lines if float(line[3]) > bin_edges[max_index-1] and float(line[3]) < bin_edges[max_index+1]]
        index_list = np.random.random_integers(0,hist_array[0]-1,hist_array[1])
        # take the same number of near zero lines as the second most angle range
        zero_list = np.take(zero_list,index_list,axis=0)
        non_zero_list = np.array([line for line in lines if float(line[3]) < bin_edges[max_index-1] or float(line[3]) > bin_edges[max_index+1]])
        #non_zero_list = np.array([line for line in lines if ((float(line[3]) < bin_edges[max_index-1] and  float(line[3]) > -0.95) 
        #                                                      or (float(line[3]) > bin_edges[max_index+1] and float(line[3]) < 0.95))])
                                  #and  (float(line[3]) > -0.95 or float(line[3]) < 0.95)])

        self.csv = np.concatenate((zero_list,non_zero_list))

        super(ImageDataGenerator,self).__init__(self.csv.shape[0],batch_size,shuffle,seed)

    def next(self):

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros(shape=(3*current_batch_size, IMAGE_ROW, IMAGE_COL, IMAGE_CH), dtype=np.float32)
        batch_y = np.zeros(shape=(3*current_batch_size))

        batch_index = 0
        start_time = time.time()
        for array_index in index_array:
            #choice = rnd.choice([ImageSelector.Left,ImageSelector.Right,ImageSelector.Center])

            for choice in ImageSelector:

                source_path = self.csv[array_index][choice.value[0]]
                filename = source_path.split('/')[-1]
                current_path = 'C:/datasets/data/' + filename  # 'G:/Training/TrainingsDatensatz/' +  '' +'data/IMG/'+
                image = cv2.imread(current_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

                angle =  float(self.csv[array_index][3])
                      
                # if the loaded image is a left carmera image increase angle by 0.25
                if choice == ImageSelector.Left:
                    if angle + 0.25 > 1:
                        angle = 1
                    else: 
                        angle = angle + 0.25
                # if the loaded image is a right carmera image deincrease angle by 0.25
                elif choice == ImageSelector.Right:
                    if angle - 0.25 < -1:
                        angle = -1
                    else: 
                        angle = angle - 0.25

                if self.mode == ImageDataGeneratorMode.Train:
                    image, angle = self.select_random_argumentation(image,angle)

                batch_x[batch_index,:,:,:] =  (image / 255.) - 0.5
                batch_y[batch_index] = angle
                batch_index += 1

        elapsed_time = time.time() - start_time
        #print ("elapsed_time: {0}".format(elapsed_time))
        return batch_x,batch_y

    def select_random_argumentation(self,image, angle):
        choiceList = rnd.sample([DataArgumentationOptions.Unchanged,
                             DataArgumentationOptions.Flipp,
                             DataArgumentationOptions.Brightness,
                             DataArgumentationOptions.Shadow], rnd.randint(1,3)) 
                              

        for choice in choiceList:
            if choice == DataArgumentationOptions.Unchanged:
                break
            elif choice == DataArgumentationOptions.Brightness:
                image = self.random_brightness(image)
            elif choice == DataArgumentationOptions.Flipp:
                image, angle = self.random_flipp(image,angle)
            elif choice == DataArgumentationOptions.Shadow:
                image = self.random_shadow(image)

        return image, angle
    
    def random_brightness(self, image, lower_range = 0.4, upper_range = 0.6):

        brightness_image = np.zeros(image.shape[0:2],np.uint8)
        if np.random.choice([BrightnessType.Bright,BrightnessType.Dark]) == ShadowType.Bright:
            brightness_image[:,:] = 255

        brightness_alpha = np.random.uniform(lower_range,upper_range)
        image[:,:,0] = cv2.addWeighted(image[:,:,0],1. - brightness_alpha,brightness_alpha,brightness_alpha,0)
        return image 

    def random_flipp(self,image,angle):
        return cv2.flip(image,1), angle * -1

    def random_shadow(self, image, alpha_range = 0.2):
        
        shadow_image = np.zeros(image.shape[0:2],np.uint8)

        shadow_border_points_count = 4
        shape_points = np.random.uniform(0,image.shape[0],(shadow_border_points_count,2)).astype(np.int32) + (np.random.randint(0,image.shape[0]),0) 
        cv2.fillConvexPoly(shadow_image, shape_points, 255)

        if np.random.choice([ShadowType.Bright,ShadowType.Dark]) == ShadowType.Bright:
            shadow_image = 255 - shadow_image
        
        shadow_alpha = np.random.uniform(0,alpha_range)
        image[:,:,0] = cv2.addWeighted(image[:,:,0],1. - shadow_alpha,shadow_image,shadow_alpha,0)

        return image
                                                     

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
    model.add(Dropout(0.5))
    model.add(Dense(84))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    
    model.add(Dense(1))
    return model

# https://github.com/commaai/research
def model_commaai():

    input_shape = (IMAGE_ROW,IMAGE_COL, IMAGE_CH)

    model = Sequential()
    model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape = input_shape))
    model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same")) 
    model.add(ELU())
    model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
  
    return model

def model_commaai_small():

    input_shape = (IMAGE_ROW,IMAGE_COL, IMAGE_CH)

    model = Sequential()
    model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape = input_shape))
    model.add(Convolution2D(4, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(8, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(125))
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
validation_generator = ImageDataGenerator(lines,ImageDataGeneratorMode.Validation) #Validation


anlges = []
array_index = 0
for image_batch, angle_batch in train_generator:
    for image, angle in zip(image_batch, angle_batch):
        anlges.append(angle)

        image = ((image + 0.5) * 255).astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
        cv2.arrowedLine(image,(160,320),(int(160+50 * (3 * angle)), 80),(0,0,255),3) # int(80+50*angle)
        cv2.imwrite("temp/test_{0:07d}.png".format(array_index),image)
        array_index += 1
        
    break;
    
n, bins, patches = plt.hist(anlges, 50, normed=1, facecolor='green', alpha=0.75)

plt.axis([min(anlges), max(anlges), 0, max(n)])
plt.show()


model = model_commaai()


print( model.summary())


model.compile(loss='mse', optimizer=Adam(lr=0.001))

earlyStopping = EarlyStopping(monitor='val_loss', min_delta= 0.001, patience=4, verbose=2, mode='min')
checkpoint = ModelCheckpoint('model_8.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
csv_logger = CSVLogger('train_stats.csv')

callbacks_list = [checkpoint, earlyStopping, csv_logger]

history = model.fit_generator(train_generator,
                    validation_data= validation_generator,
                    samples_per_epoch = len(lines), # 20000
                    callbacks=callbacks_list,
                    nb_epoch=15,
                    validation_steps=10) 


def plot_history(history):

    ax1 = plt.plot()
    plt.title('loss')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss' ], loc='upper left')

    plt.show()

plot_history(history)