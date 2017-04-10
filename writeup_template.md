# **Behavioral Cloning** 


[//]: # (Image References)

[image1]: ./examples/biased_dataset.png "Biased Dataset"
[image2]: ./examples/DataAugmentation2.png "Data Augmentation"
[image10]: ./examples/unchanged.png "Road Image"
[image11]: ./examples/bright_image_with_shadow.png "bright image with shadow"
[image12]: ./examples/bright_shadow.png "bright shadow"
[image13]: ./examples/dark_image.png "dark image"
[image14]: ./examples/dark_shadow.png "dark shadow"

[image20]: ./examples/test_2017_04_08_10_57_22_172.png "Problem Image"

[image30]: ./examples/loss_function.png "Loss Function"

[image40]: ./gif/DataAugmentationSlow.gif "Data Augmentation"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model and performe the data augmentation 
* drive.py for driving the car in autonomous mode
* final_model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py final_model.h5
```

#### 3. Submission code

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. In addition the file contains the code for the image data generator to argument the training images

##### 4. ImageDataGenerator 

Tue to memory restriciton and efficiency I used the fit_generator and performed the data augmentation as needed and not in advance.
Because the training data is heavily biased towords a centered stering angle as shown in the following image ![Biased Data][image1]. 
I used a couple of augmentation approaches to compensate for that. 

1. ##### Stering angle histogramm
I generated from all angles a histogramm with 51 bins. And selected just as many images with with most common stering angle as are in the second bin. The effect of this transformation with additional augmentation is shown in the next image. ![Data Augmentation][image2]

```python
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
```

2. ##### Flip 
The first track is a circle with a high bias towards left stering angles. By randomly fliping images the effect is filtered out.

```python
    def random_flipp(self,image,angle):
        return cv2.flip(image,1), angle * -1
```

3. ##### Brightness and Shadows
To make the model more robust to brightness changes and shadows on the road I added two functions to add random brightness and shadows to images.  

![bright image with shadow][image11]
![bright shadow][image12]
![dark image][image13]
![dark shadow][image14]


```python
    def random_brightness(self, image, lower_range = 0.4, upper_range = 0.6):

        brightness_image = np.zeros(image.shape[0:2],np.uint8)
        if np.random.choice([BrightnessType.Bright,BrightnessType.Dark]) == ShadowType.Bright:
            brightness_image[:,:] = 255

        brightness_alpha = np.random.uniform(lower_range,upper_range)
        image[:,:,0] = cv2.addWeighted(image[:,:,0],1. - brightness_alpha,brightness_alpha,brightness_alpha,0)
        return image 
        
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
```

4. ##### Left and right Images

Each line in the csv file contains additonaly a left and right camera image. I used this files to additionaly increase my trainig size. 
For left images the stering angle is increased by 0.25 for right images the angle is decreased by the same factor. 

```python
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
```


5. ##### Random augmentation selection

Each image is augmentated by up to three augmentation strategies per image. Results are shown in the following gif file.

![Combined Augmentation][image40]


```python
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
```



### Model Architecture and Training Strategy

#### 1. Model architecture

In my attempts to find a model that can well performe on both tracks I tryed some models. Starting with the [Commaai Model](https://github.com/commaai/research) and the [NVIDIA Model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). After proper data augmentation and realising the diffrent color channel order in the drive.py (RGB vs BGR) I figured out that the used model architeture was not importend and both models were able to run both tracks successfuly. Finaly I used the Commaai mode.
 
 
| Layer (type)             |    Output Shape           |   Param  | Hint  |   
| :-------------: |:-------------:| :-----:| :-----:|
| Lambda |   | lambda x : (x/255.) -0.5 | Exception add model saving. Not used.  |
| cropping2d_1 (Cropping2D) |   (None, 65, 318, 3)     |   0     | (70,25) |   
| conv2d_1 (Conv2D)          |  (None, 17, 80, 16)      |  3088    |  |
| elu_1 (ELU)               |   (None, 17, 80, 16)      |  0       |  |  
| conv2d_2 (Conv2D)         |   (None, 9, 40, 32)     |    12832  |   |  
| elu_2 (ELU)              |    (None, 9, 40, 32)      |   0      |    | 
| conv2d_3 (Conv2D)         |   (None, 5, 20, 64)    |     51264   |   | 
| flatten_1 (Flatten)       |   (None, 6400)         |     0      |    | 
| dropout_1 (Dropout)      |    (None, 6400)         |     0     |    |  
| elu_3 (ELU)              |    (None, 6400)         |     0     |    |  
| dense_1 (Dense)          |    (None, 512)          |     3277312 |    | 
| dropout_2 (Dropout)     |     (None, 512)          |     0     |    |  
| elu_4 (ELU)             |     (None, 512)          |     0      |   |  
| dense_2 (Dense)         |     (None, 1)            |     513   |   |   


#### 2. Overfitting

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different image generator configurations. If the Train argument is passed the images will be agumented in the Validation mode not. That was used to ensure that the model was not overfitting.

```python
        train_generator = ImageDataGenerator(lines,ImageDataGeneratorMode.Train)
        validation_generator = ImageDataGenerator(lines,ImageDataGeneratorMode.Validation)
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with the default learning rate from 0.001.

### Training Strategy

#### 1. Training Data

##### 1 - First Track

I was able to run the first track without problems by using the provided data. 

##### 2 - Second Track

On the second track this was a disastrous fail. 
I recorded two laps in the simulator manuelly in both directions. 
The Car stated each time with a sharp turn and hit the barrier between the two roads. 
![barrier image][image20]

To overcome this behavior I placed the car in front of the barrier and performed a recorded sharp turn from the barrier away. After doing this multiple times the car was able to start on the track with out problems. With the two other places on the track where car was leaving the road I copied the process.  


The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
