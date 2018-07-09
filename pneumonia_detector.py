#importing required libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Building the convolutional neural network architecture
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (128,128,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))


#Compiling the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#preprocessing images for the model
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./ 255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(128,128),
                                                 batch_size=32,
                                                 class_mode='binary')

validation_set = validation_datagen.flow_from_directory('dataset/val',
                                            target_size=(128,128),
                                            batch_size=32,
                                            class_mode='binary')
#training the network
from time import time
print("training the neural network ... \n\n")
t0 = time()

#if the model is already trained once, comment model.fit_generator(...) line below 
# and uncomment the model.load_weigths("weights.h5") line
 
model.fit_generator(training_set,
                         steps_per_epoch=5216,
                         epochs=25,
                         validation_data=validation_set,
                         workers = 12,
                         max_q_size = 100,
                         validation_steps=16)
print("\n\n training took ",round(time()-t0,3)/3600,'hrs')

#saving the trained hyper-parameters as weights.h5
model.save_weights('weights.h5')

#after running the above command the value of parameters will be saved in a file 
#names weights.h5 , in order to load the weights , uncommment the below line

#model.load_weights("weights.h5")

#testing the model on a test image -- the image is of a chest X ray with pneumonia, the model should predict pneumonia
from keras.preprocessing import image
import numpy as np

test_image = image.load_img('dataset/test_image.jpeg', target_size = (128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

prediction = model.predict(test_image)

if prediction[0][0] == 1:
    print("the person is having pneumonia\n")
else:
    print("the person does not have pneumonia\n")
    