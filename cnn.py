from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

clf=Sequential()

clf.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu')) #role of this is to reduce the size of the feature set without lossing the important details

#It also removes noise and also acoounts for any geomatrical or positional invariants
clf.add(MaxPooling2D(pool_size=(2,2)))



# Adding a second convolutional layer To increase the accuracy
clf.add(Conv2D(32, (3, 3), activation = 'relu'))
clf.add(MaxPooling2D(pool_size = (2, 2)))


clf.add(Flatten())

clf.add(Dense(output_dim=128,activation='relu'))

clf.add(Dense(output_dim=1,activation='sigmoid'))

clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator


#4000 dogs and 4000 cats
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

#1000 cats and 1000 dogs
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

clf.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)



 
#Making single prediction
import numpy as np
from keras.preprocessing import image
#Load the image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
#Making it 3 dimensional i.e (64,64,3)
test_image = image.img_to_array(test_image)
#making it rank 4 tensor bc predict method expect 4d input i.e in a batch
#i.e this extra dimension is the batch 
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
