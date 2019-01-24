import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import sklearn

lines = []
for i in range(13):
    with open('./data'+str(i+1)+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

            
np.random.shuffle(lines)
# from sklearn.model_selection import train_test_split
# train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1: # Loop forever so the generator never terminates
#         np.random.shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]

#             images = []
#             labels = []
#             for batch_sample in batch_samples:
#                 source_path = batch_sample[0]
#                 filename = source_path.split('/')[-1]
#                 parentfoldername = source_path.split('/')[-3]
#                 prefix = './data1/IMG/'

#                 if 'data' in parentfoldername:
#                     prefix = './'+parentfoldername+'/IMG/'

#                 img_path = prefix+filename
#                 image = mpimg.imread(img_path)
#                 label = float(batch_sample[3])
#                 images.append(image)
#                 flipped_image = cv2.flip(image, 1)
#                 images.append(flipped_image)

#                 labels.append(label)
#                 labels.append(-1.0*label)

#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(labels)
#             yield sklearn.utils.shuffle(X_train, y_train)

# # compile and train the model using the generator function
# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)

images = []
labels = []
im_shape = None

for line in lines:
    #for i in range(3):
    if True:
        i = 0
        source_path = line[i]
        filename = source_path.split('/')[-1]
        parentfoldername = source_path.split('/')[-3]
        prefix = './data1/IMG/'
        
        if 'data' in parentfoldername:
            prefix = './'+parentfoldername+'/IMG/'
            
        img_path = prefix+filename
        #image = cv2.imread(img_path)
        image = mpimg.imread(img_path)
        images.append(image)
        flipped_image = cv2.flip(image, 1)
        images.append(flipped_image)

        im_shape = image.shape
        label = float(line[3])
        
        offset_angle = 0.2
        if i == 1:
            # left image
            label += offset_angle
        if i == 2:
            # right image
            label -= offset_angle
        
        labels.append(label)
        labels.append(-1.0*label)

X_train = np.array(images)
y_train = np.array(labels)

print("Got ", X_train.shape[0], "training images")
print("Image shape is: ", im_shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Conv2D(24, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(36, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(48, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(Dropout(0.2))
#model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=3, strides=2, padding='valid', activation='relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=3, strides=2, padding='valid', activation='relu'))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(100))
# Activation?
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model-2.h5')
