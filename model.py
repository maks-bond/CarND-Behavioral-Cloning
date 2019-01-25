import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn

lines = []
for i in range(13):
    with open('./data'+str(i+1)+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

            
np.random.shuffle(lines)

images = []
labels = []
im_shape = None

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    parentfoldername = source_path.split('/')[-3]
    # data1 images have different path prefix for images comparing to images in other folders
    prefix = './data1/IMG/'
    
    if 'data' in parentfoldername:
        prefix = './'+parentfoldername+'/IMG/'
        
    img_path = prefix+filename
    image = mpimg.imread(img_path)
    images.append(image)
    flipped_image = cv2.flip(image, 1)
    images.append(flipped_image)
    
    im_shape = image.shape
    label = float(line[3])
    
    labels.append(label)
    labels.append(-1.0*label)

X_train = np.array(images)
y_train = np.array(labels)

print("Got ", X_train.shape[0], "training images")
print("Image shape is: ", im_shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout, BatchNormalization

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 20), (0, 0))))
model.add(Conv2D(24, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(36, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv2D(48, kernel_size=5, strides=2, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
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
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, verbose=1)

model.save('model-4.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./loss.jpg')
