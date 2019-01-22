import csv
import cv2
import numpy as np

lines = []
with open('./data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
labels = []
im_shape = None

print("Got ", len(lines), "training images")

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    img_path = './data1/IMG/'+filename
    image = cv2.imread(img_path)
    images.append(image)
    im_shape = image.shape
    label = float(line[3])
    labels.append(label)

X_train = np.array(images)
y_train = np.array(labels)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=im_shape))
model.add(Conv2D(100, 5, 2, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(150, 5, 2, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(200, 5, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(200))
# Activation?
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')
