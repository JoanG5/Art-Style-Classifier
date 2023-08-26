import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import os

class_names = ['Cubism', 'Pop_Art','Surrealism']

test_images = []
test_labels = []

for class_name in class_names:
    class_dir = os.path.join('test_data', class_name)
    class_label = class_names.index(class_name)

    for filename in os.listdir(class_dir):
        img_path = os.path.join(class_dir, filename)
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (32, 32)) / 255.0
        test_images.append(img)
        test_labels.append(class_label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)


# USED TO TRAIN THE MODEL 

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(test_images, test_labels, epochs=10, validation_data=(test_images, test_labels))

# loss, accuracy = model.evaluate(test_images, test_labels)

# model.save('image_classifier.model')


model = models.load_model('image_classifier.model')

for i in range(3):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_directory, f'art{i}.jpg')
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32)) / 255.0

    plt.imshow(img, cmap=plt.cm.binary)

    prediction = model.predict(np.array([img]))
    index = np.argmax(prediction)

    print(class_names[index])