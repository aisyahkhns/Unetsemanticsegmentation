## Semantic Segmentation for Self-Driving Cars using U-Net

### Project Overview
This project focuses on semantic segmentation for self-driving cars using a U-Net architecture implemented in TensorFlow. The goal is to accurately classify each pixel in an image, enabling the self-driving car to understand and interpret its surroundings. This dataset from cityscape

### U-Net Architecture
The U-Net model is a type of convolutional neural network (CNN) designed for fast and precise image segmentation. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. Here's an outline of the U-Net architecture used in this project:

```python
inputs = tf.keras.Input((width, height, channel))

# Contracting Path
conv1 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
conv1 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
pool1 = MaxPooling2D(2, 2)(conv1)

conv2 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
conv2 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
pool2 = MaxPooling2D(2, 2)(conv2)

conv3 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
conv3 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
pool3 = MaxPooling2D(2, 2)(conv3)

conv4 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
conv4 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
pool4 = MaxPooling2D(2, 2)(conv4)

conv5 = Conv2D(1024, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
conv5 = Conv2D(1024, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
dropout5 = Dropout(0.5)(conv5)

# Expanding Path
up6 = Conv2D(512, 2, activation='relu', kernel_initializer='he_normal', padding='same')(UpSampling2D(size=(2, 2))(dropout5))
merge6 = concatenate([conv4, up6], axis=3)
conv6 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge6)
conv6 = Conv2D(512, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

up7 = Conv2D(256, 2, activation='relu', kernel_initializer='he_normal', padding='same')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge7)
conv7 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv7)

up8 = Conv2D(128, 2, activation='relu', kernel_initializer='he_normal', padding='same')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge8)
conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

up9 = Conv2D(64, 2, activation='relu', kernel_initializer='he_normal', padding='same')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge9)
conv9 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv9)
conv9 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv9)

conv10 = Conv2D(12, 1, activation='softmax')(conv9)

# Model Compilation
model = tf.keras.Model(inputs, conv10)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
```

### Key Features
- **Contracting Path:** Consists of repeated application of convolutions (Conv2D) followed by a rectified linear unit (ReLU) and a max-pooling operation (MaxPooling2D) for down-sampling.
- **Bottleneck:** The deepest layer in the U-Net, with the highest number of filters, to capture the most abstract features of the input image.
- **Expanding Path:** Involves up-sampling the feature map and concatenating it with a high-resolution feature map from the contracting path, followed by convolutions (Conv2D).

### Technologies Used
- **TensorFlow**: For building and training the deep learning model.
- **Python**: As the primary programming language.
- **Cloud Platforms**: Azure and GCP for training and deployment.

### Performance
The model achieved a Mean Intersection over Union (Mean IoU) of 0.4049, indicating a moderate level of accuracy in segmenting different objects in the environment.

### Outcomes
This project aims to achieve high accuracy in segmenting different objects in the environment, contributing to the overall safety and efficiency of self-driving cars.
