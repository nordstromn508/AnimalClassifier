"""
This scrip is data loading for model training and testing purpose
load data using imageDataGenerator
NOTE: the data should be in format:
    data
     |_ train
        |_ class1
        |_ class2
        |_ ...
     |_ val
     |_ test
More details about getting this format in preprocessing script
"""
from keras.preprocessing.image import ImageDataGenerator

# Change the shape
DEFAULT_SHAPE = (180, 180, 3)

# Basic generator only do rescale
DEFAULT_GENERATOR = ImageDataGenerator(rescale=1./255)

# Modify this for data augmentation
DATA_AUGMENTATION_GENERATOR = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    rotation_range=10,
    vertical_flip=True,
    fill_mode='nearest'
)

# Can Add more Generator


"""
load training data and process it to given target shape
The training data will be shuffled
"""
def load_train_data(dataset, generator, target_shape):
    train = generator.flow_from_directory(
        dataset + '/train',
        target_size=target_shape,
        batch_size=128,
        color_mode="rgb",
        shuffle=True,
        class_mode="categorical",
        seed=42,
    )
    return train


"""
load validation data and process it to given target shape
The validation data will not be shuffled
"""
def load_val_data(dataset, generator, target_shape):
    val = generator.flow_from_directory(
        dataset + '/val',
        target_size=target_shape,
        batch_size=128,
        color_mode="rgb",
        shuffle=True,
        class_mode='categorical',
        seed=42
    )
    return val


"""
load testing data and process it to given target shape
The testing data will not be shuffled
The label will not be loaded for predicting
"""
def load_test_data(dataset, generator, target_shape):
    test = generator.flow_from_directory(
        dataset + '/test',
        target_size=target_shape,
        color_mode="rgb",
        shuffle=False,
        class_mode=None,
        seed=42
    )
    return test

