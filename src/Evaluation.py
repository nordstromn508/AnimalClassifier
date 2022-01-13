from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

image_size = []
data_path = ''


def predict(model, plot=True):
    # need to complete this
    return 0 # return predictions and ground truth


'''
Plots the learning curve
'''
def plot_history(path):
    df = pd.read_csv(path)
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()


'''
Display a nice confusion matrix
'''
def confusion_matrix(predictions, ground_truth):
    # need to complete this
    return 0