from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import load_model
from sklearn.metrics import confusion_matrix as conf_mat, ConfusionMatrixDisplay
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


def confusion_matrix(predictions, ground_truth, plot=True):
    """
    Creates a confusion matrix and prints it to the command line
    :param predictions: predictions from the model
    :param ground_truth: ground truth labels matching predictions in order
    :param plot: boolean value to enable plotting and visualization of confusion matrix
    :return: 0 if ran without error
    """
    cm = conf_mat(ground_truth, predictions)
    print("Confusion Matrix:\n {}".format(cm))

    if plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
    return 0


# plot_history('../history/Vgg16_history.csv')
