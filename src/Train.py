"""
Model training script
"""
import os
from glob import glob
import numpy as np
from src import DataLoader, Models

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

'''
Set path and model parameters 
'''
dataset = '../data'
train_path = dataset + '/train'
test_path = dataset + '/val'

BATCH = 128 # set this
EPOCHS = 300 # set this
image_size = (180, 180) # set this

# ignore tran val steps for now
image_files = glob(train_path + '/*/*.jpg')
test_image_files = glob(test_path + '/*/*.jpg')
train_step = int(np.ceil(len(image_files)/BATCH))
val_step = int(np.ceil(len(test_image_files)/BATCH))

generator = DataLoader.DEFAULT_GENERATOR # for no data augmentation
adv_generator = DataLoader.DATA_AUGMENTATION_GENERATOR # for data augmentation

# load data
train_data = DataLoader.load_train_data(dataset, generator, image_size)
val_data = DataLoader.load_val_data(dataset, generator, image_size)

'''
Train models
'''
vgg16 = Models.vgg16((180, 180, 3))
Models.train_save(vgg16, 'Vgg16',
                  train_data, val_data, EPOCHS, BATCH)

# vgg16_tf = Models.vgg_pretrained((180, 180, 3))
# Models.train_save(vgg16_tf, 'Vgg16_transfer_learning',
#                   train_data, val_data, EPOCHS, BATCH)
#                   # int(np.ceil(4756/BATCH)), int(np.ceil(300/BATCH)))
#
#
# vgg16_tf_fineTuning = Models.vgg_pretrained(image_size, True)
# Models.train_save(vgg16_tf_fineTuning, 'Vgg16_transfer_learning_fineTuning',
#                   train_data, val_data, EPOCHS, BATCH)
#                   # int(np.ceil(4756/BATCH)), int(np.ceil(300/BATCH)))