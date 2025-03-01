import numpy as np
# from keras_preprocessing.image import ImageDataGenerator
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# from keras.models import models
from glob import glob
import os
import argparse
from get_data import get_data, read_params
# import matplotlib.pyplot as plt
# from keras.applications.vgg16 import VGG16
# import tensorflow as tf

def train_model(config_file):
    config = get_data(config_file)
    train = config['train']['trainable']
    if train == True:
        img_size = config['train']['image_size']
        train_set = config['train']['train_path']
        test_set = config['train']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        horizontal_flip = config['img_augment']['horizontal_flip']
        vertical_flip = config['img_augment']['vertical_flip']
        class_mode = config['img_augment']['class_mode']
        batch = config['img_augment']['batch_size']
        loss = config['train']['loss']
        optimizer = config['train']['optimizer']
        metrics = config['train']['metrics']
        epochs = config['train']['epochs']
        model_path = config['train']['sav_dir']
        
        print(train_set)
        print(test_set)
        print(img_size)
        print(num_cls)
        print(rescale)
        print(shear_range)
        print(zoom_range)






if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    train_model(config_file=passed_args.config)