# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:10:36 2018

@author: shiro
"""

import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, GlobalMaxPooling2D, Dropout
from keras import backend as K
from keras.applications.mobilenet import MobileNet

def copy_weights(oldmodel, newmodel):
    dic_w = {}
    for layer in oldmodel.layers:
        dic_w[layer.name] = layer.get_weights()
    
    for i, layer in enumerate(newmodel.layers):
        if layer.name in dic_w and layer.name != 'dense_output' and layer.name != 'input':
            #print(newmodel.layers[i].get_weights()[0].shape)
            #print(newmodel.layers[i].get_weights()[0][:,:,0,0])
            newmodel.layers[i].set_weights(dic_w[layer.name])
            #print(layer.name)
            #print(newmodel.layers[i].get_weights()[0][:,:,0,0])
    return newmodel


def relu_clip(max_value=1.):        
    def relu_custom(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_custom

def mobilenet_yolo(input_shape, grid_size=7, num_class=20, bounding_boxes=2, dropout=0.1):
    #img_input = Input(shape=input_shape)
    
    model = MobileNet(input_shape=input_shape, include_top=False, weights=None, pooling='avg')
    model = copy_weights(MobileNet(input_shape=(224,224,3), include_top=True, weights='imagenet'), model)
    
    img_input = model.layers[0].input
    x = model.layers[-1].output
    model = output_layers(img_input, x, grid_size, num_class, bounding_boxes, dropout=dropout)
    return model
def vgg16(input_shape, grid_size=7, num_class=20, bounding_boxes=2, dropout=0.1):
    img_input, conv_layers = vgg16_conv(input_shape)
    model = output_layers(img_input, conv_layers, grid_size, num_class, bounding_boxes, dropout=dropout)
    return model

def vgg16_conv(input_shape=(448,448,3)):
    
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#    # Block 2
#    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
#    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
#
#    # Block 3
#    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
#    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
#    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#
#    # Block 4
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#
#    # Block 5
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
#    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = GlobalMaxPooling2D()(x)
    #model = Model(img_input, x)
    return img_input, x

def output_layers(input_layer, conv_layer, grid_size=(7,7), num_class=20, bounding_boxes=2, dropout = 0.5):
    x = Dropout(dropout, name='dropout')(conv_layer)
    output_boxes = Dense(grid_size[0]*grid_size[1]*(bounding_boxes*5+ num_class), name='dense_output')(x)
    #output_confidence = Dense(grid_size*grid_size*bounding_boxes, activation='tanh')(conv_layer)
    return Model(inputs=input_layer, outputs=output_boxes, name='YOLOv1')
