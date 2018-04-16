# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pickle
from model import vgg16, mobilenet_yolo
from simple_parser import get_raw_data, test_data
import cv2
import tensorflow as tf
from keras import backend as K
from itertools import cycle
import math
import keras
from keras.activations import softmax, sigmoid
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope # mobilenet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
K.set_session(sess)
#K.set_learning_phase(1)
def keybyvalue(liste, value):
   for key in liste:
       if liste[key] == value:
           return key
       
class BatchGenerator():
    def __init__(self, X, Y, resize):
        self.x = X
        self.y = Y
        self.resize_shape = resize
    def imgs_from_paths(self, list_paths, path=''):
        l = []
        for p in list_paths:
            img = cv2.imread(path+p)
            if img == None:
                print("Error path")
                exit()
            l.append(cv2.resize(img, self.resize_shape))
        return np.asarray(l)
    
    def next_(self, batch_size=1, random=False, path=''):
        max_val = len(self.y)
        ite = cycle(list(range(max_val)))
        while True:
            if random:
                idx = np.random.randint(max_val, size=batch_size)
                listofpaths = [self.x[i] for i in idx]
                
                yield self.imgs_from_paths(listofpaths, path=path), self.y[idx]
            else:
                idx = []
                for cpt in range(batch_size):
                    idx.append(next(ite))
                idx = np.asarray(idx)
                #print(idx)
               
                listofpaths = [self.x[i] for i in idx]
                
                yield self.imgs_from_paths(listofpaths), self.y[idx]
                
        
class BoundBox:

    def __init__(self, x=0., y=0., w=0., h=0., classe=0):
        self.x, self.y, self.w, self.h = x, y ,w,  h
        self.xmin, self.ymin, self.xmax, self.ymax, self.c = x-w, y-h, x+w, y+h, classe



    def compute_iou(self, box):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.xmin, box.xmin)
        yA = max(self.ymin, box.ymin)
        xB = min(self.xmax, box.xmax)
        yB = min(self.ymax, box.ymax)

        if xA < xB and yA < yB:
            # compute the area of intersection rectangle
            interArea = (xB - xA) * (yB - yA)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (self.xmax - self.xmin) * (self.ymax - self.ymin)
            boxBArea = (box.xmax - box.xmin) * (box.ymax - box.ymin)
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the intersection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 0

        assert iou >= 0
        assert iou <= 1.01

        return iou

class YOLOv1():
    # parameters
    def __init__(self, input_shape=(448,448,3), model=None, grid_size=(7,7), bounding_boxes=2, number_classes=20):
        ## array (3, 2) => hauteur 3, largeur 2
        self.Sx = grid_size[1] # separate image in cell (per line/column)
        self.Sy = grid_size[0] # separate image in cell (per line/column)
        self.B = bounding_boxes # number of bounding boxes per cell
        self.C = number_classes # number of classes
        self.output_shape = (self.Sy, self.Sx, self.B*5+ self.C)
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.resize_shape = input_shape
        if model == 'vgg16':
            self.model = vgg16(input_shape=input_shape, grid_size=(self.Sx, self.Sy), bounding_boxes=self.B, num_class=self.C)
            self.model.summary()
        if model == 'mobilenet':
            self.model = mobilenet_yolo(input_shape=input_shape, grid_size=(self.Sx, self.Sy), bounding_boxes=self.B, num_class=self.C)
            self.model.summary()
        self.class_to_number = None
     
    def config_(self):
        print('### CONFIG ###')
        print('resize input : ', self.resize_shape)
        print('class : ', self.C)
        print('Bounding Boxes :', self.B)
        print('grid size : ', (self.Sy, self.Sx))
        print(' output shape :', self.output_shape)
        print(self.class_to_number)
        
    def save_config(self, filename='config_yolo.pkl'):
        yolo_save = YOLOv1(input_shape=self.resize_shape, model=None, grid_size=(self.Sy, self.Sx), bounding_boxes=self.B, number_classes=self.C)
        yolo_save.class_to_number = self.class_to_number
        pickle.dump(yolo_save, open(filename, 'wb'))
        
    def load_config(self, filename='config_yolo.pkl'):
        temp = pickle.load( open(filename, 'rb'))
        self.Sx = temp.Sx
        self.Sy = temp.Sy
        self.resize_shape = temp.resize_shape
        self.output_shape = temp.output_shape
        self.C = temp.C
        self.B = temp.B
        self.class_to_number = temp.class_to_number
        
    # loading and preprocess data before training step 
    def load_data(self, filename):
        datas, classes_count_train, classes_count_test, class_mapping = get_raw_data(filename, resize=True, resize_shape=self.resize_shape)
        self.class_to_number = class_mapping
        print(class_mapping)
        print(len(datas))
        X_train, Yboxes_train, X_test, Yboxes_test = self.separate_train_test(datas, dic=self.class_to_number)

        return X_train, Yboxes_train, X_test, Yboxes_test, classes_count_train, classes_count_test, class_mapping
    
    def separate_train_test(self, all_data, dic=None):
        X_train = []
        Bboxes_train = []
        
        X_test = []
        Bboxes_test = []
        
        for i, file in enumerate(all_data):
            if file['imageset'] == 'trainval':
                #train.append({})
                #train[-1]['filepath'] = file['filepath']
                #train[-1]['bboxes'] = self.convert_list_to_bbox(file['bboxes'], dic)
                X_train.append(file['filepath'])
                Bboxes_train.append(self.convert_list_to_np(file['bboxes'], dic))
            else:
                #test.append({})
                X_test.append(file['filepath'])
                Bboxes_test.append(self.convert_list_to_np(file['bboxes'], dic))

        return X_train, np.asarray(Bboxes_train), X_test, np.asarray(Bboxes_test)

    def convert_list_to_bbox(self, liste, dic=None):
        bbox = []
        for l in liste:
            if dic is None:
                box = BoundBox(x=l['x']/(self.resize_shape[0]-1), y=l['y']/(self.resize_shape[1]-1),
                               w=l['w']/(self.resize_shape[0]-1), h=l['h']/(self.resize_shape[1]-1), 
                               classe=l['class']) # normalise coordinate between 0 and 1
            else:
                box = BoundBox(x=l['x']/(self.resize_shape[0]-1), y=l['y']/(self.resize_shape[1]-1),
                               w=l['w']/(self.resize_shape[0]-1), h=l['h']/(self.resize_shape[1]-1), 
                               classe=dic[l['class']])
            bbox.append(box)
        return bbox
    
    def convert_list_to_np(self, liste, dic): #height width
        '''
            convert list of box into numpy array
            return the ground truth box, the class object and the box coordinate(normalise between 0 and 1)
        '''
        y_bbox = np.zeros((self.Sy, self.Sx, self.B, 4))
        y_confidence = np.zeros((self.Sy, self.Sx, self.B), dtype=np.uint8)
        y_class = np.zeros((self.Sy, self.Sx, self.C), dtype=np.uint8)
        
        cpt_confidence = np.zeros((self.Sy, self.Sx), dtype=np.uint8)
        
        height = self.resize_shape[0]
        width = self.resize_shape[1]
        height_cell = math.ceil(height/self.Sy)
        width_cell = math.ceil(width/self.Sx)
        
        for l in liste:
            X = l['x']
            Y = l['y']
            W = l['w']
            H = l['h']
            Class = dic[l['class']]
            #print(X, width_cell, X//width_cell)
            # compute ground truth box + class + bbox
            # multi box, if there is two object of the same class in one cell, then we return this two. In other case return one object.
            cpt = cpt_confidence[Y//height_cell, X//width_cell] 
            if cpt == 0:
                y_confidence[Y//height_cell, X//width_cell, :] = 1
                cpt_confidence[Y//height_cell, X//width_cell] += 1
                y_class[Y//height_cell, X//width_cell, Class] = 1
                y_bbox[Y//height_cell, X//width_cell, :, 0] = float(X/width)
                y_bbox[Y//height_cell, X//width_cell, :, 1] = float(Y/height)
                y_bbox[Y//height_cell, X//width_cell, :, 2] = float(W/width)
                y_bbox[Y//height_cell, X//width_cell, :, 3] = float(H/height)
            else:
                if y_class[Y//height_cell, X//width_cell, Class] == 1:
                    if cpt < self.B:
                        y_confidence[Y//height_cell, X//width_cell, cpt] = 1
                        y_bbox[Y//height_cell, X//width_cell, cpt, 0] = float(X/width)
                        y_bbox[Y//height_cell, X//width_cell, cpt, 1] = float(Y/height)
                        y_bbox[Y//height_cell, X//width_cell, cpt, 2] = float(W/width)
                        y_bbox[Y//height_cell, X//width_cell, cpt, 3] = float(H/height)
                        cpt_confidence[Y//height_cell, X//width_cell] += 1
        return np.concatenate((y_bbox.flatten(), y_confidence.flatten(), y_class.flatten()), axis=0)
    
    # training steps
    def train(self, filename, path=''):
        X_train, Bboxes_train, X_test, Bboxes_test, classes_count_train, classes_count_test, _ = self.load_data(filename)
        print(len(X_train), Bboxes_train.shape)
        
        if self.C != len(self.class_to_number):
            print('number of class wrong check it ! ')
            return 0
        
        self.model.compile(optimizer='adam', loss=self.loss_yolo)
        batch_size=1
        self.save_config()
        batchTrain = BatchGenerator(X_train, Bboxes_train, resize=(self.resize_shape[0],self.resize_shape[1]) ).next_(batch_size=batch_size, random=True, path=path)
        batchTest = BatchGenerator(X_test, Bboxes_test, resize=(self.resize_shape[0],self.resize_shape[1]) ).next_(batch_size=batch_size, random=False, path=path)
        #x, y = next(batchTrain)
        
        # callback
        callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                                           batch_size=32, write_graph=True, write_grads=False, 
                                                           write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                                                           embeddings_metadata=None)
        output = 'YOLOv1'#-{epoch:02d}
        checkpoints = ModelCheckpoint(output+'.hdf5', verbose=1, save_best_only=True, period=1)
        callbacks_list = [callback_tensorboard, checkpoints]
        #print(x, y)
        #self.print_img(x, y, False)
        step_train = len(X_train)//batch_size
        step_test = len(X_test)//batch_size
        epochs=50
        self.model.fit_generator(batchTrain,
          steps_per_epoch=step_train,
          epochs=epochs,
          verbose=1,
          validation_data=batchTest,
          validation_steps=step_test,
          callbacks=callbacks_list)
#        print('train batch')
#        print(self.model.train_on_batch(x,y))
#        print(self.convert_output(self.model.predict(x)))
#        print('train batch')
#        print(self.model.train_on_batch(x,y))
        return 0
    # label = 7x7(5+20) = 14XX
        
    def predict_on_img(self, filename, treshold=0.5):
        img = cv2.resize(cv2.imread(filename), (self.resize_shape[0], self.resize_shape[1]))
        img_ = np.expand_dims(img, axis=0)
        print("\n\n ### IMG SHAPE #### : ", img_.shape)
        cv2.imshow('img', img)
        
        x = self.model.predict(img_)
        print(x.shape)        
        cv2.waitKey(0)
        bbox, confidence, classes = self.convert_output(x)
        
        for i in range(self.Sy):
            for j in range(self.Sx):
                for k in range(self.B):
                    prob = confidence[0,i,j,k]* np.max(classes[0, i,j,k])
                    print(prob)
                    if prob>treshold:
                        print('Object :', keybyvalue(self.class_to_number, np.argmax(classes[i,j,k]) ))
                        x = int(bbox[0, i, j, k, 0]*(self.resize_shape[1]))
                        y = int(bbox[0, i, j, k, 1]*(self.resize_shape[0]))
                        w = int(bbox[0, i, j, k, 2]*(self.resize_shape[1]))
                        h = int(bbox[0, i, j, k, 3]*(self.resize_shape[0])) 
                        cv2.circle(img, (x,y), 10, (0,255,0), -1  )
                        cv2.rectangle(img, (x-w, y-h), (x+w, y+h),(0,255,0),3)
                        cv2.imshow('img', img)
                        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def load_network(self, path):
        

        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
            self.model = load_model('YOLOv1.hdf5')
        
    def ground_truth_box_and_iou(self, prediction_boxes, datas):
        '''
            FOR ONLY ONE IMAGE
            P_object  : probability the cell has an object or not
            input : data (list of boxes) regarding object coordinate for an image 
            return : an array of confidence (0 : false, 1: True) depending of the grid size. class and iou
        '''
        gtb = np.zeros((self.Sx, self.Sy))
        iou = np.zeros((self.B, self.Sx, self.Sy))
        y_class = np.zeros((self.Sx, self.Sy, self.C))
        separate_predictor = round(self.C/self.B) # so as to determine which predictor is associated to which class object
        #y_class[:,:, self.C-1] = 1 #class background
        
        
        for box in datas: # box = one box in the image of the dataset 'truth'

            X = round(box.x*self.resize_shape[0])
            Y = round(box.y*self.resize_shape[1])
            gtb[X%self.Sx, Y%self.Sy] = 1
            #y_class[X%self.Sx, Y%self.Sy, self.C-1] = 0
            y_class[X%self.Sx, Y%self.Sy, box.c] = 1
            
            # each predictor is associated to a class object
            i = int(box.c/separate_predictor)
            box_temp = BoundBox(x=prediction_boxes[i, 0, X%self.Sx, Y%self.Sy], y=prediction_boxes[i, 1, X%self.Sx, Y%self.Sy], 
                                    w=prediction_boxes[i, 2, X%self.Sx, Y%self.Sy], h=prediction_boxes[i, 3, X%self.Sx, Y%self.Sy])
            iou[i, X%self.Sx, Y%self.Sy] = box.compute_iou(box_temp)
                
        return gtb, iou, y_class
    
    def compute_confidence(self, prediction, datas):
        '''
            compute confidence on 1 image
            datas : list of boxes in dataset
            prediction : array (Sx, Sy, 4, B) [box, confidence, class]
            confidence = Pobject * IOU pred/groundtruth, array (B, Sx, Sy)
        '''
        
        gtb_img, iou_img, y_class = self.ground_truth_box_and_iou(prediction, datas)
        return gtb_img*iou_img, y_class
    
    def compute_error(self, prediction, datas):
        '''
            compute for one image
        '''
        pred_bboxes = prediction[:self.Sx*self.Sy*4*self.B].reshape(self.B, 4, self.Sx, self.Sy)
        pred_confidence = prediction[self.Sx*self.Sy*4:self.Sx*self.Sy*5]
        pred_class = prediction[self.Sx*self.Sy*5:]
        
        confidence, y_class = self.compute_confidence(pred_bboxes ,datas)

        
    def loss_yolo(self, y_true, y_pred):  
        '''
            dense layer : Sx * Sy * B * ((5) + C) 
            bbox : Sx * Sy * B * 4
            confidence = Sx * Sy * B * 1
            class : Sx * Sy * B * C
            
            7*7*4 = 196
            7*7*1 = 49
            7*7*20 = 980
        '''
        # reshape into cell
        y_true_bbox = sigmoid(K.reshape(y_true[:, :self.Sx*self.Sy*4*self.B], (-1, self.Sy, self.Sx, self.B, 4)))
        y_pred_bbox = sigmoid(K.reshape(y_pred[:, :self.Sx*self.Sy*4*self.B], (-1, self.Sy, self.Sx, self.B, 4)))
        y_true_confidence = sigmoid(K.reshape(y_true[:, self.Sx*self.Sy*4*self.B:self.Sx*self.Sy*5*self.B], (-1, self.Sy, self.Sx, self.B)))
        y_pred_confidence = sigmoid(K.reshape(y_pred[:, self.Sx*self.Sy*4*self.B:self.Sx*self.Sy*5*self.B], (-1, self.Sy, self.Sx, self.B)))
        y_true_class = softmax(K.reshape(y_true[:, self.Sx*self.Sy*5*self.B:], (-1, self.Sy, self.Sx, self.C)), axis=3)
        y_pred_class = softmax(K.reshape(y_pred[:, self.Sx*self.Sy*5*self.B:], (-1, self.Sy, self.Sx, self.C)), axis=3)
        
        # keep only boxes which exist in the dataset, if not put 0
        y_pred_bbox = y_pred_bbox * K.cast((y_true_bbox > 0), dtype='float32')
        
        # compute loss bbox
        loss_bbox = K.reshape(K.square( y_true_bbox[:,:,:,:,0:2] - y_pred_bbox[:,:,:,:,0:2]), (-1, self.Sx*self.Sy*2*self.B)) + K.reshape(K.square( K.sqrt(y_true_bbox[:, :, :, :, 2:]) - K.sqrt(y_pred_bbox[:, :, :, :, 2:])), (-1, self.Sx*self.Sy*2*self.B))
        loss_bbox = K.sum(loss_bbox, axis=1)*self.lambda_coord
        
        # compute loss confidence
        xmin_true = y_true_bbox[:,:,:,:, 0] - y_true_bbox[:,:,:,:, 2]
        ymin_true = y_true_bbox[:,:,:,:, 1] - y_true_bbox[:,:,:,:, 3]
        xmax_true = y_true_bbox[:,:,:,:, 0] + y_true_bbox[:,:,:,:, 2]
        ymax_true = y_true_bbox[:,:,:,:, 1] + y_true_bbox[:,:,:,:, 3]
        
        xmin_pred = y_pred_bbox[:,:,:,:, 0] - y_pred_bbox[:,:,:,:, 2]
        ymin_pred = y_pred_bbox[:,:,:,:, 1] - y_pred_bbox[:,:,:,:, 3]
        xmax_pred = y_pred_bbox[:,:,:,:, 0] + y_pred_bbox[:,:,:,:, 2]
        ymax_pred = y_pred_bbox[:,:,:,:, 1] + y_pred_bbox[:,:,:,:, 3]
        
        #print(' Xmin true : ', K.int_shape(xmin_true))
        
        xA = K.maximum(xmin_true, xmin_pred)
        yA = K.maximum(ymin_true, ymin_pred)
        xB = K.minimum(xmax_true, xmax_pred)
        yB = K.minimum(ymax_true, ymax_pred)
        #print('Xa : ', K.int_shape(xA))
        #if xA < xB and yA < yB:
        #condition1 = K.cast((xA<xB), dtype='float32')
        #condition2 =  K.cast( (yA<yB), dtype='float32')
        #condition = condition1 + condition2
        condition = K.cast((xA<xB), dtype='float32') + K.cast( (yA<yB), dtype='float32')
        # find which iou to compute
        tocompute = K.cast( K.equal(condition, 2.0), dtype='float32')
        del condition
            # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA) * tocompute
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred) 
        boxBArea = (xmax_true - xmin_true) * (ymax_true - ymin_true)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        eps=0.0000001
        iou = (interArea / (eps+boxAArea + boxBArea - interArea)) * y_true_confidence * y_pred_confidence
        #print('iou shape : ', K.int_shape(iou))
        #print('tocompute shape : ', K.int_shape(tocompute))
        conf_obj = iou - y_pred_confidence*y_true_confidence
        conf_nobj = y_pred_confidence * K.cast( (y_true_confidence<1.0), dtype='float32')
        loss_confidence = K.reshape( K.square(conf_obj), (-1, self.Sy*self.Sx*self.B)) + self.lambda_noobj*K.reshape( K.square(conf_nobj), (-1, self.Sy*self.Sx*self.B))
        loss_confidence = K.sum(loss_confidence, axis=1)
        #print('loss confidence shape :', K.int_shape(loss_confidence))
            
            
        # keep only prediction class if there is an object in the cell, else put class to 0
        y_pred_class = (K.reshape(y_true_confidence[:,:,:,0], (-1, self.Sy, self.Sx, 1)) * y_pred_class)
        
        # compute loss class
        loss_class = K.sum(K.square( y_pred_class  - y_true_class), axis=3)
        loss_class = K.sum(loss_class, axis=2)
        loss_class = K.sum(loss_class, axis=1)
        #print(K.int_shape(loss_bbox))
        #print(K.int_shape(loss_class))
        
        loss = K.mean(loss_bbox) + K.mean(loss_confidence) + K.mean(loss_class)
        #loss = K.mean(loss_confidence) + K.mean(loss_class) #K.mean(loss_bbox) #K.mean(loss_confidence) #K.mean(self.lambda_noobj * loss_confidence)
        return loss
        
    def convert_output(self, output):
        print(output.shape, self.Sx, self.Sy, 4*self.B)
        y_pred_bbox = sigmoid(K.reshape(output[:, :self.Sx*self.Sy*4*self.B], (-1, self.Sy, self.Sx, self.B, 4)))
        y_pred_confidence = sigmoid(K.reshape(output[:, self.Sx*self.Sy*4*self.B:self.Sx*self.Sy*5*self.B], (-1, self.Sy, self.Sx, self.B)))

        y_pred_class = softmax(K.reshape(output[:, self.Sx*self.Sy*5*self.B:], (-1, self.Sy, self.Sx, self.C)), axis=3)        

        #x = K.eval(y_pred_class[0,0,0,:])
        #print(x)
        #print(np.sum(x))
        return K.eval(y_pred_bbox), K.eval(y_pred_confidence), K.eval(y_pred_class)
        
    def print_img(self, all_data_path, all_data_boxes, load_img=True):
        for i, data_ in enumerate(all_data_path):
            #print(data_)
            if load_img:
                img = cv2.imread(data_)
                img = cv2.resize(img, (self.resize_shape[0], self.resize_shape[1]))
            else:
                img = data_
            print('number of boxes :' , len(all_data_boxes[i]))
            
            print(all_data_boxes[i].shape)
            y_bboxes = all_data_boxes[i, :self.Sx*self.Sy*self.B*4].reshape((self.Sy, self.Sx, self.B, 4))
            y_confidence = all_data_boxes[i, self.Sx*self.Sy*self.B*4:self.Sx*self.Sy*self.B*5].reshape((self.Sy, self.Sx, self.B))
            y_class = all_data_boxes[i, self.Sx*self.Sy*self.B*5:].reshape((self.Sy, self.Sx, self.C))
            print(y_bboxes.shape, y_confidence.shape, y_class.shape)
            
            
            
            res = np.where(y_confidence>0.5)
            print(res)
            res= np.array([res[0], res[1], res[2]]).T
            print(res)
            for i_, j_, cpt in res:
                print(i_, j_, cpt)
                x = int(y_bboxes[i_, j_, cpt, 0]*(self.resize_shape[1]))
                y = int(y_bboxes[i_, j_, cpt, 1]*(self.resize_shape[0]))
                w = int(y_bboxes[i_, j_, cpt, 2]*(self.resize_shape[1]))
                h = int(y_bboxes[i_, j_, cpt, 3]*(self.resize_shape[0])) 
                cv2.circle(img, (x,y), 10, (0,255,0), -1  )
                cv2.rectangle(img, (x-w, y-h), (x+w, y+h),(0,255,0),3)
                cv2.imshow('wind2', img)
                label = np.where(y_class[i_,j_]==1)[0]
                print(keybyvalue(self.class_to_number, label))
                cv2.waitKey(0)
                
            cv2.destroyAllWindows()

 
         
keras.losses.loss_yolo = YOLOv1().loss_yolo  
filename = 'VOC2007.txt'
yolo = YOLOv1(input_shape=(448,448,3), model=None, grid_size=(7,7), bounding_boxes=2, number_classes=20)
yolo.load_network('YOLOv1.hdf5')
yolo.train(filename, path='../../')

#yolo.load_data(filename)
#yolo.save_config()
#exit()

#exit()
#yolo.load_model('YOLOv1.hdf5')
#yolo.load_config()

#yolo.model.summary()
#yolo.predict_on_img('../../dataset/VOCdevkit/VOC2007train/JPEGImages/000012.jpg', treshold=0.5)