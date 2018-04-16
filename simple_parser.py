import cv2
import numpy as np


def get_raw_data(input_path, resize=True, resize_shape=(448,448,3)):
    found_bg = False
    all_imgs= {} # save all images

    classes_count_train = {} # count number classes train
    classes_count_test = {} # count number classes test
    class_mapping = {} # key of the class (number)

    visualise = True

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(',')
            #print(line_split)
            (filename, width, height, x1, y1, x2, y2, class_name, mode) = line_split # get data files
            width = int(width)
            height = int(height)
            if resize == True:
                x, y, w, h = compute_coordinate_reshape(int(x1), int(y1), int(x2), int(y2), (width, height, 3), resize_shape)
            else:
                x, y, w, h = compute_coordinate_reshape(int(x1), int(y1), int(x2), int(y2), (width, height, 3), (width, height, 3))
            if mode == 'training':
                if class_name not in classes_count_train: # count class in the dataset
                    classes_count_train[class_name] = 1
                else:
                    classes_count_train[class_name] += 1
            else:
                if class_name not in classes_count_test: # count class in the dataset
                    classes_count_test[class_name] = 1
                else:
                    classes_count_test[class_name] += 1
                    
                    
            if class_name not in class_mapping:
                if class_name == 'bg' and not found_bg: # if class name is background
                    print('Found class name with special name bg. Will be treated as a'
                          ' background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs: # if the file is not in the list currently, put in.
                all_imgs[filename] = {}

                #img = cv2.imread(filename)
                #(rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                #all_imgs[filename]['width'] = width
                #all_imgs[filename]['height'] = height
                all_imgs[filename]['bboxes'] = []
                if mode=='training':
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x': int(float(x)), 'y': int(float(y)), 'w': int(float(w)),
                 'h': int(float(h))})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch
        

        return all_data, classes_count_train, classes_count_test, class_mapping



def compute_coordinate_reshape(x1, y1, x2, y2, shape, resize_shape): # considering edge leftbottom and righttop
    x_coeff = 1.0*resize_shape[0]/shape[0]
    y_coeff = 1.0*resize_shape[1]/shape[1]
    x1new = int(x_coeff*x1)
    x2new = int(x_coeff*x2)
    y1new = int(y_coeff*y1)
    y2new = int(y_coeff*y2)
    
    x = int((x1new+x2new)/2)
    w = int((x2new-x1new)/2)
    y = int((y1new+y2new)/2)
    h = int((y2new-y1new)/2)
    
    if x-w<0:
        w = w - abs(x-w)
    if x+w>resize_shape[0]-1:
        w = w - abs(resize_shape[0]-1 -x-w)
    
    if y-h<0:
        h = h - abs(y-h)
    if y+h>resize_shape[1]-1:
        h = h - abs(resize_shape[1]-1 -y-h)
        
    return x, y, w, h

def test_data(all_data, resize=True, resize_shape=(448,448,3)):
    for data in all_data:
        img = cv2.imread(data['filepath'])
        if resize:
            img = cv2.resize(img, (resize_shape[0], resize_shape[1]))
        print('number of boxes :' , len(data['bboxes']))
        for i in range(len(data['bboxes'])):
            x = data['bboxes'][i]['x']
            y = data['bboxes'][i]['y']
            w = data['bboxes'][i]['w']
            h = data['bboxes'][i]['h']
            cv2.rectangle(img, (x-w, y-h), (x+w, y+h),(0,255,0),3)
            cv2.imshow('wind2', img)
            print(data['bboxes'][i]['class'])
        
        cv2.waitKey(0)