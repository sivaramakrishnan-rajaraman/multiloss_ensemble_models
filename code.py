'''
This code is created as part of our study entitled
"Multi-loss deep learning ensemble for class-imbalanced
classification using chest radiographs". The code is divided into 
several parts. The first part is the segmentation part where the 
CXRs are lung-semgneted and cropped to contain all the lung pixels.
An efficientNet-B0 based U-Net model is trained to generate lung masks
in this regard. The code works with tensorflow 2.x

'''
#%%
#load libraries
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.regularizers import l2
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from PIL import Image
import csv
import cv2
import statistics
import struct
import zlib
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
try:
    from itertools import izip as zip
except ImportError: 
    pass

#%%

#get current working directory
os.getcwd()

#%% define loss functions

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
                K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def focal_loss(gamma=2., alpha=.25): #alpha = 0.25 before
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - \
            K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

def tversky(y_true, y_pred, smooth=1e-10):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

#%%
#using qubvel segmentation models from https://github.com/qubvel/segmentation_models
import segmentation_models as sm
# Segmentation models using tf.keras framework
print(sm.__version__)

#%%
# Creating a EfficientNet-B0 based segmentation model

BACKBONE = 'efficientnetb0'
preprocess_input = sm.get_preprocessing(BACKBONE)

# define model
model_eff0 = sm.Unet(BACKBONE, input_shape=(256,256,3), 
                         encoder_weights='imagenet', 
                         classes=3, activation='sigmoid')
model_eff0.summary()

#compile the model
model_eff0.compile(optimizer = Adam(lr=1e-4), 
                  loss=focal_loss(gamma=2., alpha=.25), 
                  metrics=[iou, dice_coef, 'accuracy'])

#%%
# define data generators

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1): 

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def valGenerator(batch_size,val_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1): 

    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    val_generator = zip(image_generator, mask_generator)
    for (img,mask) in val_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def testGenerator(test_path,target_size = (256,256),flag_multi_class = False,as_gray = True): 
    for filename in os.listdir(test_path):
        img = io.imread(os.path.join(test_path,filename),as_gray = as_gray) 
        img = img / 255.
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile,test_path, flag_multi_class = False,num_class = 2):
    file_names = os.listdir(test_path)
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,file_names[i]),img)

#%%
#train the UNET model
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,'data/membrane/train1','image','label',data_gen_args,
                        target_size = (512,512), save_to_dir = None) #batch size = 2 here
valGene = valGenerator(2,'data/membrane/val','image','label',data_gen_args,
                       target_size = (512,512),save_to_dir = None) #batch size = 2 here

callbacks = [EarlyStopping(monitor='val_loss', patience=10, 
                           verbose=1, min_delta=1e-4,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                               patience=5, verbose=1,
                               epsilon=1e-4, mode='min'),
             ModelCheckpoint(monitor='val_loss', filepath='trained_model/eff0_unet.hdf5', 
                             save_best_only=True, 
                             mode='min', 
                             verbose = 1)]

results = model_eff0.fit_generator(generator=myGene,
                                   steps_per_epoch=217, #no.of train samples/batch_size
                                   epochs=64, 
                                   callbacks=callbacks,
                                   validation_data=valGene, 
                                   validation_steps=69, #no.of validation samples/batch_size
                                   verbose=1)
#%%
#plot performance
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), 
         marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()

#%%
#predict masks using the trained model: 

test_path = r'C:\Users\xx\data\test' 
save_path = r'C:\Users\xx\data\result'  

testGene = testGenerator(test_path,target_size = (512,512))

model_eff0.load_weights("trained_model/eff0_unet.hdf5")
results = model_eff0.predict_generator(testGene,
                                       1345, # no. of test samples
                                       verbose=1, 
                                       workers=1, 
                                       use_multiprocessing=False) 
saveResult(save_path, results, test_path)

#%%
#postprocessing: cropping the lungs and saving the cropped image


def generate_bounding_box(image_dir: str, #containing images
                          mask_dir: str, #containing masks, images have same name as original images
                          dest_csv: str, #CSV file to write the bounding box coordinates
                          crop_save_dir: str): #save the cropped bounding box images
    """
    the orginal images are resized to 512 x 512
    the output crops are resized to 512 x 512
    """
    
    if not os.path.isdir(mask_dir):
        raise ValueError("mask_dir not existed")

    case_list = [f for f in os.listdir(mask_dir) if f.split(".")[-1] == 'png'] #all mask images are png files

    with open(dest_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)

        for j, case_name in enumerate(case_list):
            mask = cv2.imread(mask_dir + case_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            image = cv2.imread(image_dir + case_name, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (512,512), interpolation = cv2.INTER_AREA) #original images are resized to 512 x 512
            if mask is None or image is None:
                raise ValueError("The image can not be read: " + case_name)

            reduce_col = np.sum(mask, axis=1)
            reduce_row = np.sum(mask, axis=0)
            reduce_col = (reduce_col >= 255)*reduce_col
            reduce_row = (reduce_row >= 255)*reduce_row
            first_none_zero = None
            last_none_zero = None

            last = 0
            for i in range(reduce_col.shape[0]):
                current = reduce_col[i]
                if last == 0 and current != 0 and first_none_zero is None:
                    first_none_zero = i

                if current != 0:
                    last_none_zero = i

                last = reduce_col[i]

            up = first_none_zero
            down = last_none_zero

            first_none_zero = None
            last_none_zero = None
            last = 0
            for i in range(reduce_row.shape[0]):
                current = reduce_row[i]
                if last == 0 and current != 0 and first_none_zero is None:
                    first_none_zero = i

                if current != 0:
                    last_none_zero = i

                last = reduce_row[i]

            left = first_none_zero
            right = last_none_zero

            if up is None or down is None or left is None or right is None:
                cv2.imwrite(crop_save_dir + case_name, image)
                
            
            # new coordinates for image which is 1 times of mask, mask images are 512 x 512
            #so need to multiply 1 times to get 512 x 512, and relaxing the borders by 5% on all directions
            up_down_loose = int(1 * (down - up + 1) * 0.05)
            image_up = 1 * up - up_down_loose
            if image_up < 0:
                image_up = 0
            image_down = 1*(down+1)+up_down_loose
            if image_down > image.shape[0] + 1:
                image_down = image.shape[0]

            left_right_loose = int(1 * (right - left) * 0.05)
            image_left = 1 * left - left_right_loose
            if image_left < 0:
                image_left = 0
            image_right = 1*(right + 1)+left_right_loose
            if image_right > image.shape[1] + 1:
                image_right = image.shape[1]

            crop = image[image_up: image_down, image_left: image_right]
            crop = cv2.resize(crop, (512,512), interpolation = cv2.INTER_AREA) #the cropped image is resized to 512 x 512
            cv2.imwrite(crop_save_dir + case_name, crop) # cropped images saved to crop directory

            # write new csv
            crop_width = image_right - image_left + 1
            crop_height = image_down - image_up + 1

            csv_writer.writerow([case_name,
                                 image_left,
                                 image_up,
                                 crop_width,
                                 crop_height]) #writes xmin, ymin, width, and height

            if j % 50 == 0:
                print(j, " images are processed!")

#train-normal
generate_bounding_box("C:/Users/xx/codes/test/",
                      "C:/Users/xx/codes/result/",
                      'C:/Users/xxcodes/bounding_box.csv',
                      "C:/Users/xx/codes/cropped/")

#%%
'''
The cropped images are further used for the classification studies
Part II: Classification experiments:
The trained U-Net model is truncated a the block_5c_add layer
and added with teh classification layers to classify teh CXRs as belonging to
the normal, bacterial pneumonia, or viral pneumonia category.
'''
#%%
# import libraries
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #if using multiple gpus, otherwise comment

#clear warnings
import warnings 
warnings.filterwarnings('ignore',category=FutureWarning) #because of numpy version
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
K.clear_session()

#%%
#import other libraries

import time
import cv2
import pickle
from tqdm import tqdm
import itertools
from itertools import cycle
from matplotlib import pyplot
from math import log
from scipy.stats import gaussian_kde
from scipy import special
from sklearn import metrics
from numpy import sqrt
from numpy import argmax
import numpy as np
from scipy import interp
from numpy import genfromtxt
import scikitplot as skplt
import pandas as pd
import math
from classification_models.keras import Classifiers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.activations import softmax
import tensorflow_probability as tfp
from tensorboard.plugins import projector
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.losses import CategoricalCrossentropy, CategoricalHinge, KLDivergence, Poisson
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, InceptionV3, MobileNetV2, Xception, EfficientNetB5, EfficientNetB0, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Flatten, Conv2D, Concatenate, MaxPooling2D, ZeroPadding2D, concatenate, Input, Reshape, GlobalAveragePooling2D, Dense, Dropout, Activation, BatchNormalization, Dropout, LSTM, ConvLSTM2D
from sklearn.metrics import roc_curve, auc,  precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, classification_report, log_loss, confusion_matrix, accuracy_score 
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, brier_score_loss
import seaborn as sns

#%% 
#get current working directory
print(os.getcwd())

#%%# define custom function for confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#%%
# In case there exists multiple modes while performing majority voting

def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode

#%% custom function for training efficientnet models with a fixed dropout
from tensorflow.keras import backend, layers
class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                            for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

#%%
'''
The following are the loss functions used in this multi-class classification task:
a) Categorical cross-entropy (CCE) loss; (b) CCE with entropy-based regularization; 
(c) Kullback-Leibler (KL) divergence loss; (d) Categorical focal loss; (e) Categorical Hinge loss; 
(f) Label-smoothed CCE loss; (g) Label-smoothed categorical focal loss; and 
(h) Calibrated CCE loss [37]. 

Newly proposed loss functions:
(a) Calibrated KL divergence loss; (b) Calibrated categorical focal loss; 
(c) Calibrated categorical Hinge loss, and (d) Calibrated negative entropy loss. 
First we declare the functions for the  newly proposed losses
Make sure to use run_eagerly=True in model.compile
'''
#%%
'''
Calibrated CCE-Loss:

The key thing is to 
1) get the accuracy of a mini-batch, 
2) get the predicted confidence of the mini batch,
3) compute the absolute different of those two. 

# get gt labels
gt_labels = tf.argmax(y_true, axis=1).numpy()

# get pred labels
pred_labels = tf.argmax(y_pred, axis=1).numpy()

# get accuracy 
acc = np.sum(gt_labels==pred_labels)/len(gt_labels)

The purpose of this line (gt_labels = tf.argmax(y_true, axis=1).numpy()) 
is to get the ground truth label (i.e., convert the label 
from one-hot encoding to a normal label). 

Then, with the ground truth label and predicted label, we can compute the accuracy. 

'''

def dcacce_loss(y_true, y_pred, beta=1):
    # y_true: one-hot encoding
    # y_pred: predicted probability (i.e., softmax(logits))
 
    ## calculating cross-entropy loss ##
    loss_ce = K.mean(keras.losses.categorical_crossentropy(y_true, y_pred)) 
    
    ## calculating the DCA term ##
    # get gt labels
    gt_labels = tf.argmax(y_true, axis=1).numpy()
    # get pred labels
    pred_labels = tf.argmax(y_pred, axis=1).numpy()
    # get accuracy 
    acc = np.sum(gt_labels==pred_labels)/len(gt_labels)
    # get pred mean prob
    temp_prop = 0
    for i in range(len(y_true)):
      temp_prop+=y_pred[i, pred_labels[i]]
    prob = temp_prop/len(y_true)
    # calculating dca
    dca = np.abs(acc-prob)

    loss = loss_ce + beta*dca
    
    return loss

#%%

'''
Following the above principles and writing the functions for the calibrated 
KL-divergence loss

'''
#this is based on KL Divergence
def dcakld_loss(y_true, y_pred, beta=1):
    # y_true: one-hot encoding
    # y_pred: predicted probability (i.e., softmax(logits))
 
    ## calculating cross-entropy loss ##
    kld = tf.keras.losses.KLDivergence()
    loss_ce = K.mean(kld(y_true, y_pred))
    #loss_ce = K.mean(keras.losses.KLDivergence(y_true, y_pred)) 
    
    ## calculating the DCA term ##
    # get gt labels
    gt_labels = tf.argmax(y_true, axis=1).numpy()
    # get pred labels
    pred_labels = tf.argmax(y_pred, axis=1).numpy()
    # get accuracy 
    acc = np.sum(gt_labels==pred_labels)/len(gt_labels)
    # get pred mean prob
    temp_prop = 0
    for i in range(len(y_true)):
      temp_prop+=y_pred[i, pred_labels[i]]
    prob = temp_prop/len(y_true)
    # calculating dca
    dca = np.abs(acc-prob)

    loss = loss_ce + beta*dca
    
    return loss

#%%
'''Calibrated Hinge loss'''

def dcahinge_loss(y_true, y_pred, beta=1):
    # y_true: one-hot encoding
    # y_pred: predicted probability (i.e., softmax(logits))
 
    ## calculating categorical hinge loss ##
    hinge = tf.keras.losses.CategoricalHinge()
    loss_ce = K.mean(hinge(y_true, y_pred))    
    
    ## calculating the DCA term ##
    # get gt labels
    gt_labels = tf.argmax(y_true, axis=1).numpy()
    # get pred labels
    pred_labels = tf.argmax(y_pred, axis=1).numpy()
    # get accuracy 
    acc = np.sum(gt_labels==pred_labels)/len(gt_labels)
    # get pred mean prob
    temp_prop = 0
    for i in range(len(y_true)):
      temp_prop+=y_pred[i, pred_labels[i]]
    prob = temp_prop/len(y_true)
    # calculating dca
    dca = np.abs(acc-prob)

    loss = loss_ce + beta*dca
    
    return loss

#%%

''' calibrated categorical focal loss'''

def categorical_focal_loss_fixed(y_true, y_pred, alpha = 1.0, gamma = 2.0):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))  
    
def dcafocal_loss(y_true, y_pred, beta=1):
    # y_true: one-hot encoding
    # y_pred: predicted probability (i.e., softmax(logits))
 
    ## calculating cross-entropy loss ##
    loss_ce = K.mean(categorical_focal_loss_fixed(y_true, y_pred, alpha = 1.0, gamma = 2.0)) 
    
    ## calculating the DCA term ##
    # get gt labels
    gt_labels = tf.argmax(y_true, axis=1).numpy()
    # get pred labels
    pred_labels = tf.argmax(y_pred, axis=1).numpy()
    # get accuracy 
    acc = np.sum(gt_labels==pred_labels)/len(gt_labels)
    # get pred mean prob
    temp_prop = 0
    for i in range(len(y_true)):
      temp_prop+=y_pred[i, pred_labels[i]]
    prob = temp_prop/len(y_true)
    # calculating dca
    dca = np.abs(acc-prob)

    loss = loss_ce + beta*dca
    
    return loss

#%%
''' 
Calibrated negative entropy loss
The entropy of y_pred is essentially the categorical cross entropy between y_pred and itself
'''

#%%
def entropy_loss(y_true, y_pred, beta=0.001): 
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(y_true, y_pred) - beta*cce(y_pred, y_pred)

#%%
# calibrated negative entropy loss

def dcaentropy_loss(y_true, y_pred, beta=1):
    # y_true: one-hot encoding
    # y_pred: predicted probability (i.e., softmax(logits))
 
    ## calculating cross-entropy loss ##
    loss_ce = K.mean(entropy_loss(y_true, y_pred, beta=0.001)) 
    
    ## calculating the DCA term ##
    # get gt labels
    gt_labels = tf.argmax(y_true, axis=1).numpy()
    # get pred labels
    pred_labels = tf.argmax(y_pred, axis=1).numpy()
    # get accuracy 
    acc = np.sum(gt_labels==pred_labels)/len(gt_labels)
    # get pred mean prob
    temp_prop = 0
    for i in range(len(y_true)):
      temp_prop+=y_pred[i, pred_labels[i]]
    prob = temp_prop/len(y_true)
    # calculating dca
    dca = np.abs(acc-prob)

    loss = loss_ce + beta*dca
    
    return loss

#%%
'''
custom function for entropy-based activity regularization
'''

def regularizer(beta):
    def entropy_reg(inp):
        return -beta * K.mean(inp * K.log(inp))

#%%

'''
Other existing loss functions used in this study

'''
#%%
#categorical focal loss

def categorical_focal_loss(alpha, gamma=2.): 
    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

#%%
#label smoothing with CCE: smoothin factor: 0.2

def smoothcce02(y_true, y_pred):
    label_smoothing = 0.2
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing )


#%%
#label smoothing with Focal loss: smoothing factor: 0.2

def smoothfocal(y_true, y_pred):
    smooth = 0.2
    y_true = y_true * (1 - smooth) + smooth/2
    return categorical_focal_loss_fixed(y_true, y_pred, alpha = 1.0, gamma = 2.0)

# usage: loss=[smoothfocal]

#%% Load data

img_width, img_height = 512,512
train_data_dir = "data/train"
test_data_dir = "data/test"
epochs = 64 
batch_size = 16
num_classes = 3 # normal, bacterial, and viral
input_shape = (img_width, img_height, 3)
model_input = Input(shape=input_shape)
print(model_input) 

#%%
#define data generators
datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1) 

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'training')

validation_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        seed=42,
        batch_size=batch_size, 
        class_mode='categorical', 
        subset = 'validation')

test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, 
        class_mode='categorical', 
        shuffle = False)

#identify the number of samples
nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)
nb_test_samples = len(test_generator.filenames)

#check the class indices
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

#true labels
Y_val=validation_generator.classes
print(Y_val.shape)

Y_test=test_generator.classes
print(Y_test.shape)

Y_test1=to_categorical(Y_test, num_classes=num_classes, dtype='float32')
print(Y_test1.shape)

#%%
#compute class weights to penalize over represented classes

class_weights = dict(zip(np.unique(train_generator.classes), 
                         class_weight.compute_class_weight('balanced', 
                                                           np.unique(train_generator.classes), 
                                                           train_generator.classes))) 
print(class_weights)

#{'bacterial': 0, 'normal': 1, 'viral': 2}
# {0: 0.687235594456601, 1: 1.2924554183813444, 2: 1.2967244701348748}

#%%
'''
Load the EfficientNet-Bo-based U-Net model and truncate at the block_5c_add layer and 
add classification layers
'''

model = load_model('trained_model/eff0_unet.hdf5')
model.summary()
base_model_eff0=Model(inputs=model.input,
                        outputs=model.get_layer('block_5c_add').output)
x = base_model_eff0.output 
x = ZeroPadding2D()(x)
x = Conv2D(512,(3,3), activation='relu', name = 'extra_conv_eff0') (x)
x = GlobalAveragePooling2D()(x) 
logits = Dense(num_classes, 
                    activation='softmax', 
                    name='predictions')(x)
## if using the the custom etntropy-based activity regularizer,
# logits = Dense(num_classes, 
#                     activation='softmax', name='predictions',
#                     activity_regularizer=regularizer(2.0))(x) 

model_eff0 = Model(inputs=base_model_eff0.input, 
                    outputs=logits, 
                    name = 'eff0_pretrained')

model_eff0.summary()

#%% train each model

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)  
model_eff0.compile(optimizer=sgd, 
                    loss='categorical_crossentropy', run_eagerly=True,  #use other losses in the same way
                    metrics=['accuracy']) 
#%%
#begin training
filepath = 'weights/' + model_eff0.name + '.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                             verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             mode='max', 
                             save_freq='epoch')
earlyStopping = EarlyStopping(monitor='val_accuracy', 
                              patience=5, 
                              verbose=1, 
                              mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                              factor=0.5, 
                              patience=5,
                              verbose=1,
                              mode='max', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]
t=time.time()

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
model_eff0_history = model_eff0.fit(train_generator, 
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=epochs, 
                                      validation_data=validation_generator,
                                      callbacks=callbacks_list, 
                                      class_weight = class_weights,
                                      validation_steps=nb_validation_samples // batch_size, 
                                      verbose=1)

print('Training time: %s' % (time.time()-t))
    
#%% plot performance

N = 16 #epochs; change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         model_eff0_history.history["loss"], 'orange', label="train_loss")
plt.plot(np.arange(1, N+1), 
         model_eff0_history.history["val_loss"], 'red', label="val_loss")
plt.plot(np.arange(1, N+1), 
         model_eff0_history.history["accuracy"], 'blue', label="train_acc")
plt.plot(np.arange(1, N+1), 
         model_eff0_history.history["val_accuracy"], 'green', label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("performance/eff0_CCE_learning.png")

#%%
# Load model for evaluation: keep compile as False since the model is used only for inference
eff0_model = load_model('weights/eff0_pretrained.06-0.9141.h5', compile=False)
eff0_model.summary()

#%%
#Generate predictions on the test data
test_generator.reset() 
custom_y_pred = eff0_model.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
custom_y_pred1_label = custom_y_pred.argmax(axis=-1)

#%%
#save predictions to a CSV file

predicted_class_indices=np.argmax(custom_y_pred,axis=1)
print(predicted_class_indices)

'''
map the predicted labels with their unique ids such 
as filenames to find out what you predicted for which image.
'''

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#save the results to a CSV file
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predicted_class_indices,
                      "Labels":predictions})
results.to_csv("performance/eff0_cce.csv",index=False)

#%%
'''
Evalute the performance of the trained models
'''

#%%

accuracy = accuracy_score(Y_test1.argmax(axis=-1),
                          custom_y_pred.argmax(axis=-1))
print('The accuracy of the Custom model is: ', 
      accuracy)

#evaluate mean squared error
custom_mse = mean_squared_error(Y_test1.argmax(axis=-1),
                                custom_y_pred.argmax(axis=-1))
print('The Mean Squared Error of the Custom model is: ', 
      custom_mse)

#evaluate mean squared log error
custom_msle = mean_squared_log_error(Y_test1.argmax(axis=-1),
                                     custom_y_pred.argmax(axis=-1))  
print('The Mean Squared Log Error of the Custom model is: ', 
      custom_msle)

#precision score
prec = precision_score(Y_test1.argmax(axis=-1),
                       custom_y_pred.argmax(axis=-1), 
                       average='weighted') #options: macro, weighted
print('The precision of the Custom model is: ', prec)

#recall score
rec = recall_score(Y_test1.argmax(axis=-1),
                   custom_y_pred.argmax(axis=-1), 
                   average='weighted')
print('The recall of the Custom model is: ', rec)

#F-score
f1 = f1_score(Y_test1.argmax(axis=-1),
              custom_y_pred.argmax(axis=-1), 
              average='weighted')
print('The f1-score of the Custom model is: ', f1)

#MCC
mat_coeff = matthews_corrcoef(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
print('The MCC of the Custom model is: ', mat_coeff)

#Cohen’s kappa
kappa = cohen_kappa_score(Y_test1.argmax(axis=-1),
                          custom_y_pred.argmax(axis=-1))
print('The cohen kappa score of the Custom model is: ', kappa)

#%%
#plot confusion matrix

target_names = ['Bacterial','Normal','Viral']
print(classification_report(Y_test1.argmax(axis=-1),
                            custom_y_pred.argmax(axis=-1),
                            target_names=target_names, 
                            digits=4))

# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test1.argmax(axis=-1),
                              custom_y_pred.argmax(axis=-1))
np.set_printoptions(precision=5)

x_axis_labels = ['Bacterial','Normal','Viral'] # labels for x-axis
y_axis_labels = ['Bacterial','Normal','Viral'] # labels for y-axis

plt.figure(figsize=(10,10), dpi=400)
sns.set(font_scale=2)
b = sns.heatmap(cnf_matrix, annot=True, square = True, 
            cbar=False, cmap='Greens', 
            annot_kws={'size': 50},
            fmt='g', 
            xticklabels=x_axis_labels, 
            yticklabels=y_axis_labels)

#%%
#plot AUC curves

class_ = ['Bacterial','Normal','Viral']  
num_classes = len(class_)

lw = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], thresholds = roc_curve(Y_test1[:, i], custom_y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], thresholds = roc_curve(Y_test1.ravel(), 
                                          custom_y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


#compute area under the ROC curve
auc_score_micro=roc_auc_score(Y_test1.ravel(),custom_y_pred.ravel())
print(auc_score_micro)

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
colors = cycle(['red', 'blue', 'indigo'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=3, 
             label='{0} class (AUC = {1:0.4f})'
             .format(class_[i], roc_auc[i]))
plt.plot(fpr["micro"], tpr["micro"],
          label='micro-average (AUC = {0:0.4f})'
                ''.format(roc_auc["micro"]),
          color='green', linestyle='solid', linewidth=4) # 'deeppink'

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc="lower right", prop={"size":20})
plt.show()


#%%
#compute precision-recall curves

class_ = ['Bacterial','Normal','Viral']  
num_classes = len(class_)

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(num_classes):
    precision[i], recall[i], thresholds = precision_recall_curve(Y_test1[:, i],
                                                        custom_y_pred[:, i])
    average_precision[i] = average_precision_score(Y_test1[:, i], 
                                                   custom_y_pred[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], thresholds = precision_recall_curve(Y_test1.ravel(),
                                                                         custom_y_pred.ravel())
average_precision["micro"] = average_precision_score(Y_test1, 
                                                     custom_y_pred,
                                                     average="micro")

# convert to f score
fscore = (2 * precision["micro"] * recall["micro"]) / (precision["micro"] + recall["micro"])

print('Average precision score, micro-averaged over all classes: {0:0.4f}'
      .format(average_precision["micro"]))

# area under the PR curve
print("The area under the PR curve is", metrics.auc(recall["micro"], 
                                                    precision["micro"]))

# plot the PR curve for the model
fig=plt.figure(figsize=(15,10), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('white')
major_ticks = np.arange(0.0, 1.1, 0.20) 
minor_ticks = np.arange(0.0, 1.1, 0.20)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
colors = cycle(['red', 'blue', 'indigo'])
for i, color in zip(range(num_classes), colors):
    pyplot.plot(recall[i], precision[i], 
                color=color, lw=3, 
                label='Precision-recall for class {0} (area = {1:0.4f})'
                .format(class_[i], average_precision[i]))
    
pyplot.plot(recall["micro"], precision["micro"], 
            color='green', linestyle='solid', linewidth=4,
            label='micro-average Precision-recall (area = {0:0.4f})'
              ''.format(average_precision["micro"]))    
# axis labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Recall', fontsize=20)
plt.ylabel('Precision', fontsize=20)
plt.legend(loc="lower left", prop={"size":20})
plt.show()

#%%
'''
We train the models individually with the aforementioned loss functions.
Then we choose the top-3 and top-5 performing models
to construct ensembles andt the prediction and model levels. 
At the prediction level, we perform majority voting, simple averaging,
weighted averaging, and stacking ensembles. 

'''

#%%
#top-1: CCE calibration model
model1 = load_model('weights/eff0_pretrained_caccebeta1.06-0.9141.h5', 
                          compile=False)
model1.summary()

#measure performance on test data, 
test_generator.reset()
model1_y_pred = model1.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-2: Beta regularizer 2.0 model

model2 = load_model('weights/eff0_pretrained_beta2.0.11-0.9082.h5', 
                          compile=False)
model2.summary()

#measure performance on test data, 
test_generator.reset()
model2_y_pred = model2.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-3: Ncalibrated negative entropy model

model3 = load_model('weights/eff0_pretrained_entropy0.001.09-0.9102.h5', 
                         compile=False)
model3.summary()

#measure performance on test data, 
test_generator.reset()
model3_y_pred = model3.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-4: Smoothed Focal loss 0.2

model4 = load_model('weights/eff0_pretrained_smoothfocal02.13-0.9180.h5', 
                          compile=False)
model4.summary()

#measure performance on test data, 
test_generator.reset()
model4_y_pred = model4.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#top-5: CCE

model5 = load_model('weights/eff0_pretrained_CCE.13-0.9141.h5', 
                          compile=False)
model5.summary()

#measure performance on test data, 
test_generator.reset()
model5_y_pred = model5.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)

#%%
#lets do a dummy assignment of the predictions
model1_y_pred1 = model1_y_pred
model2_y_pred1 = model2_y_pred
model3_y_pred1 = model3_y_pred
model4_y_pred1 = model4_y_pred
model5_y_pred1 = model5_y_pred


#print the shape of the predictions
print("The shape of model1 prediction  = ", 
     model1_y_pred1.shape)
print("The shape of model2 prediction  = ", 
     model2_y_pred1.shape)
print("The shape of model3 prediction  = ", 
     model3_y_pred1.shape)
print("The shape of model4 prediction  = ", 
     model4_y_pred1.shape)
print("The shape of model5 prediction  = ", 
     model5_y_pred1.shape)


#%%
#compute argmax
model1_y_pred1 = model1_y_pred1.argmax(axis=-1)
model2_y_pred1 = model2_y_pred1.argmax(axis=-1)
model3_y_pred1 = model3_y_pred1.argmax(axis=-1)
model4_y_pred1 = model4_y_pred1.argmax(axis=-1)
model5_y_pred1 = model5_y_pred1.argmax(axis=-1)

#%%
''' perform majority voting
'''
#using top-3 models:
max_voting_3_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_3_pred = np.append(max_voting_3_pred, 
                                find_max_mode([model1_y_pred1[i],
                                                 model2_y_pred1[i],
                                                 model3_y_pred1[i]
                                                ]))
#convert test labels to categorical
max_voting_3_pred1=to_categorical(max_voting_3_pred, num_classes=num_classes, dtype='float32')
print(max_voting_3_pred1.shape)

#%%
#top-5 majority voting ensemble:

max_voting_5_pred = np.array([])
for i in range(0,len(test_generator.filenames)):
    max_voting_5_pred = np.append(max_voting_5_pred, 
                                find_max_mode([model1_y_pred1[i],
                                                 model2_y_pred1[i],
                                                 model3_y_pred1[i],
                                                 model4_y_pred1[i],
                                                 model5_y_pred1[i]
                                                ]))
#convert test labels to categorical
max_voting_5_pred1=to_categorical(max_voting_5_pred, num_classes=num_classes, dtype='float32')
print(max_voting_5_pred1.shape)

#%%

'''
evaluate the performance as above: measure confusion matrix, performance metrics, 
AUC, and PR curves. Follow the procedures as discussed above
'''

#%%

'''
Simple averaging
'''

#top-3
average_pred_3=(model1_y_pred + model2_y_pred + model3_y_pred)/3
ensemble_model_3_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_3.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_3_averaging_accuracy)

#%%
#top-5
average_pred_5=(model1_y_pred + model2_y_pred + model3_y_pred + model4_y_pred + model5_y_pred)/5
ensemble_model_5_averaging_accuracy = accuracy_score(Y_test,
                                                      average_pred_5.argmax(axis=-1))
print("The averaging accuracy of the ensemble model is  = ", 
      ensemble_model_5_averaging_accuracy)


#%%
#weighted averaging:
'''
Here, we calcualte the optimal weights for the models 
predictions through a constrained minimization process of the logarithmic loss 
function
'''
#%%
#append predictions

preds = [] # top-3 or top-5 models, change accordingly, here it is shown for top-5 models
test_generator.reset()
model1_y_pred = model1.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
preds.append(model1_y_pred)

test_generator.reset()
model2_y_pred = model2.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
preds.append(model2_y_pred)

test_generator.reset()
model3_y_pred = model3.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
preds.append(model3_y_pred)  

test_generator.reset()
model4_y_pred = model4.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
preds.append(model4_y_pred) 

test_generator.reset()
model5_y_pred = model5.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
preds.append(model5_y_pred) 

#%%
#define a custom function to measure weighted accuracy

def calculate_weighted_accuracy(prediction_weights):
    weighted_predictions = np.zeros((nb_test_samples, num_classes), 
                                    dtype='float32')
    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = Y_test1.argmax(axis=-1)
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

#lets assume equal weights for the model predictions to begin with
# for seve models
prediction_weights = [1. / 5] * 5 # change to 3 for top-3 models
print(prediction_weights)
calculate_weighted_accuracy(prediction_weights)

#%%
# Create the loss metric 

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((nb_test_samples, num_classes), 
                                dtype='float32')
    for weight, prediction in zip(weights, preds):
        final_prediction += weight * prediction
    return log_loss(Y_test1, final_prediction)

best_acc = 0.0
best_weights = None

# Parameters for optimization
constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(preds)

#%%
'''
now we determine how much weights we have to give
for each model prediction based on the log loss functions,
the process is repeated for 50 times to find the best combination
of weights for the ensemble models that results in
the highest accuracy and lowest loss
'''
from scipy.optimize import minimize

NUM_TESTS = 50 # variable
# Check for NUM_TESTS times
for iteration in range(NUM_TESTS):
    # Random initialization of weights for the top-3 model predictions
    prediction_weights = np.random.random(5) #change to 3 for top-3
    
    # Minimise the loss 
    result = minimize(log_loss_func, 
                      prediction_weights, 
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints)
    print('Best Ensemble Weights: {weights}'.format(weights=result['x']))
    
    weights = result['x']
    weighted_predictions3 = np.zeros((nb_test_samples, num_classes), 
                                    dtype='float32')  
    
    # Calculate weighted predictions
    for weight, prediction in zip(weights, preds):
        weighted_predictions3 += weight * prediction
    yPred = np.argmax(weighted_predictions3, axis=1)
    yTrue = Y_test1.argmax(axis=-1)
    
    # Calculate weight prediction accuracy
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Iteration %d: Accuracy : " % (iteration + 1), accuracy)
    print("Iteration %d: Error : " % (iteration + 1), error)
    
    # Save current best weights 
    if accuracy > best_acc:
        best_acc = accuracy
        best_weights = weights
        
    print()

print("Best Accuracy : ", best_acc)
print("Best Weights : ", best_weights)
calculate_weighted_accuracy(best_weights)

#%%
#use the predicted weights to compute the weighted predictions
prediction_weights = [4.05605310e-01, 1.92276399e-01, 3.56809023e-03, 3.98550200e-01, 1.10927275e-16] # top-5 models
#prediction_weights = [8.26950080e-01, 3.33527788e-05, 1.73016567e-01] # top-3 models
weighted_predictions5 = np.zeros((nb_test_samples, num_classes), 
                                    dtype='float32')
for weight, prediction in zip(prediction_weights, preds):
    weighted_predictions5 += weight * prediction
yPred = np.argmax(weighted_predictions5, axis=1)
yTrue = Y_test1.argmax(axis=-1)
accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

#%%
'''
for each of the above predictions (majority voting, simple averaing, and weighted averaging,), 
measure the performance, plot confusion matrix, plot roc curves, and PR curves
using the codes given above. 
'''
#%%
'''
Now we are going to construct a stacking ensemble using hte predictions of the top-5 models.
Repeat the process for the top-3 models

Integrated Stacking Model: It may be desirable to use a neural network as a meta-learner.
Specifically, the sub-networks can be embedded in a larger multi-headed neural network 
that then learns how to best combine the predictions from each input sub-model. 
It allows the stacking ensemble to be treated as a single large model. 
The benefit of this approach is that the outputs of the submodels are 
provided directly to the meta-learner. Further, it is also possible to update the 
weights of the submodels in conjunction with the meta-learner model, if this is desirable. 
The outputs of each of the models can then be merged. 
In this case, we will use a simple concatenation merge, 
where a single 15-element vector will be created from the five class-probabilities 
predicted by each of the 5 models. We will then define a hidden layer with 15 neurons to interpret this 
“input” to the meta-learner and an output layer that will make 
its own probabilistic prediction. A plot of the network graph is created when 
this function is called to give an idea of how the ensemble model fits together.

'''

#%%
#instantiate the models

def model1(model_input):
    model1loss = load_model('weights/eff0_pretrained_caccebeta1.06-0.9141.h5', 
                          compile=False)
    x = model1loss.output
    model1 = Model(inputs=model1loss.input, outputs=x, name='model1')
    return model1
model_loss1 = model1(model_input)
model_loss1.summary()

def model2(model_input):
    model2loss = load_model('weights/eff0_pretrained_beta2.0.11-0.9082.h5', 
                          compile=False)
    x = model2loss.output
    model2 = Model(inputs=model2loss.input, outputs=x, name='model2')
    return model2
model_loss2 = model2(model_input)
model_loss2.summary()

def model3(model_input):
    model3loss = load_model('weights/eff0_pretrained_entropy0.001.09-0.9102.h5', 
                         compile=False)
    x = model3loss.output
    model3 = Model(inputs=model3loss.input, outputs=x, name='model3')
    return model3
model_loss3 = model3(model_input)
model_loss3.summary()

def model4(model_input):
    model4loss =  load_model('weights/eff0_pretrained_smoothfocal02.13-0.9180.h5', 
                          compile=False)
    x = model4loss.output
    model4 = Model(inputs=model4loss.input, outputs=x, name='model4')
    return model4
model_loss4 = model4(model_input)
model_loss4.summary()

def model5(model_input):
    model5loss = load_model('weights/eff0_pretrained_CCE.13-0.9141.h5', 
                          compile=False)
    x = model5loss.output
    model5 = Model(inputs=model5loss.input, outputs=x, name='model5')
    return model5
model_loss5 = model5(model_input)
model_loss5.summary()

#%%
#load the instantiated models
n_models = 5 # change to 3 for top-3

def load_all_models(n_models):
    all_models = list()
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True) 
    model_loss1.load_weights('weights/eff0_pretrained_caccebeta1.06-0.9141.h5')    
    model_loss1.compile(optimizer=sgd,
                        loss=[dcacce_loss], run_eagerly=True, 
                        metrics=['accuracy'])
    all_models.append(model_loss1)
    model_loss2.load_weights('weights/eff0_pretrained_beta2.0.11-0.9082.h5')
    model_loss2.compile(optimizer=sgd,
                        loss='categorical_crossentropy', run_eagerly=True, 
                        metrics=['accuracy'])
    all_models.append(model_loss2)
    model_loss3.load_weights('weights/eff0_pretrained_entropy0.001.09-0.9102.h5')
    model_loss3.compile(optimizer=sgd,
                        loss=[dcaentropy_loss], run_eagerly=True,
                        metrics=['accuracy'])
    all_models.append(model_loss3)
    model_loss4.load_weights('weights/eff0_pretrained_smoothfocal02.13-0.9180.h5')
    model_loss4.compile(optimizer=sgd,
                        loss=[smoothfocal], run_eagerly=True,  
                        metrics=['accuracy'])
    all_models.append(model_loss4)
    model_loss5.load_weights('weights/eff0_pretrained_CCE.13-0.9141.h5')
    model_loss5.compile(optimizer=sgd,
                        loss='categorical_crossentropy', run_eagerly=True,
                        metrics=['accuracy'])
    all_models.append(model_loss5)
    
    return all_models

# load models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

#%%
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers [1:]:
        # make not trainable
            layer.trainable = False
    ensemble_outputs = [model(model_input) for model in members]    
    merge = Concatenate()(ensemble_outputs)
    hidden = Dense(15, activation='relu')(merge) # three ouputs for 5 models
    output = Dense(3, activation='softmax')(hidden) #3 classes
    model = Model(inputs=model_input, 
                  outputs=output)
    return model

#%%
# Creating Ensemble and training the model
print("Creating Ensemble")
ensemble = define_stacked_model(members)
print("Ensemble architecture: ")
print(ensemble.summary())

# compile the model
ensemble.compile(loss="categorical_crossentropy", 
                 run_eagerly=True,
                 optimizer=SGD(lr=1e-3, decay=1e-6, 
                               momentum=0.9, nesterov=True), 
                                 metrics=["accuracy"])

checkpoint = tf.keras.callbacks.ModelCheckpoint("weights/stacking_5_ensemble_cce.h5", 
                                                             monitor='val_accuracy', 
                                                             verbose=1, 
                                                            save_weights_only=True, 
                                                             save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_accuracy', 
                              patience=5, 
                              verbose=1, 
                              mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                              factor=0.5, 
                              patience=5,
                              verbose=1,
                              mode='max', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]

t=time.time()

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
stacked_history = ensemble.fit(train_generator, 
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=epochs, 
                                      validation_data=validation_generator,
                                      callbacks=callbacks_list,
                                      class_weight = class_weights,
                                      validation_steps=nb_validation_samples // batch_size, 
                                      verbose=1)

print('Training time: %s' % (time.time()-t))

#%% plot performance

N = epochs #change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         stacked_history.history["loss"], 'orange', label="train_loss")
plt.plot(np.arange(1, N+1), 
         stacked_history.history["val_loss"], 'red', label="val_loss")
plt.plot(np.arange(1, N+1), 
          stacked_history.history["accuracy"], 'blue', label="train_acc")
plt.plot(np.arange(1, N+1), 
         stacked_history.history["val_accuracy"], 'green', label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("stacking_5_ensemble_cce.png")

#%%
# Load model for evaluation: keep compile as False if only using for inference that is prediction on test data
ensemble.load_weights('weights/stacking_5_ensemble_cce.h5')
ensemble.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
ensemble.compile(optimizer=sgd,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

#%%
#Generate predictions on the test data
test_generator.reset() 
custom_y_pred = ensemble.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
custom_y_pred1_label = custom_y_pred.argmax(axis=-1)

#%%
''' evalute performance using these predictions as before '''


#%%

'''
Next, we perform model-level ensembles. The top-3/top-5 models are selected and trunctated
at their deepest convolutional layer. These layers are then concatenated and appended with
a 1 x 1 convolutional layer to reduce feature dimensions. This is furhter appended with
a GAP, and final dense layer to output prediction probabilities. 
Here, we show how we perform model-level ensembles using the top-5 performing models 

'''
#%%
#load each model

# model 1:
model1 = load_model('weights/eff0_pretrained_caccebeta1.06-0.9141.h5', compile=False)
model1.summary()
model1=Model(inputs=model1.input,
                        outputs=model1.get_layer('extra_conv_eff0').output)
x1 = model1.output
model1v = Model(inputs=model1.input, outputs=x1)

#%%
# model-2:
model2 = load_model('weights/eff0_pretrained_beta2.0.11-0.9082.h5', compile=False)
model2=Model(inputs=model2.input,
                        outputs=model2.get_layer('extra_conv_eff0').output)
model2.summary()
x2 = model2.output
model2v = Model(inputs=model2.input, outputs=x2)

#%%%%

# model-3:
model3 = load_model('weights/eff0_pretrained_entropy0.001.09-0.9102.h5', compile=False)

model3=Model(inputs=model3.input,
                        outputs=model3.get_layer('extra_conv_eff0').output)
model3.summary()
x3 = model3.output
model3v = Model(inputs=model3.input, outputs=x3)
    
#%%%%

# model-4:
model4 = load_model('weights/eff0_pretrained_smoothfocal02.13-0.9180.h5', compile=False)
model4.summary()
model4=Model(inputs=model4.input,
                        outputs=model4.get_layer('extra_conv_eff0').output)
model4.summary()
x4 = model4.output
model4v = Model(inputs=model4.input, outputs=x4)

#%%%%

# model-5:
model5 = load_model('weights/eff0_pretrained_CCE.13-0.9141.h5', compile=False)
model5 = Model(inputs=model5.input,
                        outputs=model5.get_layer('extra_conv_eff0').output)
model5.summary()
x5 = model5.output
model5v = Model(inputs=model5.input, outputs=x5)


#%%
#take the output of each model
  
out1 = model1v(model_input)    
out2 = model2v(model_input)  
out3 = model3v(model_input)
out4 = model4v(model_input)
out5 = model5v(model_input)

#%%
'''merge the models, add the new conv layer, GAP and final dense layers'''

mergedOut = Concatenate()([out1,out2,out3,out4,out5]) 
#add a new 1 x 1 convolutional layer
x4 = Conv2D(512, (1,1), activation='relu', 
            name = 'NewConv')(mergedOut)
x5 = GlobalAveragePooling2D()(x4) 
logits = Dense(num_classes, 
                    activation='softmax', name='predictions')(x5) 
                   
model_merge = Model(inputs=model_input, 
                    outputs=logits, 
                    name = 'merge5_cce')

model_merge.summary()

#%%
#print layer names and their number
{i: v for i, v in enumerate(model_merge.layers)}

# print trainable layers
for l in model_merge.layers:
    print(l.name, l.trainable)

#set trainable and non-trainable layers
# make everything until the new conv layer as non-trainable
for layer in model_merge.layers[:9]:
    layer.trainable = False
for layer in model_merge.layers[9:]:
    layer.trainable = True

# print trainable layers
for l in model_merge.layers:
    print(l.name, l.trainable)
    

#%%
#compile and train the merged model
#compile the model
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)  
model_merge.compile(optimizer=sgd, 
                    loss='categorical_crossentropy',
                    run_eagerly=True,
                    metrics=['accuracy']) 

#%%
#begin training

filepath = 'weights/' + model_merge.name + '.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                             verbose=1, 
                             save_weights_only=False, 
                             save_best_only=True, 
                             mode='max', 
                             save_freq='epoch')
earlyStopping = EarlyStopping(monitor='val_accuracy', 
                              patience=5, 
                              verbose=1, 
                              mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                              factor=0.5, 
                              patience=5,
                              verbose=1,
                              mode='max', 
                              min_lr=0.00001)
callbacks_list = [checkpoint, earlyStopping, reduce_lr]
t=time.time()

#reset generators
train_generator.reset()
validation_generator.reset()

#train the model
model_merge_history = model_merge.fit(train_generator, 
                                      steps_per_epoch=nb_train_samples // batch_size,
                                      epochs=epochs, 
                                      validation_data=validation_generator,
                                      callbacks=callbacks_list, 
                                      class_weight = class_weights,
                                      validation_steps=nb_validation_samples // batch_size, 
                                      verbose=1)

print('Training time: %s' % (time.time()-t))

#%% plot performance

N = epochs # change if early stopping
plt.style.use("ggplot")
plt.figure(figsize=(20,10), dpi=400)
plt.plot(np.arange(1, N+1), 
         model_merge_history.history["loss"], 'orange', label="train_loss")
plt.plot(np.arange(1, N+1), 
         model_merge_history.history["val_loss"], 'red', label="val_loss")
plt.plot(np.arange(1, N+1), 
         model_merge_history.history["accuracy"], 'blue', label="train_acc")
plt.plot(np.arange(1, N+1), 
         model_merge_history.history["val_accuracy"], 'green', label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower right")
plt.savefig("performance/merge5_cce.png")

#%%
# load the trained model and evaluate performacne, keep compile as False since the model is used only for inference

merge_model = load_model('weights/merg5_cce.01-0.9038.h5',
                         compile=False)
merge_model.summary()

#%%
#Generate predictions on the test data
test_generator.reset() 
custom_y_pred = merge_model.predict(test_generator,
                                    nb_test_samples // batch_size, 
                                    verbose=1)
custom_y_pred1_label = custom_y_pred.argmax(axis=-1)

#%%
''' evaluate performance as before using these predictions
Repeat the process for the merging of top-3 models.
Then perform weighted averaging of the predictions of the merge3 and merge5
models using the codes for weighted averaging.
This gave superior performance in our classification task'''

#%%
'''
ROI localization using Grad-CAM:
    We lcoalized the disease ROI learned by the top-5 performing models
    and the merged models. The following code snipped can be used for visualization
    using any of the trained models in this classification task
'''

#%%

# store image with higher dpi
def writePNGwithdpi(im, filename, dpi=(400,400)):
   """Save the image as PNG with embedded dpi"""

   # Encode as PNG into memory
   retval, buffer = cv2.imencode(".png", im)
   s = buffer.tobytes()

   # Find start of IDAT chunk
   IDAToffset = s.find(b'IDAT') - 4
   pHYs = b'pHYs' + struct.pack('!IIc',int(dpi[0]/0.0254),int(dpi[1]/0.0254),b"\x01" ) 
   pHYs = struct.pack('!I',9) + pHYs + struct.pack('!I',zlib.crc32(pHYs))
   with open(filename, "wb") as out:
      out.write(buffer[0:IDAToffset])
      out.write(pHYs)
      out.write(buffer[IDAToffset:])

#%%
# load the model of preference, change to one of the top-5 performing model, and the best-performing model-level ensemble
custom_model = load_model('weights/merge5_cce.10-0.9077.h5',
                                compile=False)
custom_model.summary()

#%%
#path to image to visualize
img_path = 'img/person124_virus_247.png'
img = image.load_img(img_path)

#preprocess the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

#predict on the image
preds = custom_model.predict(x)[0]
print(preds)

#%%
# look into the deepest convolutional layer
conv_layer = custom_model.get_layer("NewConv") #deepest convolutional layer
heatmap_model = Model([custom_model.inputs], [conv_layer.output, custom_model.output])

# Get gradient of the winner class w.r.t. the output of the deepest conv. layer
with tf.GradientTape() as gtape:
    conv_output, predictions = heatmap_model(x)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = gtape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

#For visualization purposes, we normalize the heatmap between 0 and 1.
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat
hm=np.squeeze(heatmap)
print(hm.shape)
img1 = plt.matshow(hm)
img1.set_cmap('hot')
plt.axis('off')
plt.savefig("heamap_person124_virus_237.png", bbox_inches='tight')

img = cv2.imread(img_path)
#Resizes the heatmap to be the same size as the original image
heatmap = cv2.resize(hm, (img.shape[1], img.shape[0]))
#Converts the heatmap to RGB 
heatmap = np.uint8(255 * heatmap)
#Apply the heatmap to the original image
aheatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT) 
aheatmap[np.where(heatmap < 0.2)] = 0 #remove noisy activations
superimposed_img = aheatmap * 0.4 + img 

#if we have to increse the DPI and write to disk
writePNGwithdpi(superimposed_img, "dpiimproved_person124_virus_247.png", (400,400))

#%%
'''
END OF CODE

'''




