from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import pickle
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import cv2
import time
import vtk
import h5py
import copy
import math
import gzip
import random
import pathlib
from datetime import datetime

import tensorflow as tf

from keras.layers import Dense, Input, Concatenate
from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D, Add
from keras.layers import BatchNormalization, LayerNormalization, Activation, Dropout, SpatialDropout2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, GaussianNoise, GaussianDropout
from keras.constraints import maxnorm
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, LogCosh, CosineSimilarity, mean_squared_logarithmic_error
from keras.regularizers import l1, l2
#from keras.utils import plot_model
from keras.utils import np_utils
#from keras.utils.all_utils import Sequence
from tensorflow.keras.utils import Sequence
from keras import optimizers
from keras import backend as K
from functools import partial

from sklearn.preprocessing import StandardScaler

class data_generator(Sequence):

    def __init__(self,art_file_loc,art_file_name,valid_frac,train_or_valid):
        self.art_file_loc = art_file_loc
        self.art_file_name = art_file_name
        self.valid_frac = valid_frac
        self.train_or_valid = train_or_valid
        self.N_train = 0
        self.N_valid = 0
        self.N_cam = 3

    def open_dataset(self):
        # Artificial data:
        self.open_artificial_data_set()

    def open_artificial_data_set(self):
        os.chdir(self.art_file_loc)
        self.art_data_file = h5py.File(self.art_file_name,'r',libver='latest')
        key_list = list(self.art_data_file.keys())
        label_keys = [lbl_key for lbl_key in key_list if 'label_' in lbl_key]
        self.N_batches = int(len(label_keys))
        if self.train_or_valid == 0:
            self.N_train = int((1.0-self.valid_frac)*self.N_batches)
            self.indices = np.arange(self.N_train)
        elif self.train_or_valid == 1:
            self.N_train = int((1.0-self.valid_frac)*self.N_batches)
            self.N_valid = self.N_batches-self.N_train
            self.indices = np.arange(self.N_train,self.N_batches)
        else:
            print('error: invalid train_or_valid number')
        np.random.shuffle(self.indices)
        # Load batch to extract batch shape
        test_batch = self.art_data_file['batch_' + str(0)]
        self.batch_shape = test_batch.shape
        self.batch_size = self.batch_shape[0]

    def return_N_train(self):
        return self.N_train

    def return_N_valid(self):
        return self.N_valid

    def return_batch_shape(self):
        return self.batch_shape

    def return_N_batches(self):
        return self.N_batches

    def read_batch(self,idx):
        sample_out = []
        sample_out.append(self.art_data_file['batch_' + str(idx)])
        sample_out.append(self.art_data_file['label_' + str(idx)])
        return sample_out

    def close_dataset(self):
        self.art_data_file.close()

    def set_calib_param(self,c2w_matrices,w2c_matrices,ds):
        self.ds = ds
        self.c2w_matrices = c2w_matrices
        self.w2c_matrices = w2c_matrices

    def project3D_to_uv(self,pts_in):
        uv_pts = []
        #zero_vec = np.array([[0.0],[0.0],[0.0],[1.0]])
        for n in range(self.N_cam):
            uv_n = np.dot(self.w2c_matrices[n],pts_in)
            uv_pts.append(np.dot(self.w2c_matrices[n],pts_in))
        return uv_pts

    def vibrate(self,sample_in):
        X_sample = np.copy(sample_in[0])
        Y_sample = np.copy(sample_in[1])
        for i in range(self.batch_size):
            # pick random 3D translation:
            rand_3D_vec = np.ones((4,1))
            rand_3D_vec[0:3] = np.random.randn(3,1)*2.0*self.ds
            # Compute u and v shifts
            rand_uv_shifts = self.project3D_to_uv(rand_3D_vec)
            for n in range(self.N_cam):
                u_roll = int(rand_uv_shifts[n][0])
                v_roll = int(rand_uv_shifts[n][1])
                np.roll(X_sample[i,:,:,0,n],u_roll,axis=1)
                np.roll(X_sample[i,:,:,0,n],v_roll,axis=0)
            # Label shift
            Y_sample[i,5] += np.squeeze(rand_3D_vec[0])
            Y_sample[i,6] += np.squeeze(rand_3D_vec[1])
            Y_sample[i,7] += np.squeeze(rand_3D_vec[2])
            Y_sample[i,14] += np.squeeze(rand_3D_vec[0])
            Y_sample[i,15] += np.squeeze(rand_3D_vec[1])
            Y_sample[i,16] += np.squeeze(rand_3D_vec[2])
        return X_sample, Y_sample

    def __len__(self):
        if self.train_or_valid == 0:
            data_len = self.N_train
        elif self.train_or_valid == 1:
            data_len = self.N_valid
        return data_len

    def __getitem__(self,ind):
        idx = self.indices[ind]
        sample = self.read_batch(idx)
        X = sample[0]
        y = {'state': sample[1]}
        return (X, y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class data_generator2(Sequence):

    def __init__(self,file_loc,file_name,valid_frac,train_or_valid):
        self.file_loc = file_loc
        self.file_name = file_name
        self.valid_frac = valid_frac
        self.train_or_valid = train_or_valid
        self.N_train = 0
        self.N_valid = 0
        self.N_cam = 3

    def open_dataset(self):
        file_path = pathlib.Path(self.file_loc) / self.file_name
        self.label_file = h5py.File(str(file_path),'r')
        self.label_keys = list(self.label_file.keys())
        self.N_labels = int(len(self.label_keys))
        self.indices = np.arange(0,self.N_labels)
        np.random.shuffle(self.indices)

    def close_dataset(self):
        self.label_file.close()

    def set_batch_size(self,batch_size_in):
        self.batch_size = batch_size_in
        self.create_batches()

    def create_batches(self):
        self.batch_indices = []
        batch_list = []
        for i,ind in enumerate(self.indices):
            if i%self.batch_size==0 and i>0:
                self.batch_indices.append(batch_list)
                batch_list = []
                batch_list.append(ind)
            else:
                batch_list.append(ind)
        self.N_batches = len(self.batch_indices)
        if self.train_or_valid==0:
            self.N_train = self.N_batches
        elif self.train_or_valid==1:
            self.N_valid = self.N_batches

    def return_N_train(self):
        return self.N_train

    def return_N_valid(self):
        return self.N_valid

    #def read_sample(self,idx):
    #    sample_out = []
    #    group = self.label_file[self.label_keys[idx]]
    #    frames = (np.copy(group['imgs']))
    #    sample_out.append(frames)
    #    state = group['label']
    #    sample_out.append(state)
    #    return sample_out

    def read_sample(self,idx):
        sample_out = []
        group = self.label_file[self.label_keys[idx]]
        frames = np.copy(group['frames'])
        sample_out.append(frames)
        state = group['state']
        sample_out.append(state)
        return sample_out

    def read_batch(self,idx):
        X = np.zeros((self.batch_size,224,224,1,self.N_cam))
        y = np.zeros((self.batch_size,37))
        for i,lbl_id in enumerate(self.batch_indices[idx]):
            sample = self.read_sample(lbl_id)
            scaled_sample = sample[0]
            if np.amax(sample[0])>1.0:
                scaled_sample = (255-scaled_sample)/255.
            noisy_sample = np.clip(np.random.rand(224,224,self.N_cam)*0.1,0,1)+scaled_sample
            X[i,:,:,0,:] = noisy_sample
            y[i,:] = sample[1]
        X = self.shake_it(X,40)
        #X = self.add_black_borders(X,20)
        y[:,0:4] = self.check_q(y[:,0:4])
        y[:,7:11] = self.check_q(y[:,7:11])
        y[:,14:18] = self.check_q(y[:,14:18])
        y[:,21:25] = self.check_q(y[:,21:25])
        y[:,29:33] = self.check_q(y[:,29:33])
        return X,y

    def setup_scaler(self):
        X_data = np.zeros((self.N_labels,37))
        for i in range(self.N_labels):
            group = self.label_file[self.label_keys[i]]
            X_data[i,:] = group['state']
        self.scaler = StandardScaler()
        self.scaler.fit(X_data)

    def set_scaler(self,scaler_in):
        self.scaler = scaler_in

    def return_scaler(self):
        return self.scaler

    def check_q(self,q_in):
        q_out = q_in
        for i in range(self.batch_size):
            if q_in[i,0]<0.0:
                q_out[i,:] = -1.0*q_in[i,:]
        return q_out

    def shake_it(self,X_in,uv_range):
        X_out = X_in
        uv_shift = np.random.randint(-uv_range,uv_range,2*self.N_cam)
        for n in range(self.N_cam):
            for i in range(self.batch_size):
                np.roll(X_out[i,:,:,0,n],uv_shift[2*n+0],axis=1)
                np.roll(X_out[i,:,:,0,n],uv_shift[2*n+1],axis=0)
        return X_out

    def add_black_borders(self,X_in,uv_range):
        X_out = X_in
        uv_shift = np.random.randint(-uv_range,uv_range,4*self.N_cam)
        for n in range(self.N_cam):
            for i in range(self.batch_size):
                X_out[:uv_shift[4*n+0],:] = 1
                X_out[:,:uv_shift[4*n+1]] = 1
                X_out[-uv_shift[4*n+2]:,:] = 1
                X_out[:,-uv_shift[4*n+3]:] = 1
        return X_out

    def invert_images(self,X_in):
        X_out = X_in
        rand_val = np.random.randn(1)
        if rand_val>0.5:
            X_out = 1.0-X_out
        return X_out

    def __len__(self):
        if self.train_or_valid == 0:
            data_len = self.N_train
        elif self.train_or_valid == 1:
            data_len = self.N_valid
        return data_len

    def __getitem__(self,ind):
        X,y = self.read_batch(ind)
        #X_shake = self.shake_it(X,20)
        #q_head = self.check_q(y[:,0:4])
        #t_head = y[:,4:7]
        #q_thorax = self.check_q(y[:,7:11])
        #t_thorax = y[:,11:14]
        #q_abdomen = self.check_q(y[:,14:18])
        #t_abdomen = y[:,18:21]
        #q_L = self.check_q(y[:,21:25])
        #t_L = y[:,25:28]
        #x_L = y[:,28]
        #q_R = self.check_q(y[:,29:33])
        #t_R = y[:,33:36]
        #x_R = y[:,36]
        q_head = y[:,0:4]
        t_head = y[:,4:7]
        q_thorax = y[:,7:11]
        t_thorax = y[:,11:14]
        q_abdomen = y[:,14:18]
        t_abdomen = y[:,18:21]
        q_L = y[:,21:25]
        t_L = y[:,25:28]
        x_L = y[:,28]
        q_R = y[:,29:33]
        t_R = y[:,33:36]
        x_R = y[:,36]
        y_out = {'q_h': q_head,
            't_h': t_head,
            'q_t': q_thorax,
            't_t': t_thorax,
            'q_a': q_abdomen,
            't_a': t_abdomen,
            'q_L': q_L,
            't_L': t_L,
            'x_L': x_L,
            'q_R': q_R,
            't_R': t_R,
            'x_R': x_R}
        return (X,y_out)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class Network():

    def __init__(self):
        self.N_cam = 3
        self.output_dim = 37
        self.input_shape = (224,224,1,self.N_cam)
        self.batch_size = 10

    def set_N_epochs(self,N_epochs_in):
        self.N_epochs = N_epochs_in

    def set_batch_size(self,batch_size_in):
        self.batch_size = batch_size_in

    def load_camera_calibration(self,c_params,c2w_matrices,w2c_matrices,ds):
        self.ds = ds
        self.c_params = c_params
        self.c2w_matrices = c2w_matrices
        self.w2c_matrices = w2c_matrices

    def set_calibration_loc(self,calib_file,ds,cam_mag,cam_dist):
        self.calib_file = calib_file
        self.ds = ds
        self.cam_mag = cam_mag
        self.cam_dist = cam_dist

    def LoadCalibrationMatrix(self):
        self.c_params = np.loadtxt(self.calib_file, delimiter='\t')
        self.N_cam = self.c_params.shape[1]

    def CalculateProjectionMatrixes(self):
        self.img_size = [224,224]
        self.w2c_matrices = []
        self.c2w_matrices = []
        for i in range(self.N_cam):
            #self.img_size.append((int(self.c_params[13,i]),int(self.c_params[12,i])))
            # Calculate world 2 camera transform:
            C = np.array([[self.c_params[0,i], self.c_params[2,i], 0.0, 0.0],
                [0.0, self.c_params[1,i], 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
            q0 = self.c_params[3,i]
            q1 = self.c_params[4,i]
            q2 = self.c_params[5,i]
            q3 = self.c_params[6,i]
            R = np.array([[2.0*pow(q0,2)-1.0+2.0*pow(q1,2), 2.0*q1*q2+2.0*q0*q3,  2.0*q1*q3-2.0*q0*q2],
                [2.0*q1*q2-2.0*q0*q3, 2.0*pow(q0,2)-1.0+2.0*pow(q2,2), 2.0*q2*q3+2.0*q0*q1],
                [2.0*q1*q3+2.0*q0*q2, 2.0*q2*q3-2.0*q0*q1, 2.0*pow(q0,2)-1.0+2.0*pow(q3,2)]])
            T = np.array([0.0,0.0,0.0])
            K = np.array([[R[0,0], R[0,1], R[0,2], T[0]],
                [R[1,0], R[1,1], R[1,2], T[1]],
                [R[2,0], R[2,1], R[2,2], T[2]],
                [0.0, 0.0, 0.0, 1.0]])
            W2C_mat = np.dot(C,K)
            C2W_mat = np.dot(np.linalg.inv(K),np.linalg.pinv(C))
            self.w2c_matrices.append(W2C_mat)
            self.c2w_matrices.append(C2W_mat)
        self.uv_shift = []
        for i in range(self.N_cam):
            uv_trans = np.zeros((3,1))
            uv_trans[0] = self.img_size[0]/2.0
            uv_trans[1] = self.img_size[1]/2.0
            self.uv_shift.append(uv_trans)

    def set_weights_file(self,weights_loc_in,weights_file_in):
        self.weights_loc = weights_loc_in
        self.weights_file = weights_file_in

    def set_learning_rate(self, learning_rate, decay, decay_steps=10000):
        self.learning_rate = learning_rate
        self.decay = decay
        self.decay_steps = decay_steps

    def set_network_loc(self,network_loc,net_weights_file):
        self.network_loc = network_loc
        self.net_weights_file = net_weights_file

    def set_artificial_data(self,art_file_loc,art_file_name):
        self.art_file_loc = art_file_loc
        self.art_file_name = art_file_name

    def set_annotated_data(self,man_file_loc,man_file_train,man_file_valid):
        self.man_file_loc = man_file_loc
        print(self.man_file_loc)
        self.man_file_train = man_file_train
        print(self.man_file_train)
        self.man_file_valid = man_file_valid
        print(self.man_file_valid)

    def load_network(self):
        self.fly_net = self.build_network()
        #self.fly_net.compile(loss={'state': 'mean_squared_logarithmic_error'}, 
        #    optimizer=optimizers.Adam(lr=self.learning_rate, decay=self.decay),
        #    metrics={'state': 'mean_squared_logarithmic_error'})
        #self.fly_net.compile(loss={'state': 'mse'}, 
        #    optimizer=optimizers.Adam(lr=self.learning_rate, decay=self.decay),
        #    metrics={'state': 'mse'})
        #self.fly_net.compile(loss={'state': 'mse'}, 
        #    optimizer=optimizers.SGD(lr=self.learning_rate, decay=self.decay),
        #    metrics={'state': 'mse'})
        #self.fly_net.compile(loss={'q_h': self.quaternion_loss,
        #    't_h': 'mae',
        #    'q_t': self.quaternion_loss,
        #    't_t': 'mae',
        #    'q_a': self.quaternion_loss,
        #    't_a': 'mae',
        #    'q_L': self.quaternion_loss,
        #    't_L': 'mae',
        #    'x_L': 'mae',
        #    'q_R': self.quaternion_loss,
        #    't_R': 'mae',
        #    'x_R': 'mae'},
        #    optimizer=optimizers.Adam(lr=self.learning_rate, decay=self.decay))
        #self.fly_net.compile(loss={'q_h': 'LogCosh',
        #    't_h': 'LogCosh',
        #    'q_t': 'LogCosh',
        #    't_t': 'LogCosh',
        #    'q_a': 'LogCosh',
        #    't_a': 'LogCosh',
        #    'q_L': 'LogCosh',
        #    't_L': 'LogCosh',
        #    'x_L': 'LogCosh',
        #    'q_R': 'LogCosh',
        #    't_R': 'LogCosh',
        #    'x_R': 'LogCosh'}, 
        #    optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay))

        #def quaternion_loss(self,y_true, y_pred):
        #    dist1 = tf.reduce_mean(tf.abs(y_true-y_pred), axis=-1)
        #    dist2 = tf.reduce_mean(tf.abs(y_true+y_pred), axis=-1)
        #    loss = tf.where(dist1<dist2, dist1, dist2)
        #    return tf.reduce_mean(loss)

        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #    initial_learning_rate=self.learning_rate, 
        #    decay_steps=self.decay_steps,
        #    decay_rate=self.decay,
        #    )
        
        self.fly_net.compile(loss={'q_h': 'mse',
            't_h': 'mse',
            'q_t': 'mse',
            't_t': 'mse',
            'q_a': 'mse',
            't_a': 'mse',
            'q_L': 'mse',
            't_L': 'mse',
            'x_L': 'mse',
            'q_R': 'mse',
            't_R': 'mse',
            'x_R': 'mse'}, 
            #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule))
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, weight_decay=self.decay))
        self.fly_net.summary()
        try:
            os.chdir(self.weights_loc)
            self.fly_net.load_weights(self.weights_file)
            print('Weights loaded.')
        except:
            print('Could not load weights.')

    #def weighted_loss(self,y_true,y_pred,weight):
    #    return K.mean(K.abs(y_true-y_pred)*weight)

    def weighted_loss(self,y_true,y_pred):
        return K.mean(K.square(y_true-y_pred))

    def quaternion_loss(self,y_true,y_pred):
        return K.mean(tf.math.acos(K.abs(tf.clip_by_value(tf.tensordot(-y_true,y_pred,axes=0),-0.9999,0.9999))))

    def build_network(self):
        input_mdl = Input(shape=self.input_shape)
        branches_1 = []
        noise_rate = 0.01
        dropout_rate = 0.05

        for n in range(self.N_cam):
            branch_n = Lambda(lambda x: x[:,:,:,:,n])(input_mdl)
            #branch_n = GaussianNoise(noise_rate)(branch_n)
            branch_n = Conv2D(32,kernel_size=(7,7),strides=2,padding='same',activation='selu')(branch_n)
            branch_n = BatchNormalization()(branch_n)
            branch_n = MaxPooling2D(pool_size=(2,2),strides=2)(branch_n)
            branch_n = Dropout(dropout_rate)(branch_n)
            branch_n = Conv2D(64,kernel_size=(1,1),strides=(1,1),padding='same',activation='selu')(branch_n)
            branch_n = BatchNormalization()(branch_n)
            branch_n = Dropout(dropout_rate)(branch_n)
            branch_n = Conv2D(64,kernel_size=(5,5),strides=2,padding='same',activation='selu')(branch_n)
            branch_n = BatchNormalization()(branch_n)
            branch_n = MaxPooling2D(pool_size=(2,2),strides=2)(branch_n)
            branch_n = Dropout(dropout_rate)(branch_n)
            branch_n = Conv2D(128,kernel_size=(1,1),strides=(1,1),padding='same',activation='selu')(branch_n)
            branch_n = BatchNormalization()(branch_n)
            branch_n = Dropout(dropout_rate)(branch_n)
            branch_n = Conv2D(128,kernel_size=(3,3),strides=2,padding='same',activation='selu')(branch_n)
            branch_n = BatchNormalization()(branch_n)
            branch_n = MaxPooling2D(pool_size=(2,2),strides=2)(branch_n)
            branch_n = Dropout(dropout_rate)(branch_n)
            branch_n = Conv2D(256,kernel_size=(1,1),strides=(1,1),padding='same',activation='selu')(branch_n)
            branch_n = BatchNormalization()(branch_n)
            branch_n = Dropout(dropout_rate)(branch_n)
            branch_n = Conv2D(256,kernel_size=(3,3),strides=(3,3),padding='same',activation='selu')(branch_n)
            branch_n = BatchNormalization()(branch_n)
            branch_n = Dropout(dropout_rate)(branch_n)
            branch_n = Conv2D(1024,kernel_size=(1,1),strides=(1,1),padding='same',activation='selu')(branch_n)
            branch_n = BatchNormalization()(branch_n)
            branches_1.append(branch_n)
        model_conc = Concatenate(axis=2)(branches_1)
        model_conc = Flatten()(model_conc)

        # body branch:
        body_dense = Dropout(dropout_rate)(model_conc)
        body_dense = Dense(256,activation='selu')(body_dense)
        body_dense = Dropout(dropout_rate)(body_dense)

        # q_head
        q_head = Dense(4,activation='linear',name='q_h')(body_dense)
        # t_head
        t_head = Dense(3,activation='linear',name='t_h')(body_dense)
        # q_thorax
        q_thorax = Dense(4,activation='linear',name='q_t')(body_dense)
        # t_thorax
        t_thorax = Dense(3,activation='linear',name='t_t')(body_dense)
        # q_abdomen
        q_abdomen = Dense(4,activation='linear',name='q_a')(body_dense)
        # t abdomen
        t_abdomen = Dense(3,activation='linear',name='t_a')(body_dense)

        # wing L branch:
        wing_L_dense = Dropout(dropout_rate)(model_conc)
        wing_L_dense = Dense(1024,activation='selu')(wing_L_dense)
        wing_L_dense = Dropout(dropout_rate)(wing_L_dense)
        #wing_L_dense = Dense(1024,activation='selu')(wing_L_dense)
        #wing_L_dense = Dropout(dropout_rate)(wing_L_dense)

        # q_L
        q_L = Dense(4,activation='linear',name='q_L')(wing_L_dense)
        # t_L
        t_L = Dense(3,activation='linear',name='t_L')(wing_L_dense)
        # x_L
        x_L = Dense(1,activation='linear',name='x_L')(wing_L_dense)

        # wing R branch:
        wing_R_dense = Dropout(dropout_rate)(model_conc)
        wing_R_dense = Dense(1024,activation='selu')(wing_R_dense)
        wing_R_dense = Dropout(dropout_rate)(wing_R_dense)
        #wing_R_dense = Dense(1024,activation='selu')(wing_R_dense)
        #wing_R_dense = Dropout(dropout_rate)(wing_R_dense)

        # q_R
        q_R = Dense(4,activation='linear',name='q_R')(wing_R_dense)
        # t_R
        t_R = Dense(3,activation='linear',name='t_R')(wing_R_dense)
        # x_R
        x_R = Dense(1,activation='linear',name='x_R')(wing_R_dense)
        
        output_list = [q_head,t_head,q_thorax,t_thorax,q_abdomen,t_abdomen,q_L,t_L,x_L,q_R,t_R,x_R]
        model = Model(inputs=input_mdl, outputs=output_list, name='FlyNet')
        return model


    def train_network(self):
        self.train_generator = data_generator2(self.man_file_loc,self.man_file_train,0.0,0)
        self.valid_generator = data_generator2(self.man_file_loc,self.man_file_valid,1.0,1)
        self.train_generator.open_dataset()
        self.valid_generator.open_dataset()
        self.train_generator.set_batch_size(self.batch_size)
        self.valid_generator.set_batch_size(self.batch_size)
        N_train = self.train_generator.return_N_train()
        N_valid = self.valid_generator.return_N_valid()

        print(f'N_train: {N_train}')
        print(f'N_valid: {N_valid}')

        # Load / build network
        self.load_network()

        # Create new weights file:
        date_now = datetime.now()
        history = self.fly_net.fit_generator( 
                generator = self.train_generator,
                steps_per_epoch = N_train,
                epochs = self.N_epochs,
                validation_data = self.valid_generator,
                validation_steps = N_valid,
                verbose = 1
                )

        print(f'type(history) = {type(history)}')
        print(f'type(history.history) = {type(history.history)}')
        print(f'type(history.history["loss"]) = {type(history.history["loss"])}')
        print(history.history.keys())

        # Save weights file
        dt_string = date_now.strftime("%d_%m_%Y_%H_%M_%S")
        weights_file_out = f'weights_{dt_string}.h5'
        self.fly_net.save_weights(weights_file_out)
        print(f'saved: {weights_file_out}')
        self.train_generator.close_dataset()
        self.valid_generator.close_dataset()

        # Save history file
        history_file_out = f'history_{dt_string}.pkl'
        with open(history_file_out, 'wb') as f:
            pickle.dump(histroy.history, f)
        print(f'saved: {history_file_out}')

        # Plot results:
        fig, axs = plt.subplots(6,1,sharex=True)
        t_epoch = np.arange(1,self.N_epochs+1)
        axs[0].plot(t_epoch,np.log10(history.history['loss']),color='b')
        axs[0].plot(t_epoch,np.log10(history.history['val_loss']),color='c')
        axs[1].plot(t_epoch,np.log10(history.history['q_h_loss']),color='r')
        axs[1].plot(t_epoch,np.log10(history.history['t_h_loss']),color='b')
        axs[1].plot(t_epoch,np.log10(history.history['val_q_h_loss']),color='m')
        axs[1].plot(t_epoch,np.log10(history.history['val_t_h_loss']),color='c')
        axs[2].plot(t_epoch,np.log10(history.history['q_t_loss']),color='r')
        axs[2].plot(t_epoch,np.log10(history.history['t_t_loss']),color='b')
        axs[2].plot(t_epoch,np.log10(history.history['val_q_t_loss']),color='m')
        axs[2].plot(t_epoch,np.log10(history.history['val_t_t_loss']),color='c')
        axs[3].plot(t_epoch,np.log10(history.history['q_a_loss']),color='r')
        axs[3].plot(t_epoch,np.log10(history.history['t_a_loss']),color='b')
        axs[3].plot(t_epoch,np.log10(history.history['val_q_a_loss']),color='m')
        axs[3].plot(t_epoch,np.log10(history.history['val_t_a_loss']),color='c')
        axs[4].plot(t_epoch,np.log10(history.history['q_L_loss']),color='r')
        axs[4].plot(t_epoch,np.log10(history.history['t_L_loss']),color='b')
        axs[4].plot(t_epoch,np.log10(history.history['x_L_loss']),color='g')
        axs[4].plot(t_epoch,np.log10(history.history['val_q_L_loss']),color='m')
        axs[4].plot(t_epoch,np.log10(history.history['val_t_L_loss']),color='c')
        axs[4].plot(t_epoch,np.log10(history.history['val_x_L_loss']),color='y')
        axs[5].plot(t_epoch,np.log10(history.history['q_R_loss']),color='r')
        axs[5].plot(t_epoch,np.log10(history.history['t_R_loss']),color='b')
        axs[5].plot(t_epoch,np.log10(history.history['x_R_loss']),color='g')
        axs[5].plot(t_epoch,np.log10(history.history['val_q_R_loss']),color='m')
        axs[5].plot(t_epoch,np.log10(history.history['val_t_R_loss']),color='c')
        axs[5].plot(t_epoch,np.log10(history.history['val_x_R_loss']),color='y')
        plt.show()
    

    def train_network_annotated(self):
        self.train_generator = data_generator(self.man_file_loc,self.man_file_name,0.1,0)
        self.valid_generator = data_generator(self.man_file_loc,self.man_file_name,0.1,1)
        self.train_generator.open_data_set()
        self.valid_generator.open_data_set()
        #self.train_generator.set_calib_param(self.c2w_matrices,self.w2c_matrices,self.ds)
        #self.valid_generator.set_calib_param(self.c2w_matrices,self.w2c_matrices,self.ds)
        # Get batch size and N_train and N_valid:
        batch_shape = self.train_generator.return_batch_shape()
        print(batch_shape)
        self.batch_size = batch_shape[0]
        self.input_shape = batch_shape[1:]
        print(self.input_shape)
        N_batches = self.train_generator.return_N_batches()
        print(N_batches)
        N_train = self.train_generator.return_N_train()
        print(N_train)
        N_valid = self.valid_generator.return_N_valid()
        print(N_valid)
        # Load / build network
        #self.load_network()
        # Create new weights file:
        date_now = datetime.now()
        dt_string = date_now.strftime("%d_%m_%Y_%H_%M_%S")
        weights_file_out = 'weights_' + dt_string + '.h5'
        print('weigth file out: ' + weights_file_out)
        self.fly_net.fit_generator(generator=self.train_generator,
            steps_per_epoch = N_train,
            epochs = self.N_epochs,
            validation_data = self.valid_generator,
            validation_steps = N_valid,
            verbose = 1)
        os.chdir(self.network_loc)
        self.fly_net.save_weights(weights_file_out)
        print('saved: ' + str(weights_file_out))
        self.train_generator.close_data_set()
        self.valid_generator.close_data_set()

    def predict_single_frame(self,imgs_in):
        imgs_scaled = (255-imgs_in)/255.0
        imgs = np.zeros((1,self.input_shape[0],self.input_shape[1],self.input_shape[2],self.input_shape[3]))
        imgs[0,:,:,0,:] = imgs_scaled
        state_pred = self.fly_net.predict(imgs)
        #print(state_pred)
        # Concatenate predictions into a single vector:
        state_out = np.zeros(37,dtype=np.float64)
        state_out[0:4] = np.squeeze(state_pred[0])
        state_out[4:7] = np.squeeze(state_pred[1])
        state_out[7:11] = np.squeeze(state_pred[2])
        state_out[11:14] = np.squeeze(state_pred[3])
        state_out[14:18] = np.squeeze(state_pred[4])
        state_out[18:21] = np.squeeze(state_pred[5])
        state_out[21:25] = np.squeeze(state_pred[6])
        state_out[25:28] = np.squeeze(state_pred[7])
        state_out[28] = np.squeeze(state_pred[8])
        state_out[29:33] = np.squeeze(state_pred[9])
        state_out[33:36] = np.squeeze(state_pred[10])
        state_out[36] = np.squeeze(state_pred[11])
        #for i in range(37):
        #    #i_item = state_pred[0,i].item()
        #    state_out[i] = i_item
        return state_out

    def predict_batch(self,img_batch_in,state_batch_in):
        n_batch = img_batch_in.shape[3]
        imgs_scaled = (255-img_batch_in)/255.0
        imgs = np.zeros((n_batch,self.input_shape[0],self.input_shape[1],self.input_shape[2],self.input_shape[3]))
        for j in range(n_batch):
            imgs[j,:,:,0,:] = imgs_scaled[:,:,:,j]
        state_pred = self.fly_net.predict(imgs)
        state_out = np.zeros((37,n_batch),dtype=np.float64)
        state_out[0:4,:] = np.squeeze(np.transpose(state_pred[0]))
        state_out[4:7,:] = np.squeeze(np.transpose(state_pred[1]))
        state_out[7:11,:] = np.squeeze(np.transpose(state_pred[2]))
        state_out[11:14,:] = np.squeeze(np.transpose(state_pred[3]))
        state_out[14:18,:] = np.squeeze(np.transpose(state_pred[4]))
        state_out[18:21,:] = np.squeeze(np.transpose(state_pred[5]))
        state_out[21:25,:] = np.squeeze(np.transpose(state_pred[6]))
        state_out[25:28,:] = np.squeeze(np.transpose(state_pred[7]))
        state_out[28,:] = np.squeeze(np.transpose(state_pred[8]))
        state_out[29:33,:] = np.squeeze(np.transpose(state_pred[9]))
        state_out[33:36,:] = np.squeeze(np.transpose(state_pred[10]))
        state_out[36,:] = np.squeeze(np.transpose(state_pred[11]))
        return state_out

    def train_on_tracked_data(self,results_folder,results_files,label_folder,n_samples_mov):
        # create hdf5 label file:
        os.chdir(label_folder)
        label_file = h5py.File('tracked_labels_valid.h5','w',libver='latest')
        # navigate to tracked results:
        os.chdir(results_folder)
        label_cntr = 0
        for file in results_files:
            # load filtered data
            try:
                tracked_data = h5py.File(file[0],'r',libver='latest')
                #key_list = list(tracked_data.keys())
                #traces_keys = [data_key for data_key in tracked_data.keys() if 'traces' in data_key]
                print(file[0])
                for i in file[1]:
                    # get filtered data:
                    trace_head       = np.copy(tracked_data['traces_m_'+str(i)+'_s_1_head']['filtered'])
                    trace_thorax  = np.copy(tracked_data['traces_m_'+str(i)+'_s_1_thorax']['filtered'])
                    trace_abdomen = np.copy(tracked_data['traces_m_'+str(i)+'_s_1_abdomen']['filtered'])
                    trace_wing_L  = np.copy(tracked_data['traces_m_'+str(i)+'_s_1_wing_L']['filtered'])
                    trace_wing_R  = np.copy(tracked_data['traces_m_'+str(i)+'_s_1_wing_R']['filtered'])
                    # random sampling of current movie:
                    r_samples = np.random.choice(trace_wing_L.shape[1],size=n_samples_mov)
                    # find the images of the random samples:
                    for j in range(n_samples_mov):
                        #print('label: '+str(label_cntr))
                        frame_ij_c1 = np.transpose(np.copy(tracked_data['frame_m_'+str(i)+'_f_'+str(r_samples[j]+1)+'_s_1_c_1']))
                        frame_ij_c2 = np.transpose(np.copy(tracked_data['frame_m_'+str(i)+'_f_'+str(r_samples[j]+1)+'_s_1_c_2']))
                        frame_ij_c3 = np.transpose(np.copy(tracked_data['frame_m_'+str(i)+'_f_'+str(r_samples[j]+1)+'_s_1_c_3']))
                        frame_ij = np.stack([frame_ij_c1,frame_ij_c2,frame_ij_c3],axis=2)
                        state_ij = np.zeros(37)
                        state_ij[0:7]   = trace_head[0:7,r_samples[j]]
                        state_ij[7:14]  = trace_thorax[0:7,r_samples[j]]
                        state_ij[14:21] = trace_abdomen[0:7,r_samples[j]]
                        state_ij[21:29] = trace_wing_L[0:8,r_samples[j]]
                        state_ij[29:37] = trace_wing_R[0:8,r_samples[j]]
                        #print(state_ij)
                        # save to label file:
                        grp_key = 'label_'+str(label_cntr)
                        grp = label_file.create_group(grp_key)
                        grp.create_dataset('frames',data=frame_ij)
                        grp.create_dataset('state',data=state_ij)
                        label_cntr += 1
                tracked_data.close()
                print(label_cntr)
            except:
                print('could not load: '+file[0])
        label_file.close()



# -------------------------------------------------------------------------------------------------------------------

#if __name__ == '__main__':
#    net = Network()
#    weights_folder = '/home/flythreads/Documents/FlyNet4/networks/FlyNet/weights'
#    #weights_file = 'weights_26_04_2021_20_48_12.h5'
#    #weights_file = 'weights_15_04_2021_10_28_25.h5'
#    #weights_file = 'weights_05_04_2021_20_37_49.h5'
#    #weights_file = 'weights_01_03_2022_11_39_07.h5'
#    weights_file = 'weights_23_03_2022_22_43_35.h5'
#    net.set_weights_file(weights_folder,weights_file)
#    net.set_learning_rate(1.0e-4,1.0e-6)
#    #net.set_learning_rate(1.0e-4,1.0e-6)
#
#    # create dataset based on tracked data:
#    #results_folder = '/media/flythreads/FlyamiDataB/opto_genetic_dataset/results'
#    #results_files  = ['seqs_Session_01_12_2020_10_22.h5py',
#    #    'seqs_Session_01_12_2020_11_25.h5py',
#    #    'seqs_Session_01_12_2020_12_34.h5py',
#    #    'seqs_Session_01_12_2020_13_00.h5py',
#    #    'seqs_Session_01_12_2020_14_24.h5py',
#    #    'seqs_Session_02_12_2020_13_59.h5py',
#    #    'seqs_Session_02_12_2020_15_59.h5py',
#    #    'seqs_Session_03_12_2020_12_52.h5py']
#
#    results_folder = '/media/flythreads/FlyamiDataB/flyami/results'
#    results_files = [
#        ['seqs_Session_04_01_2021_16_14.h5py', [1,2]],
#        ['seqs_Session_04_01_2021_16_30.h5py', [1,2,3,4,5,6,8]],
#        ['seqs_Session_08_02_2021_15_14.h5py', [1,2,3,4,5,6]],
#        ['seqs_Session_17_02_2021_13_18.h5py', [1,2,3,5,7,8]],
#        ['seqs_Session_02_03_2021_11_54.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_28_10_2020_13_18.h5py', [1,2,3]],
#        ['seqs_Session_06_11_2020_10_55.h5py', [1,2,3,4,5,6,8]],
#        ['seqs_Session_13_11_2020_11_31.h5py', [1,2,3]],
#        ['seqs_Session_17_11_2020_12_38.h5py', [1,2,3,4,5,6,7]],
#        ['seqs_Session_06_01_2021_14_02.h5py', [1,2,3,4,5,6,7]],
#        ['seqs_Session_10_02_2021_12_59.h5py', [1,2,3,4,5,6]],
#        ['seqs_Session_05_03_2021_14_32.h5py', [1]],
#        ['seqs_Session_31_03_2021_16_07.h5py', [1]],
#        ['seqs_Session_02_11_2020_11_15.h5py', [3,4,5,6,7]],
#        ['seqs_Session_08_12_2020_14_40.h5py', [1]],
#        ['seqs_Session_05_01_2021_15_21.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_19_02_2021_13_00.h5py', [1,2,3,4,5,6,7]],
#        ['seqs_Session_02_03_2021_12_50.h5py', [1,2,3]],
#        ['seqs_Session_04_03_2021_11_17.h5py', [1,2,3,4,6,7]],
#        ['seqs_Session_04_03_2021_15_06.h5py', [2,5,7,8]],
#        ['seqs_Session_28_10_2020_11_58.h5py', [1,3,4,5,6,7,8]],
#        ['seqs_Session_06_11_2020_11_36.h5py', [1,2,3]],
#        ['seqs_Session_05_01_2021_14_55.h5py', [1]],
#        ['seqs_Session_04_02_2021_12_44.h5py', [2]],
#        ['seqs_Session_08_02_2021_14_31.h5py', [1,2,3,4,5,6,7]],
#        ['seqs_Session_27_02_2021_13_19.h5py', [1,2,4,6]],
#        ['seqs_Session_01_03_2021_14_58.h5py', [1,2,4]],
#        ['seqs_Session_03_03_2021_14_57.h5py', [1,2,4,6,7,8]],
#        ['seqs_Session_11_03_2021_10_51.h5py', [1,2,3]],
#        ['seqs_Session_08_12_2020_15_30.h5py', [1,2,3,4]],
#        ['seqs_Session_17_02_2021_14_16.h5py', [1,3,4,5,6,7,8]],
#        ['seqs_Session_22_02_2021_12_45.h5py', [1,2,4,5]],
#        ['seqs_Session_27_02_2021_14_57.h5py', [1,2,3]],
#        ['seqs_Session_27_02_2021_15_12.h5py', [1]],
#        ['seqs_Session_22_03_2021_11_19.h5py', [1]],
#        ['seqs_Session_24_03_2021_13_39.h5py', [1,2]],
#        ['seqs_Session_13_11_2020_10_23.h5py', [2,3,4,5,7]],
#        ['seqs_Session_17_11_2020_14_34.h5py', [1,3]],
#        ['seqs_Session_17_02_2021_12_27.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_01_03_2021_11_14.h5py', [1,2,3,5]],
#        ['seqs_Session_08_11_2020_11_56.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_17_11_2020_13_51.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_04_01_2021_14_42.h5py', [1,2,3,4,5,6,7]],
#        ['seqs_Session_06_01_2021_15_32.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_07_01_2021_14_17.h5py', [1,2,3,4,5,6,7]],
#        ['seqs_Session_04_02_2021_13_51.h5py', [1,2]],
#        ['seqs_Session_19_02_2021_15_47.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_22_02_2021_11_36.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_25_02_2021_13_35.h5py', [1,2,3]],
#        ['seqs_Session_25_02_2021_14_05.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_25_02_2021_14_05.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_28_10_2020_14_05.h5py', [1,2,3]],
#        ['seqs_Session_08_11_2020_12_24.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_15_01_2021_15_35.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_18_02_2021_14_16.h5py', [1,2,3,4]],
#        ['seqs_Session_04_03_2021_12_45.h5py', [1,2,3,4,5]],
#        ['seqs_Session_23_03_2021_13_19.h5py', [1,2,3]],
#        ['seqs_Session_04_11_2020_13_43.h5py', [1,2,3,4,6,8]],
#        ['seqs_Session_05_01_2021_13_40.h5py', [8]],
#        ['seqs_Session_15_01_2021_16_34.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_10_02_2021_12_08.h5py', [1,3,4,5,6,7,8]],
#        ['seqs_Session_01_03_2021_11_42.h5py', [1,2,3,4,5]],
#        ['seqs_Session_03_03_2021_15_32.h5py', [1,2]],
#    man_file_train = 'tracked_labels.h5'
#    man_file_valid = 'tracked_labels_valid.h5'
#    #man_file_train = 'labels.h5'
#    #man_file_valid = 'valid_labels.h5'
#    #man_file_train = 'labels.h5'
#    #man_file_valid = 'tracked_labels_valid.h5'
#    net.set_annotated_data(man_file_loc,man_file_train,man_file_valid)
#    net.set_batch_size(500)
#    net.load_network()
#    net.set_N_epochs(50)
#    net.train_network()
#    plt.show()
#        ['seqs_Session_13_03_2021_13_53.h5py', [1,2,3,4,5]],
#        ['seqs_Session_08_11_2020_11_24.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_04_01_2021_15_30.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_05_01_2021_14_17.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_07_01_2021_15_55.h5py', [1]],
#        ['seqs_Session_08_02_2021_15_51.h5py', [1,2,3,4,5]],
#        ['seqs_Session_01_03_2021_14_26.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_05_03_2021_11_35.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_06_11_2020_12_16.h5py', [1,2,3,4,5,6,7,8]],
#        ['seqs_Session_04_02_2021_11_58.h5py', [4,5]],
#        ['seqs_Session_22_02_2021_15_57.h5py', [1,2,3,4,5,6,8]],
#        ['seqs_Session_05_03_2021_10_53.h5py', [1,2,3,4,5,6,7,8]]
#    ]
#
#
#    man_file_loc = '/home/flythreads/Documents/FlyNet4/networks/labels'
#
#    #net.train_on_tracked_data(results_folder,results_files,man_file_loc,10)
#
#    
#    man_file_train = 'tracked_labels.h5'
#    man_file_valid = 'tracked_labels_valid.h5'
#    #man_file_train = 'labels.h5'
#    #man_file_valid = 'valid_labels.h5'
#    #man_file_train = 'labels.h5'
#    #man_file_valid = 'tracked_labels_valid.h5'
#    net.set_annotated_data(man_file_loc,man_file_train,man_file_valid)
#    net.set_batch_size(500)
#    net.load_network()
#    net.set_N_epochs(50)
#    net.train_network()
#    plt.show()
