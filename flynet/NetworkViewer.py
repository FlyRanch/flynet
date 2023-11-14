from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
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

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

from keras.layers import Dense, Input, Concatenate
from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D, Add
from keras.layers import BatchNormalization, Activation, Dropout, SpatialDropout2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from keras.constraints import maxnorm
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.regularizers import l1, l2
from keras.utils import plot_model
from keras.utils import np_utils
from keras.utils import Sequence
from keras import optimizers
from keras import backend as K

class data_generator(Sequence):

    def __init__(self,art_file_loc,art_file_name,valid_frac,train_or_valid):
        self.art_file_loc = art_file_loc
        self.art_file_name = art_file_name
        self.valid_frac = valid_frac
        self.train_or_valid = train_or_valid
        self.N_train = 0
        self.N_valid = 0

    def open_data_set(self):
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
        #uv_out = np.copy(self.art_data_file['key_pts_uv_' + str(idx)])
        #uv_out = uv_out/192.0
        #pts_3D_out = np.copy(self.art_data_file['key_pts_3D_' + str(idx)])
        #sample_out.append(pts_3D_out)
        sample_out.append(self.art_data_file['label_' + str(idx)])
        return sample_out

    def close_data_set(self):
        self.art_data_file.close()

    def __len__(self):
        if self.train_or_valid == 0:
            data_len = self.N_train
        elif self.train_or_valid == 1:
            data_len = self.N_valid
        return data_len

    def __getitem__(self,ind):
        idx = self.indices[ind]
        sample = self.read_batch(idx)
        rand_shift = np.random.randint(9,size=3)-4
        X_sample = np.copy(sample[0])
        np.roll(X_sample[:,:,:,0,0],rand_shift[0],axis=2)
        np.roll(X_sample[:,:,:,0,0],rand_shift[1],axis=1)
        np.roll(X_sample[:,:,:,0,1],rand_shift[2],axis=2)
        #X_noise = (20.0/255.0)*np.random.rand(X_sample.shape[0],X_sample.shape[1],X_sample.shape[2],X_sample.shape[3],X_sample.shape[4])
        X = X_sample
        y = {'state': sample[1]}
        return (X, y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class data_generator_predictor(Sequence):

    def __init__(self,file_loc,file_name):
        self.file_loc = file_loc
        self.file_name = file_name

    def open_data_set(self):
        os.chdir(self.file_loc)
        self.data_file = h5py.File(self.file_name,'r',libver='latest')
        self.key_list = list(self.data_file.keys())
        self.N_batches = int(len(self.key_list)/2)
        self.indices = np.arange(self.N_batches)
        # Load batch to extract batch shape
        test_batch = self.data_file[str(self.key_list[0])]
        self.batch_shape = test_batch.shape

    def return_batch_shape(self):
        return self.batch_shape

    def return_N_batches(self):
        return self.N_batches

    def read_batch(self,idx):
        sample_out = []
        img_batch = np.copy(self.data_file[str(self.key_list[idx])])
        #sample_out.append(self.data_file[str(self.key_list[idx])]) # img batch
        sample_out.append(img_batch)
        sample_out.append(self.data_file[str(self.key_list[self.N_batches+idx])]) # label batch
        return sample_out

    def close_data_set(self):
        self.data_file.close()

    def __len__(self):
        return int(self.N_batches)

    def __getitem__(self,ind):
        idx = self.indices[(ind%self.N_batches)]
        sample = self.read_batch(idx)
        X = sample[0]
        Y = sample[1]
        return (X, Y)

    def on_epoch_end(self):
        #np.random.shuffle(self.indices)
        pass

class FullNetwork():

    def __init__(self):
        self.N_cam = 3
        self.output_dim = [120, 18]
        self.img_size = [192, 192]
        self.delta_s = 0.040

    def set_N_cam(self,N_cam_in):
        self.N_cam = N_cam_in

    def set_N_epochs(self,N_epochs_in):
        self.N_epochs = N_epochs_in

    def SetProjectionMatrixes(self,c_par,w2c_mat,c2w_mat):
        self.c_params = c_par
        self.w2c_matrices = w2c_mat
        self.c2w_matrices = c2w_mat
        self.uv_shift = []
        for i in range(self.N_cam):
            uv_trans = np.zeros((3,1))
            uv_trans[0] = self.img_size[0]/2.0
            uv_trans[1] = self.img_size[1]/2.0
            self.uv_shift.append(uv_trans)

    def set_learning_rate(self,learning_rate,decay):
        self.learning_rate = learning_rate
        self.decay = decay

    def set_artificial_data(self,art_file_loc,art_file_name):
        self.art_file_loc = art_file_loc
        self.art_file_name = art_file_name

    def set_img_file(self,img_file_loc,img_file_name):
        self.img_file_loc = img_file_loc
        self.img_file_name = img_file_name

    def set_network_loc(self,network_loc,net_weights_file):
        self.network_loc = network_loc
        self.net_weights_file = net_weights_file

    def set_calibration_loc(self,calib_file,ds,cam_mag,cam_dist):
        self.calib_file = calib_file
        self.ds = ds
        self.cam_mag = cam_mag
        self.cam_dist = cam_dist

    def create_render_window(self):
        # renderer
        self.ren = vtk.vtkRenderer()
        self.ren.SetUseDepthPeeling(True)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.ren.SetBackground(1.0, 1.0, 1.0)
        self.renWin.SetSize(192, 192)
        # initialize
        self.iren.Initialize()
        self.renWin.SetOffScreenRendering(1)
        #self.renWin.Render()

    def load_model(self,mdl_dir,mdl_name):
        os.chdir(mdl_dir + '/' + mdl_name)
        print(os.getcwd())
        sys.path.append(os.getcwd())
        from Drosophila import Drosophila
        self.mdl = Drosophila()
        self.create_render_window()
        self.mdl.SetRendering(self.ren,self.renWin,self.iren)
        self.ren.SetUseDepthPeeling(True)
        self.mdl.LoadCalibrationMatrix(self.calib_file)
        self.mdl.CalculateProjectionMatrixes(self.ds,self.cam_mag,self.cam_dist)
        self.mdl.ConstructModel()
        self.mdl.SetStartPose()
        #scale_list = [0.83,1.0,0.9,0.85,0.85]
        scale_list = [1.0,1.0,1.0,1.0,1.0]
        self.mdl.SetScale(scale_list)
        self.mdl.RandomBounds()
        self.mdl_selected = True

    def CalculateVoxelMaps(self,n_layer):
        Nu = int(self.img_size[0]/pow(2,n_layer))
        Nv = int(self.img_size[1]/pow(2,n_layer))
        Nx = int(self.img_size[0]/pow(2,n_layer))
        Ny = int(self.img_size[0]/pow(2,n_layer))
        Nz = int(self.img_size[0]/pow(2,n_layer))
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0
        N_vox = Nx*Ny*Nz
        ds = self.delta_s*pow(2,n_layer)
        projection_mats = []
        indices = []
        values = []
        dense_shape = []
        for n in range(self.N_cam):
            indices.append([])
            values.append([])
            dense_shape.append([])
        vox_cntr = 0
        for k in range(Nz):
            print('slice: ' + str(k+1) + '/' + str(Nz))
            for j in range(Ny):
                for i in range(Nx):
                    xyz = np.array([[x0-((Nx-1)/2.0)*ds+i*ds],[y0-((Ny-1)/2.0)*ds+j*ds],[z0-((Nz-1)/2.0)*ds+k*ds],[1.0]])
                    uv_coord = []
                    is_voxel = 0
                    for n in range(self.N_cam):
                        uv_coord.append((np.dot(self.w2c_matrices[n],xyz)+self.uv_shift[n])*(1.0/pow(2,n_layer)))
                        if uv_coord[n][0]>0.0 and uv_coord[n][0]<Nu:
                            if uv_coord[n][1]>0.0 and uv_coord[n][1]<Nv:
                                is_voxel += 1
                    if is_voxel == self.N_cam:
                        vox_cntr += 1
                        for n in range(self.N_cam):
                            uv_ind = int(uv_coord[n][1])*Nu+int(uv_coord[n][0]) # tensorflow reshape is Row-major!!!
                            indices[n].append([k*Nx*Ny+j*Nx+i,uv_ind])
                            values[n].append(1.0)
        print('number of voxels: ' + str(vox_cntr))
        for n in range(self.N_cam):
            dense_shape[n].append(N_vox)
            dense_shape[n].append(Nu*Nv)
        return (indices,values,dense_shape)

    def get_projection_matrices(self,n_layer):
        print('Calculating voxel map ...')
        proj_mats = self.CalculateVoxelMaps(n_layer)
        self.proj_matrices = []
        for n in range(self.N_cam):
            self.proj_matrices.append(tf.sparse.SparseTensor(indices=proj_mats[0][n],values=proj_mats[1][n],dense_shape=proj_mats[2][n]))
        print('Voxel map has been constructed')

    def vox_project(self,_,imgs):
        img_shape = imgs.shape
        for n in range(self.N_cam):
            img_vec = tf.reshape(imgs[:,:,:,n],[img_shape[0]*img_shape[1],img_shape[2]])
            if n==0:
                vox_tensor = tf.sparse.sparse_dense_matmul(self.proj_matrices[n],img_vec)
            else:
                vox_tensor = tf.multiply(vox_tensor,tf.sparse.sparse_dense_matmul(self.proj_matrices[n],img_vec))
        # reshape
        vox_slice = tf.reshape(tf.transpose(vox_tensor),[img_shape[0],img_shape[0],img_shape[0],img_shape[2]])
        return vox_slice

    def project2voxelspace(self,imgs):
        # Rearange imgs from list to tensor:
        img_stack = tf.stack(imgs,axis=-1)
        # Compute size array:
        img_shape = img_stack.shape
        t_input = tf.placeholder(tf.float32, shape=(img_shape[1],img_shape[1],img_shape[1],img_shape[3]))
        array_shape = tf.zeros_like(t_input)
        # Scan function
        vox_space = tf.scan(self.vox_project,img_stack,initializer=array_shape,infer_shape=True,back_prop=True)
        # Return
        return vox_space

    def build_network(self):
        self.input_shape = (self.img_size[0],self.img_size[1],1,self.N_cam)
        input_mdl = Input(shape=self.input_shape)
        branches_1 = []
        for n in range(self.N_cam):
            branch_n = Lambda(lambda x: x[:,:,:,:,n])(input_mdl)
            branch_n = Conv2D(96,kernel_size=(4,4),strides=2,padding='same',activation='relu')(branch_n)
            branch_n = Conv2D(128,kernel_size=(4,4),strides=2,padding='same',activation='relu')(branch_n)
            branches_1.append(branch_n)
        model_conc = Concatenate(axis=2)(branches_1)
        model_2D = Conv2D(256,kernel_size=(3,3),strides=3,padding='same',activation='relu')(model_conc)
        model_2D = Conv2D(384,kernel_size=(3,3),strides=2,padding='same',activation='relu')(model_2D)
        model_2D = Conv2D(512,kernel_size=(3,3),strides=2,padding='same',activation='relu')(model_2D)
        model_2D = Conv2D(1024,kernel_size=(4,4),strides=4,padding='same',activation='relu')(model_2D)
        model_2D = Flatten()(model_2D)
        fcn = Dropout(0.1)(model_2D)
        fcn = Dense(1024,activation='relu',use_bias=True)(fcn)
        fcn = Dense(512,activation='relu',use_bias=True)(fcn)
        fcn = Dense(256,activation='relu',use_bias=True)(fcn)
        model_S = Dense(self.output_dim[1],activation='linear',name='state')(fcn)
        # Combine models
        model = Model(inputs=input_mdl, outputs=model_S, name='FlyNet')
        return model

    def train_network(self):
        self.train_generator = data_generator(self.art_file_loc,self.art_file_name,0.1,0)
        self.valid_generator = data_generator(self.art_file_loc,self.art_file_name,0.1,1)
        self.train_generator.open_data_set()
        self.valid_generator.open_data_set()
        # Get batch size and N_train and N_valid:
        batch_shape = self.train_generator.return_batch_shape()
        print(batch_shape)
        self.batch_size = batch_shape[0]
        self.input_shape = batch_shape[1:]
        print(self.input_shape)
        #self.output_dim = [120, 18]
        self.output_dim = [60, 18]
        N_batches = self.train_generator.return_N_batches()
        print(N_batches)
        N_train = self.train_generator.return_N_train()
        print(N_train)
        N_valid = self.valid_generator.return_N_valid()
        print(N_valid)
        # Load / build network
        self.load_network()
        self.fly_net.fit_generator(generator=self.train_generator,
            steps_per_epoch = N_train,
            epochs = self.N_epochs,
            validation_data = self.valid_generator,
            validation_steps = N_valid,
            verbose = 1)
        os.chdir(self.network_loc)
        self.fly_net.save_weights(self.net_weights_file)
        self.train_generator.close_data_set()
        self.valid_generator.close_data_set()

    def load_network(self):
        self.fly_net = self.build_network()
        self.fly_net.compile(loss={'state': 'mse'}, 
            optimizer=optimizers.Adam(lr=self.learning_rate, decay=self.decay),
            metrics={'state': 'mse'})
        self.fly_net.summary()
        try:
            os.chdir(self.network_loc)
            self.fly_net.load_weights(self.net_weights_file)
            print('Weights loaded.')
        except:
            print('Could not load weights.')

    def pipeline_predict(self,img_batch_in):
        #pred_pts_3D = self.pts_3D_net.predict(img_batch_in)
        pred_state = self.fly_net.predict(img_batch_in)
        return pred_state

    def predict_on_dataset(self,data_loc,data_file_name,N_pred_batches):
        self.img_generator = data_generator_predictor(data_loc,data_file_name)
        self.img_generator.open_data_set()
        batch_shape = self.img_generator.return_batch_shape()
        batch_len = batch_shape[0]
        N_batches = self.img_generator.return_N_batches()
        output_mat = np.zeros((batch_len*N_pred_batches,self.output_dim[0]))
        state_out = np.zeros((batch_len*N_pred_batches,self.output_dim[1]))
        cntr = 0
        for i in range(0,N_pred_batches):
            print('batch: '+str(i+1))
            img_batch = self.img_generator.read_batch(i)
            pred_state = self.pipeline_predict(img_batch[0])
            #output_mat[(batch_len*cntr):(batch_len*(cntr+1)),:] = pred_3D_pts
            state_out[(batch_len*cntr):(batch_len*(cntr+1)),:] = pred_state
            cntr += 1
        self.img_generator.close_data_set()
        return state_out

    def predict_overlay_images(self,data_loc,data_file_name,output_dir):
        self.data_generator = data_generator_predictor(data_loc,data_file_name)
        self.data_generator.open_data_set()
        batch_shape = self.data_generator.return_batch_shape()
        self.batch_size = batch_shape[0]
        N_batches = self.data_generator.return_N_batches()
        for i in range(N_batches):
            print('batch: '+str(i+1))
            img_batch = self.data_generator.read_batch(i)
            #pred_3D_pts, pred_state = self.pipeline_predict(img_batch[0])
            pred_state = self.pipeline_predict(img_batch[0])
            for j in range(self.batch_size):
                img_real = (1.0-img_batch[0][j,:,:,0,:])*255
                self.mdl.OverlayPlots3(output_dir,img_real,pred_state[j,:],i*self.batch_size+j)
        self.data_generator.close_data_set()

    def predict_3D_keypoints(self,data_loc,data_file_name,output_dir):
        self.data_generator = data_generator_predictor(data_loc,data_file_name)
        self.data_generator.open_data_set()
        batch_shape = self.data_generator.return_batch_shape()
        self.batch_size = batch_shape[0]
        N_batches = self.data_generator.return_N_batches()
        for i in range(N_batches):
            print('batch: '+str(i+1))
            img_batch = self.data_generator.read_batch(i)
            pred_3D_pts, pred_state = self.pipeline_predict(img_batch[0])
            for j in range(self.batch_size):
                img_real = (1.0-img_batch[0][j,:,:,0,:])*255
                self.mdl.PlotKeyPoints(output_dir,img_real,pred_3D_pts[j,:],i*self.batch_size+j)
        self.data_generator.close_data_set()

class NetworkViewer(pg.GraphicsWindow):

    def __init__(self, parent=None):

        pg.GraphicsWindow.__init__(self)
        self.setParent(parent)

        self.w_sub = self.addLayout(row=0,col=0)

        self.v_list = []

        # Parameters:
        self.frame_nr = 0
        self.mov_nr = 0
        self.N_cam = 3
        self.mov_folders = ['mov_1','mov_2','mov_3','mov_4','mov_5','mov_6','mov_7','mov_8','mov_9','mov_10']
        self.cam_folders = ['cam_1','cam_2','cam_3','cam_4','cam_5','cam_6','cam_7','cam_8','cam_9','cam_10']
        self.frame_name = 'frame_'
        self.trig_modes = ['start','center','end']

        self.use_h5_load = False

        # Network:
        self.network = FullNetwork()

    def set_session_folder(self,session_folder,session_name):
        self.session_folder = session_folder
        self.session_name = session_name
        self.output_file_name = str(self.session_name) + '.h5'
        self.output_folder = self.session_folder +'/output'

    def set_N_cam(self,N_cam):
        self.N_cam = N_cam

    def set_N_mov(self,N_mov):
        self.N_mov = N_mov

    def setup_h5_img_load(self,img_h5_file,img_h5_N_batches,img_h5_batch_size):
        self.use_h5_load = True
        self.img_h5_file = img_h5_file
        self.img_h5_N_batches = img_h5_N_batches
        self.img_h5_batch_size = img_h5_batch_size

    def set_trigger_mode(self,trigger_mode,trigger_frame,start_frame,end_frame):
        self.trigger_mode = trigger_mode
        self.trigger_frame = trigger_frame
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_batch_size(self,batch_size_in):
        self.batch_size = batch_size_in

    def load_background(self,bckg_imgs_in):
        self.bckg_imgs = bckg_imgs_in

    def load_camera_calibration(self,calib_fldr,c_params,c2w_matrices,w2c_matrices):
        self.calib_fldr = calib_fldr
        self.c_params = c_params
        self.c2w_matrices = c2w_matrices
        self.w2c_matrices = w2c_matrices

    def load_crop_center(self,img_centers,crop_window_size,frame_size,thorax_cntr):
        self.img_centers = []
        self.frame_size = frame_size
        self.crop_window = np.zeros((self.N_cam,8),dtype=int)
        self.crop_window_size = crop_window_size
        self.thorax_center = thorax_cntr
        print('thorax center img viewer 2')
        print(self.thorax_center)
        for n in range(self.N_cam):
            self.img_centers.append([int(img_centers[n][0]),int(img_centers[n][1])])
            # Calculate crop window dimensions:
            u_L = int(img_centers[n][0])-int(crop_window_size[n][0]/2.0)
            u_R = int(img_centers[n][0])+int(crop_window_size[n][0]/2.0)
            v_D = int(self.frame_size[n][1]-img_centers[n][1])-int(crop_window_size[n][1]/2.0)
            v_U = int(self.frame_size[n][1]-img_centers[n][1])+int(crop_window_size[n][1]/2.0)
            if u_L >= 0 and u_R < self.frame_size[n][0]:
                if v_D >= 0 and v_U < self.frame_size[n][1]:
                    self.crop_window[n,0] = u_L
                    self.crop_window[n,1] = v_D
                    self.crop_window[n,2] = u_R
                    self.crop_window[n,3] = v_U
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = 0
                    self.crop_window[n,6] = crop_window_size[n][0]
                    self.crop_window[n,7] = crop_window_size[n][1]
                elif v_D < 0:
                    self.crop_window[n,0] = u_L
                    self.crop_window[n,1] = 0
                    self.crop_window[n,2] = u_R
                    self.crop_window[n,3] = v_U
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = -v_D
                    self.crop_window[n,6] = crop_window_size[n][0]
                    self.crop_window[n,7] = crop_window_size[n][1]
                elif v_U >= self.frame_size[n][1]:
                    self.crop_window[n,0] = u_L
                    self.crop_window[n,1] = v_D
                    self.crop_window[n,2] = u_R
                    self.crop_window[n,3] = self.frame_size[n][1]
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = 0
                    self.crop_window[n,6] = crop_window_size[n][0]
                    self.crop_window[n,7] = crop_window_size[n][1]-(v_U-self.frame_size[n][1])
            elif u_L < 0:
                if v_D >= 0 and v_U < self.frame_size[n][1]:
                    self.crop_window[n,0] = 0
                    self.crop_window[n,1] = v_D
                    self.crop_window[n,2] = u_R
                    self.crop_window[n,3] = v_U
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = 0
                    self.crop_window[n,6] = crop_window_size[n][0]
                    self.crop_window[n,7] = crop_window_size[n][1]
                elif v_D < 0:
                    self.crop_window[n,0] = 0
                    self.crop_window[n,1] = 0
                    self.crop_window[n,2] = u_R
                    self.crop_window[n,3] = v_U
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = -v_D
                    self.crop_window[n,6] = crop_window_size[n][0]
                    self.crop_window[n,7] = crop_window_size[n][1]
                elif v_U >= self.frame_size[n][1]:
                    self.crop_window[n,0] = 0
                    self.crop_window[n,1] = v_D
                    self.crop_window[n,2] = u_R
                    self.crop_window[n,3] = self.frame_size[n][1]
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = 0
                    self.crop_window[n,6] = crop_window_size[n][0]
                    self.crop_window[n,7] = crop_window_size[n][1]-(v_U-self.frame_size[n][1])
            elif u_R >=  self.frame_size[n][0]:
                if v_D >= 0 and v_U < self.frame_size[n][1]:
                    self.crop_window[n,0] = u_L
                    self.crop_window[n,1] = v_D
                    self.crop_window[n,2] = self.frame_size[n][0]
                    self.crop_window[n,3] = v_U
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = 0
                    self.crop_window[n,6] = crop_window_size[n][0]-(u_R-self.frame_size[n][0])
                    self.crop_window[n,7] = crop_window_size[n][1]
                elif v_D < 0:
                    self.crop_window[n,0] = u_L
                    self.crop_window[n,1] = 0
                    self.crop_window[n,2] = self.frame_size[n][0]
                    self.crop_window[n,3] = v_U
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = -v_D
                    self.crop_window[n,6] = crop_window_size[n][0]-(u_R-self.frame_size[n][0])
                    self.crop_window[n,7] = crop_window_size[n][1]
                elif v_U >= self.frame_size[n][1]:
                    self.crop_window[n,0] = u_L
                    self.crop_window[n,1] = v_D
                    self.crop_window[n,2] = self.frame_size[n][0]
                    self.crop_window[n,3] = self.frame_size[n][1]
                    self.crop_window[n,4] = 0
                    self.crop_window[n,5] = 0
                    self.crop_window[n,6] = crop_window_size[n][0]-(u_R-self.frame_size[n][0])
                    self.crop_window[n,7] = crop_window_size[n][1]-(v_U-self.frame_size[n][1])
        self.uv_offset = []
        self.uv_shift = []
        for n in range(self.N_cam):
            #uv_thorax = np.dot(self.w2c_matrices[n],self.thorax_center)
            uv_thorax = np.dot(self.w2c_matrices[n],np.array([[0.0],[0.0],[0.0],[1.0]]))
            uv_trans = np.zeros((3,1))
            uv_trans[0] = uv_thorax[0]-crop_window_size[n][0]/2.0 #+self.crop_window[n,0] #-(self.c_params[11,n]-self.frame_size[n][0])/2.0-self.crop_window[n,0]
            uv_trans[1] = uv_thorax[1]-crop_window_size[n][1]/2.0 #+self.crop_window[n,1] #-(self.c_params[10,n]-self.frame_size[n][1])/2.0-self.crop_window[n,1]
            self.uv_offset.append(uv_trans)
            self.uv_shift.append(np.zeros((3,1)))

    def load_frame(self):
        frame_list = []
        if self.use_h5_load:
            # open h5 file:
            img_data_h5 = h5py.File(self.img_h5_file,'r')
            frame_ind = 0
            if self.trigger_mode == 'start':
                frame_ind = self.frame_nr
            elif self.trigger_mode == 'center':
                if self.frame_nr < self.trigger_frame:
                    frame_ind = self.frame_nr+self.trigger_frame
                else:
                    frame_ind = self.frame_nr-self.trigger_frame
            elif self.trigger_mode == 'end':
                frame_ind = self.frame_nr
            else:
                print('error: invalid trigger mode')
            img_key = 'imgs_' + str(self.mov_nr+1) + '_' + str(frame_ind)
            img_batch = img_data_h5[img_key]
            for i in range(self.N_cam):
                img_batch_inv = 1.0-img_batch[0,:,:,0,i]
                # Crop window:
                img_cropped = np.ones((self.crop_window_size[i][0],self.crop_window_size[i][1]))
                img_cropped[self.crop_window[i,5]:self.crop_window[i,7],self.crop_window[i,4]:self.crop_window[i,6]] = img_batch_inv[self.crop_window[i,1]:self.crop_window[i,3],self.crop_window[i,0]:self.crop_window[i,2]]
                frame_list.append(img_cropped)
            img_data_h5.close()
        else:
            for i in range(self.N_cam):
                if self.trigger_mode == 'start':
                    frame_ind = self.frame_nr
                elif self.trigger_mode == 'center':
                    if self.frame_nr < self.trigger_frame:
                        frame_ind = self.frame_nr+self.trigger_frame
                    else:
                        frame_ind = self.frame_nr-self.trigger_frame
                elif self.trigger_mode == 'end':
                    frame_ind = self.frame_nr
                else:
                    print('error: invalid trigger mode')
                os.chdir(self.session_folder+'/'+self.mov_folder)
                os.chdir(self.cam_folders[i])
                img_cv = cv2.imread(self.frame_name + str(frame_ind) +'.bmp',0)
                img_cv = img_cv/255.0
                # Crop window:
                img_cropped = np.ones((self.crop_window_size[i][0],self.crop_window_size[i][1]))
                img_cropped[self.crop_window[i,5]:self.crop_window[i,7],self.crop_window[i,4]:self.crop_window[i,6]] = img_cv[self.crop_window[i,1]:self.crop_window[i,3],self.crop_window[i,0]:self.crop_window[i,2]]
                frame_list.append(img_cropped)
        return frame_list

    def set_output_progress(self,progress_in):
        self.output_progress = progress_in
        self.output_progress.setValue(0.0)

    def create_output_file(self):
        try:
            try:
                os.chdir(self.output_folder)
                print('output folder already exists, remove folder in order to overwrite')
            except:
                os.chdir(self.session_folder)
                os.mkdir('output')
                os.chdir('output')
                self.img_file = h5py.File(self.output_file_name,'w',libver='latest')
                N_frames_per_mov = self.end_frame-self.start_frame
                N_tot_frames = self.N_mov*N_frames_per_mov
                print('total number of frames: ' + str(N_tot_frames))
                self.output_progress.setValue(0.0)
                for i in range(self.N_mov):
                    print('movie ' + str(i+1))
                    self.mov_nr = i
                    batch_count = 0
                    if self.use_h5_load:
                        for j in range(self.start_frame,self.end_frame,self.batch_size):
                            print('batch: ' + str(batch_count))
                            img_out = np.zeros((self.batch_size,self.crop_window_size[0][0],self.crop_window_size[0][1],1,self.N_cam),dtype=np.float32)
                            label_out = np.zeros((self.batch_size,18),dtype=np.float32)
                            for k in range(0,self.batch_size):
                                self.frame_nr = j+k
                                frame_list = self.load_frame()
                                for n in range(self.N_cam):
                                    img_out[k,:,:,0,n] = np.clip(1.0-frame_list[n],0.0,1.0)
                            # median filtering:
                            #img_median = np.clip(np.min(img_out,axis=0,keepdims=True),0.0,1.0)
                            #img_batch = (img_out-img_median)
                            img_batch = img_out
                            self.img_file.create_dataset('imgs_' + str(i+1) + '_' + str(batch_count), data=img_batch, compression="lzf")
                            self.img_file.create_dataset('label_' + str(i+1) + '_' + str(batch_count), data=label_out, compression="lzf")
                            self.output_progress.setValue(100.0*(i*N_frames_per_mov+j)/(N_tot_frames-self.batch_size))
                            batch_count +=1
                    else:
                        self.mov_folder = self.mov_folders[i]
                        for j in range(self.start_frame,self.end_frame,self.batch_size):
                            print('batch: ' + str(batch_count))
                            img_out = np.zeros((self.batch_size,self.crop_window_size[0][0],self.crop_window_size[0][1],1,self.N_cam),dtype=np.float32)
                            label_out = np.zeros((self.batch_size,18),dtype=np.float32)
                            for k in range(0,self.batch_size):
                                self.frame_nr = j+k
                                frame_list = self.load_frame()
                                for n in range(self.N_cam):
                                    img_out[k,:,:,0,n] = np.clip(1.0-frame_list[n],0.0,1.0)
                            # median filtering:
                            #img_median = np.clip(np.min(img_out,axis=0,keepdims=True),0.0,1.0)
                            #img_batch = (img_out-img_median)
                            img_batch = img_out
                            self.img_file.create_dataset('imgs_' + str(i+1) + '_' + str(batch_count), data=img_batch, compression="lzf")
                            self.img_file.create_dataset('label_' + str(i+1) + '_' + str(batch_count), data=label_out, compression="lzf")
                            self.output_progress.setValue(100.0*(i*N_frames_per_mov+j)/(N_tot_frames-self.batch_size))
                            batch_count +=1
                self.img_file.close()
                print('created output file')
        except:
            print('error: could not create output file')
        os.chdir(self.output_folder)

    def set_encoder_progress(self,progress_in):
        print('no encoder initialized')

    def set_latent_dim(self,dim_in):
        print('no encoder initialized')

    def set_N_epochs_enc(self,N_in):
        print('no encoder initialized')

    def train_encoder(self):
        print('no encoder initialized')

    def load_encoder(self):
        print('no encoder initialized')

    def set_N_epochs_net(self,N_in):
        self.N_epochs_net = N_in

    def set_artificial_data(self,art_data_loc,art_data_file):
        self.art_data_loc = art_data_loc
        self.art_data_file = art_data_file

    def train_network(self):
        self.network.set_N_cam(self.N_cam)
        self.network.set_N_epochs(self.N_epochs_net)
        self.network.set_learning_rate(0.0005,0.00001)
        self.network.SetProjectionMatrixes(self.c_params,self.w2c_matrices,self.c2w_matrices)
        self.network.set_artificial_data(self.art_data_loc,self.art_data_file)
        self.network.set_img_file(self.art_data_loc,self.output_file_name)
        self.net_weights_file = 'network_weights.h5'
        self.network.set_network_loc(self.output_folder,self.net_weights_file)
        self.network.train_network()

    def predict_on_dataset(self):
        #self.net_weights_file = 'network_weights.h5'
        #self.network.set_network_loc(self.output_folder,self.net_weights_file)
        self.net_weights_file = 'network_weights.h5'
        self.network.set_network_loc(self.output_folder,self.net_weights_file)
        self.network.set_learning_rate(0.001,0.00001)
        self.network.SetProjectionMatrixes(self.c_params,self.w2c_matrices,self.c2w_matrices)
        self.network.load_network()
        state_out = self.network.predict_on_dataset(self.output_folder,self.output_file_name,100)
        self.plot_state_predictions(state_out)
        #self.plot_3D_keypoints(pts_3D_out)
        self.predict_overlay_images()
        #self.predict_3D_keypoints()

    def plot_3D_keypoints(self,data_in):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Left wing
        ax.plot3D(data_in[:,0],data_in[:,1],data_in[:,2], 'r')
        ax.plot3D(data_in[:,3],data_in[:,4],data_in[:,5], 'r')
        ax.plot3D(data_in[:,6],data_in[:,7],data_in[:,8], 'r')
        ax.plot3D(data_in[:,9],data_in[:,10],data_in[:,11], 'r')
        ax.plot3D(data_in[:,12],data_in[:,13],data_in[:,14], 'r')
        ax.plot3D(data_in[:,15],data_in[:,16],data_in[:,17], 'r')
        ax.plot3D(data_in[:,18],data_in[:,19],data_in[:,20], 'r')
        ax.plot3D(data_in[:,21],data_in[:,22],data_in[:,23], 'r')
        ax.plot3D(data_in[:,24],data_in[:,25],data_in[:,26], 'r')
        ax.plot3D(data_in[:,27],data_in[:,28],data_in[:,29], 'r')

        # Right wing
        ax.plot3D(data_in[:,30],data_in[:,31],data_in[:,32], 'b')
        ax.plot3D(data_in[:,33],data_in[:,34],data_in[:,35], 'b')
        ax.plot3D(data_in[:,36],data_in[:,37],data_in[:,38], 'b')
        ax.plot3D(data_in[:,39],data_in[:,40],data_in[:,41], 'b')
        ax.plot3D(data_in[:,42],data_in[:,43],data_in[:,44], 'b')
        ax.plot3D(data_in[:,45],data_in[:,46],data_in[:,47], 'b')
        ax.plot3D(data_in[:,48],data_in[:,49],data_in[:,50], 'b')
        ax.plot3D(data_in[:,51],data_in[:,52],data_in[:,53], 'b')
        ax.plot3D(data_in[:,54],data_in[:,55],data_in[:,56], 'b')
        ax.plot3D(data_in[:,57],data_in[:,58],data_in[:,59], 'b')

        ax.set_xlim([-3.0,3.0])
        ax.set_ylim([-3.0,3.0])
        ax.set_zlim([-3.0,3.0])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()

    def plot_state_predictions(self,data_mat):
        N_samples = data_mat.shape[0]
        t_vec = np.arange(0,N_samples)

        fig, axs = plt.subplots(4)
        fig.suptitle('Left wing pose')
        axs[0].plot(t_vec, data_mat[:,0],'k')
        axs[0].set_ylabel('scale')
        axs[1].plot(t_vec, data_mat[:,1],'k')
        axs[1].plot(t_vec, data_mat[:,2],'r')
        axs[1].plot(t_vec, data_mat[:,3],'g')
        axs[1].plot(t_vec, data_mat[:,4],'b')
        axs[1].set_ylabel('quaternion')
        axs[2].plot(t_vec, data_mat[:,5],'r')
        axs[2].plot(t_vec, data_mat[:,6],'g')
        axs[2].plot(t_vec, data_mat[:,7],'b')
        axs[2].set_ylabel('translation')
        axs[3].plot(t_vec, data_mat[:,8],'k')
        axs[3].set_ylabel('beta')

        fig, axs = plt.subplots(4)
        fig.suptitle('Right wing pose')
        axs[0].plot(t_vec, data_mat[:,9],'k')
        axs[0].set_ylabel('scale')
        axs[1].plot(t_vec, data_mat[:,10],'k')
        axs[1].plot(t_vec, data_mat[:,11],'r')
        axs[1].plot(t_vec, data_mat[:,12],'g')
        axs[1].plot(t_vec, data_mat[:,13],'b')
        axs[1].set_ylabel('quaternion')
        axs[2].plot(t_vec, data_mat[:,14],'r')
        axs[2].plot(t_vec, data_mat[:,15],'g')
        axs[2].plot(t_vec, data_mat[:,16],'b')
        axs[2].set_ylabel('translation')
        axs[3].plot(t_vec, data_mat[:,17],'k')
        axs[3].set_ylabel('beta')

        plt.show()

    def predict_overlay_images(self):
        calib_file = self.calib_fldr +'/cam_calib.txt'
        self.network.set_calibration_loc(calib_file,0.040,0.5,175.0)
        mdl_dir = '/home/flyami/Documents/FlyNet/FlyNet_V2/models'
        mdl_name = 'drosophila'
        self.network.load_model(mdl_dir,mdl_name)
        self.network.predict_overlay_images(self.output_folder,self.output_file_name,self.output_folder)

    def predict_3D_keypoints(self):
        calib_file = self.calib_fldr +'/cam_calib.txt'
        self.network.set_calibration_loc(calib_file,0.040,0.5,175.0)
        mdl_dir = '/home/flyami/Documents/FlyNet/FlyNet_V2/models'
        mdl_name = 'drosophila'
        self.network.load_model(mdl_dir,mdl_name)
        self.network.predict_3D_keypoints(self.output_folder,self.output_file_name,self.output_folder)
