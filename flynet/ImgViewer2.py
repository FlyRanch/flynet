from __future__ import print_function
import sys
import io
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidget, QTableWidgetItem, QVBoxLayout, QFileDialog
import numpy as np
import numpy.matlib
import os
import os.path
from os import path
import copy
import time
import json
import h5py
import cv2
import math
import random
import vtk

import pySciCam
from pySciCam.pySciCam import ImageSequence
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re


class Graph(pg.GraphItem):
    def __init__(self,graph_nr):
        self.graph_nr = graph_nr
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)
        self.onMouseDragCb = None
        
    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = copy.deepcopy(kwds)
        
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text,self.data)
        self.updateGraph()
        
    def setTexts(self, text, data):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        #for t in text:
        for i,t in enumerate(text):
            item = pg.TextItem(t)
            if len(data.keys())>0:
                item.setColor(data['textcolor'][i])
            self.textItems.append(item)
            item.setParentItem(self)
        
    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i,item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def setOnMouseDragCallback(self, callback):
        self.onMouseDragCb = callback
        
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        
        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first 
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        
        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()
        if self.onMouseDragCb:
            PosData = self.data['pos'][ind]
            PosData = np.append(PosData,ind)
            PosData = np.append(PosData,self.graph_nr)
            self.onMouseDragCb(PosData)
        
    def clicked(self, pts):
        print("clicked: %s" % pts)

#class ImgViewer2(pg.GraphicsWindow):
class ImgViewer2(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None):
        #pg.GraphicsWindow.__init__(self)
        pg.GraphicsLayoutWidget.__init__(self)
        self.setParent(parent)

        self.w_sub = self.addLayout(row=0,col=0)

        self.v_list = []
        self.img_list = []
        self.frame_list = []
        self.seq_id_list = []

        # Parameters:
        self.frame_nr = 0
        self.N_cam = 3
        self.frame_name = 'frame_'
        self.trig_modes = ['start','center','end']
        self.graph_list = []

        self.window_size = [224,224]

        self.img_view_on = True
        self.seg_view_on = False

        self.batch_size = 100
        self.batch_mov_nr = -1
        self.batch_seq_nr = -1
        self.batch_frame_nr = -1

    def set_session_folder(self,session_folder,session_name):
        self.session_folder = session_folder
        self.session_name = session_name

    def set_seq_file(self,save_fldr_in,seq_file_name_in):
        self.save_fldr = save_fldr_in
        self.seq_file_name = seq_file_name_in
        os.chdir(self.save_fldr)
        self.seq_file = h5py.File(self.seq_file_name,'r+')
        self.sort_keys()
        #self.seq_file = seq_file_in
        #self.sort_keys()

    def set_N_cam(self,N_cam):
        self.N_cam = N_cam

    def set_N_mov(self,N_mov):
        self.N_mov = N_mov

    def set_cam_folders(self,cam_folders_in):
        self.cam_folders = cam_folders_in

    def set_mov_folders(self,mov_folders_in):
        self.mov_folders = mov_folders_in

    def set_trigger_mode(self,trigger_mode,trigger_frame,start_frame,end_frame):
        self.trigger_mode = trigger_mode
        self.trigger_frame = trigger_frame
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_movie_nr(self,mov_nr):
        self.mov_nr = mov_nr-1
        try:
            self.update_seq_spin()
        except:
            print('could not update seq spin')

    def set_seq_nr(self,seq_nr_in):
        self.seq_nr = seq_nr_in-1
        try:
            self.update_frame_spin()
        except:
            print('could not update frame spin')

    def set_body_thresh(self,body_thresh_in):
        self.body_thresh = body_thresh_in

    def load_camera_calibration(self,calib_fldr,c_params,c2w_matrices,w2c_matrices):
        self.calib_fldr = calib_fldr
        self.c_params = c_params
        self.c2w_matrices = c2w_matrices
        self.w2c_matrices = w2c_matrices

    def set_seq_spin(self,seq_spin_in):
        self.seq_spin = seq_spin_in
        self.seq_spin.valueChanged.connect(self.set_seq_nr)

    def set_frame_spin(self,frame_spin_in):
        self.frame_spin = frame_spin_in
        self.frame_spin.valueChanged.connect(self.load_frame)

    def set_frame_slider(self,frame_slider_in):
        self.frame_slider = frame_slider_in
        self.frame_slider.valueChanged.connect(self.load_frame)

    def set_img_seg_view_btns(self,img_view_btn_in,seg_view_btn_in):
        self.img_view_btn = img_view_btn_in
        self.seg_view_btn = seg_view_btn_in
        self.img_view_btn.setChecked(True)
        self.seg_view_btn.setChecked(False)

    def set_min_seq_length(self,min_seq_length_in):
        self.min_seq_length = min_seq_length_in

    def sort_keys(self):
        h5_keys = list(self.seq_file.keys())
        self.seq_keys = [None]*self.N_mov
        first_frame = False
        for i,key in enumerate(h5_keys):
            key_split = key.split('_')
            if key_split[0] != 'traces' and key_split[0] != 'srf':
                if len(key_split) == 7:
                    mov_i = int(key_split[2])-1
                    if self.seq_keys[mov_i] == None:
                        self.seq_keys[mov_i] = [None]
                    seq_i = int(key_split[6])-1
                    frame_i = int(key_split[4])
                    N_seq_i = len(self.seq_keys[mov_i])
                    if (seq_i+1)>N_seq_i:
                        for j in range(N_seq_i,seq_i+1):
                            self.seq_keys[mov_i].append(None)
                    if self.seq_keys[mov_i][seq_i] == None:
                        self.seq_keys[mov_i][seq_i] = [frame_i]
                    else:
                        self.seq_keys[mov_i][seq_i].append(frame_i)
                    if not first_frame:
                        first_frame = True
                        self.mov_nr = mov_i
                        self.seq_nr = seq_i
                        self.add_frame(frame_i)
        #for i in range(self.N_mov):
        #    for j in range(len(self.seq_keys)):
        #        if len(self.seq_keys[i][j])<self.min_seq_length:
        #            self.seq_keys[i][j] = None

    def update_seq_spin(self):
        self.seq_id_list = []
        self.seq_frames_list = []
        try:
            N_seqs = 0
            for i,key in enumerate(self.seq_keys[self.mov_nr]):
                if key != None:
                    self.seq_frames_list = self.seq_keys[self.mov_nr][i]
                    self.seq_frames_list.sort()
                    if len(self.seq_frames_list)>=self.min_seq_length:
                        N_seqs += 1
                        self.seq_id_list.append(i)
            if N_seqs>0:
                self.seq_spin.setMinimum(1)
                self.seq_spin.setMaximum(N_seqs)
                self.seq_spin.setValue(1)
                self.set_seq_nr(1)
            else:
                # No sequences:
                self.seq_spin.setMinimum(0)
                self.seq_spin.setMaximum(0)
                self.seq_spin.setValue(0)
                self.set_seq_nr(0)
        except:
            # No sequences:
            self.seq_spin.setMinimum(0)
            self.seq_spin.setMaximum(0)
            self.seq_spin.setValue(0)
            self.set_seq_nr(0)

    def update_frame_spin(self):
        if len(self.seq_id_list)>0:
            self.seq_frames_list = self.seq_keys[self.mov_nr][self.seq_id_list[self.seq_nr]]
            self.seq_frames_list.sort()
            self.frame_spin.setMinimum(1)
            self.frame_spin.setMaximum(len(self.seq_frames_list))
            self.frame_spin.setValue(1)
            self.frame_slider.setMinimum(1)
            self.frame_slider.setMaximum(len(self.seq_frames_list))
            self.frame_slider.setValue(1)
            self.load_frame(1)
        else:
            self.frame_spin.setMinimum(0)
            self.frame_spin.setMaximum(0)
            self.frame_spin.setValue(0)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(0)
            self.frame_slider.setValue(0)
            self.seq_frames_list = []

    def add_frame(self,frame_nr_in):
        for n in range(self.N_cam):
            frame_key_n = 'frame_m_'+str(self.mov_nr+1)+'_f_'+str(frame_nr_in)+'_s_'+str(self.seq_nr+1)+'_c_'+str(n+1)
            img_n = np.transpose(np.copy(self.seq_file[frame_key_n]))
            self.v_list.append(self.w_sub.addViewBox(row=1,col=n,lockAspect=True))
            self.img_list.append(pg.ImageItem(img_n))
            self.v_list[n].addItem(self.img_list[n])
            self.v_list[n].disableAutoRange('xy')
            self.v_list[n].autoRange()

    '''
    def load_frame(self,frame_nr_in):
        self.frame_nr = frame_nr_in-1
        self.frame_spin.setValue(frame_nr_in)
        self.frame_slider.setValue(frame_nr_in)
        com_key = 'com_m_'+str(self.mov_nr+1)+'_f_'+str(self.seq_frames_list[self.frame_nr])+'_s_'+str(self.seq_id_list[self.seq_nr]+1)
        self.com = np.copy(self.seq_file[com_key])
        self.body_masks = []
        self.wing_masks = []
        imgs = []
        for n in range(self.N_cam):
            frame_key_n = 'frame_m_'+str(self.mov_nr+1)+'_f_'+str(self.seq_frames_list[self.frame_nr])+'_s_'+str(self.seq_id_list[self.seq_nr]+1)+'_c_'+str(n+1)
            img_n = np.transpose(np.copy(self.seq_file[frame_key_n]))
            self.window_size = img_n.shape
            imgs.append(img_n)
            fly_img,color,body_mask,wing_mask = self.find_contours(img_n)
            self.body_masks.append(body_mask)
            self.wing_masks.append(wing_mask)
            if self.img_view_btn.isChecked() == True:
                self.img_list[n].setImage(fly_img)
            elif self.seg_view_btn.isChecked() == True:
                self.img_list[n].setImage(color)
            else:
                print('invalid view')
        try:
            self.mdl.set_COM(self.com)
            imgs_concat = np.stack(imgs,axis=2)
            self.mdl.set_frame(imgs_concat,self.mov_nr,self.seq_nr,self.frame_nr)
            self.mdl.set_masks(self.body_masks,self.wing_masks)
        except:
            pass
    '''

    def load_frame(self,frame_nr_in):
        self.frame_nr = frame_nr_in-1
        self.frame_spin.setValue(frame_nr_in)
        self.frame_slider.setValue(frame_nr_in)
        self.batch_load()
        self.com = self.COM_batch[:,self.batch_index]
        self.body_masks = []
        self.wing_masks = []
        imgs = []
        zero_img = np.zeros((self.window_size[0],self.window_size[1]),dtype=np.uint8)
        for n in range(self.N_cam):
            img_n = self.frames_batch[:,:,n,self.batch_index]
            fly_img = img_n.astype(np.uint8)
            body_mask = self.body_mask_batch[:,:,n,self.batch_index]
            body_mask.astype(np.uint8)
            wing_mask = self.wing_mask_batch[:,:,n,self.batch_index]
            wing_mask.astype(np.uint8)
            color = cv2.cvtColor(zero_img,cv2.COLOR_GRAY2BGR)
            color[np.where(wing_mask==255)] = [0,0,255]
            color[np.where(body_mask==255)] = [255,0,0]
            self.body_masks.append(body_mask)
            self.wing_masks.append(wing_mask)
            if self.img_view_btn.isChecked() == True:
                self.img_list[n].setImage(fly_img)
            elif self.seg_view_btn.isChecked() == True:
                self.img_list[n].setImage(color)
            else:
                print('invalid view')
        # try to set batch index model:
        try:
            self.mdl.set_batch_index(self.batch_index,self.frame_nr)
        except:
            pass

    def find_contours(self,frame_in):
        img = frame_in.astype(np.uint8)
        blur = cv2.GaussianBlur(img,(5,5),0)
        median_thresh = int(np.median(blur)-20)
        ret,thresh = cv2.threshold(blur,median_thresh,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)       # set kernel as 3x3 matrix from numpy
        mask_inv = cv2.erode(cv2.bitwise_not(thresh),kernel,iterations=1)
        wing_mask = mask_inv
        fly_img = cv2.fastNlMeansDenoising(mask_inv*img,None,5,7,21)
        # detect body
        ret,body_mask = cv2.threshold(fly_img,self.body_thresh-20,255,cv2.THRESH_BINARY)
        #edges = cv2.Canny(fly_img,50,self.body_thresh)
        color = cv2.cvtColor(fly_img,cv2.COLOR_GRAY2BGR)
        color[np.where(wing_mask==255)] = [0,0,255]
        color[np.where(body_mask==255)] = [255,0,0]
        return img,color,body_mask,wing_mask

    def set_model(self,mdl_in):
        self.mdl = mdl_in
        self.batch_load()
        self.mdl.set_batch(self.COM_batch,self.frames_batch,self.body_mask_batch,self.wing_mask_batch,self.mov_nr,self.seq_nr,self.batch_start,self.batch_end,self.batch_index)
        self.mdl.set_batch_index(self.batch_index,self.frame_nr)
        self.mdl.set_batch_loader(self.load_frame_batch_2)
        self.mdl.set_batch_size(self.batch_size)

    def batch_load(self):
        # determine batch nr
        batch_nr = math.floor((self.frame_nr-self.start_frame)/(self.batch_size*1.0))
        center_frame = self.start_frame+batch_nr*self.batch_size+int(self.batch_size/2.0)
        self.batch_index = self.frame_nr-self.start_frame-batch_nr*self.batch_size
        if self.batch_mov_nr != self.mov_nr or self.batch_seq_nr != self.seq_nr or self.batch_frame_nr != center_frame:
            f_start,f_end,img_batch,com_batch,body_mask_btch,wing_mask_btch = self.load_frame_batch(center_frame)
            self.batch_start = f_start
            self.batch_end = f_end
            self.body_mask_batch = body_mask_btch
            self.wing_mask_batch = wing_mask_btch
            self.frames_batch      = img_batch
            self.COM_batch          = com_batch
            # Update batch indices
            self.batch_mov_nr      = self.mov_nr
            self.batch_seq_nr      = self.seq_nr
            self.batch_frame_nr  = center_frame
            try:
                self.mdl.set_batch(self.COM_batch,self.frames_batch,self.body_mask_batch,self.wing_mask_batch,self.mov_nr,self.seq_nr,self.batch_start,self.batch_end,self.batch_index)
            except:
                pass

    def load_frame_batch(self,center_frame):
        f_start = center_frame-int(self.batch_size/2.0)
        f_end   = center_frame+int(self.batch_size/2.0)
        if f_start<self.start_frame:
            f_start = self.start_frame
            f_end = self.start_frame+self.batch_size
        elif f_end>self.end_frame:
            f_start = self.end_frame-self.batch_size
            f_end = self.end_frame
        img_batch = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,self.batch_size))
        com_batch = np.zeros((4,self.batch_size),dtype=np.float64)
        for i,f_nr in enumerate(range(f_start,f_end)):
            com_key = 'com_m_'+str(self.mov_nr+1)+'_f_'+str(self.seq_frames_list[f_nr])+'_s_'+str(self.seq_id_list[self.seq_nr]+1)
            com_batch[:,i] = np.copy(self.seq_file[com_key])
            for n in range(self.N_cam):
                frame_key_n = 'frame_m_'+str(self.mov_nr+1)+'_f_'+str(self.seq_frames_list[f_nr])+'_s_'+str(self.seq_id_list[self.seq_nr]+1)+'_c_'+str(n+1)
                img_batch[:,:,n,i] = np.transpose(np.copy(self.seq_file[frame_key_n]))
        # Compute min img:
        img_min = self.compute_min_image(img_batch)
        # Subtract from img_batch
        img_sub = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,self.batch_size))
        img_min_batch = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,self.batch_size))
        body_mask_btch = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,self.batch_size))
        wing_mask_btch = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,self.batch_size))
        n_A = int(self.window_size[0]/6)
        n_B = self.window_size[0]-n_A
        n_C = int(self.window_size[1]/6)
        n_D = self.window_size[1]-n_C
        for n in range(self.N_cam):
            img_min_tiled = np.tile(img_min[n],self.batch_size)
            img_01 = np.clip(1.0-(img_batch[:,:,n,:]/255.0),0.0,1.0)
            img_sub = np.clip(img_01-img_min_tiled,0.0,1.0)
            body_mask_n = (img_min_tiled>((self.body_thresh-20)/255.0))
            # set outer pixels to zero:
            body_mask_n[:n_A,:] = 0
            body_mask_n[n_B:,:] = 0
            body_mask_n[:,:n_C] = 0
            body_mask_n[:,n_D:] = 0
            body_mask_btch[:,:,n,:] = body_mask_n*255
            wing_mask_btch[:,:,n,:] = (img_sub[:,:,:]>0.1)*255
        return f_start,f_end,img_batch,com_batch,body_mask_btch,wing_mask_btch

    def load_frame_batch_2(self,mov_nr_in,seq_nr_in,f_start,f_end):
        n_batch = f_end-f_start
        img_batch = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,n_batch))
        com_batch = np.zeros((4,n_batch),dtype=np.float64)
        for i,f_nr in enumerate(range(f_start,f_end)):
            com_key = 'com_m_'+str(mov_nr_in+1)+'_f_'+str(f_nr)+'_s_'+str(seq_nr_in+1)
            com_batch[:,i] = np.copy(self.seq_file[com_key])
            for n in range(self.N_cam):
                frame_key_n = 'frame_m_'+str(mov_nr_in+1)+'_f_'+str(f_nr)+'_s_'+str(seq_nr_in+1)+'_c_'+str(n+1)
                img_batch[:,:,n,i] = np.transpose(np.copy(self.seq_file[frame_key_n]))
        # Compute min img:
        img_min = self.compute_min_image(img_batch)
        # Subtract from img_batch
        img_sub = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,n_batch))
        img_min_batch = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,n_batch))
        body_mask_btch = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,n_batch))
        wing_mask_btch = np.zeros((self.window_size[0],self.window_size[1],self.N_cam,n_batch))
        n_A = int(self.window_size[0]/6)
        n_B = self.window_size[0]-n_A
        n_C = int(self.window_size[1]/6)
        n_D = self.window_size[1]-n_C
        for n in range(self.N_cam):
            img_min_tiled = np.tile(img_min[n],n_batch)
            img_01 = np.clip(1.0-(img_batch[:,:,n,:]/255.0),0.0,1.0)
            img_sub = np.clip(img_01-img_min_tiled,0.0,1.0)
            body_mask_n = (img_min_tiled>((self.body_thresh-20)/255.0))
            # set outer pixels to zero:
            body_mask_n[:n_A,:] = 0
            body_mask_n[n_B:,:] = 0
            body_mask_n[:,:n_C] = 0
            body_mask_n[:,n_D:] = 0
            body_mask_btch[:,:,n,:] = body_mask_n*255
            wing_mask_btch[:,:,n,:] = (img_sub[:,:,:]>0.1)*255
        return img_batch,com_batch,body_mask_btch,wing_mask_btch

    def compute_min_image(self,img_batch):
        img_01 = np.clip(1.0-(img_batch/255.0),0.0,1.0)
        img_min = []
        for n in range(self.N_cam):
            img_min.append(np.clip(np.min(img_01[:,:,n,:],axis=-1,keepdims=True),0.0,1.0))
        return img_min

    def set_batch_size_spin(self,spin_in):
        self.batch_size_spin = spin_in
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(1000)
        self.batch_size_spin.setValue(100)
        self.batch_size = 100
        self.batch_size_spin.valueChanged.connect(self.set_batch_size)

    def set_batch_size(self,val_in):
        self.batch_size = val_in
        try:
            self.mdl.set_batch_size(self.batch_size)
        except:
            pass
