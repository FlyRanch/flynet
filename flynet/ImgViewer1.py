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
import math
import copy
import time
import json
import h5py
import cv2
from scipy.optimize import least_squares
from itertools import combinations, product

import pySciCam
from pySciCam.pySciCam import ImageSequence
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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

#class ImgViewer1(pg.GraphicsWindow):
class ImgViewer1(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None):
        #pg.GraphicsWindow.__init__(self)
        pg.GraphicsLayoutWidget.__init__(self)
        self.setParent(parent)

        self.w_sub = self.addLayout(row=0,col=0)

        self.v_list = []
        self.img_list = []
        self.frame_list = []

        # Parameters:
        self.frame_nr = 0
        self.N_cam = 3
        self.frame_name = 'frame_'
        self.trig_modes = ['start','center','end']
        self.graph_list = []
        self.window_size = [256,256]
        self.crop_graph_list = []
        self.crop_cntr_xyz = np.array([[0.0],[0.0],[0.0],[1.0]])
        self.crop_window = [224,224]
        self.image_centers = [[128.0,128.0],[128.0,128.0],[128.0,128.0]]
        self.window_outline = []
        self.cropped_imgs = []
        self.crop_window_clrs = [(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
        self.c_type = 0
        self.manual_mode = False
        self.automatic_mode = True
        self.fast_mode = False
        self.mask_ROIS = []
        self.mask_positions = []
        self.mask_polys = []
        self.mask_polys_grey = []

    def set_session_folder(self,session_folder,session_name):
        self.session_folder = session_folder
        self.session_name = session_name

    def set_N_cam(self,N_cam):
        self.N_cam = N_cam

    def set_trigger_mode(self,trigger_mode,trigger_frame):
        self.trigger_mode = trigger_mode
        self.trigger_frame = trigger_frame

    def set_cam_folders(self,cam_folders_in):
        self.cam_folders = cam_folders_in

    def set_mov_folders(self,mov_folders_in):
        self.mov_folders = mov_folders_in
        self.N_mov = len(mov_folders_in)

    def set_img_format(self,img_format_in):
        self.img_format = img_format_in

    def set_start_frame(self,start_frame_in,end_frame_in):
        self.start_frame = start_frame_in
        self.end_frame = end_frame_in

    def set_movie_nr(self,mov_nr):
        self.mov_nr = mov_nr-1
        self.mov_folder = self.mov_folders[self.mov_nr]
        self.calculate_median_frame()

    def load_bckg_frames(self,bckg_path,bckg_frames,bckg_img_format):
        self.bckg_path = bckg_path
        self.bckg_frames = bckg_frames
        self.bckg_img_format = bckg_img_format
        self.bckg_imgs = []
        for bckg_frame in self.bckg_frames:
            os.chdir(self.bckg_path)
            img_cv = cv2.imread(bckg_frame+self.bckg_img_format,0)
            self.bckg_imgs.append(img_cv)

    def load_camera_calibration(self,c_params,c_type):
        self.c_type = c_type
        self.c_params = c_params
        self.world2cam = []
        self.w2c_scaling = []
        self.uv_shift = []
        if self.c_type==0:
            for n in range(self.N_cam):
                w2c_mat = np.array([[c_params[0,n],c_params[1,n],c_params[2,n],c_params[3,n]],
                    [c_params[4,n],c_params[5,n],c_params[6,n],c_params[7,n]],
                    [0.0,0.0,0.0,1.0]])
                scale_mat = np.array([[c_params[8,n],c_params[9,n],c_params[10,n],1.0],
                    [c_params[8,n],c_params[9,n],c_params[10,n],1.0],
                    [0.0,0.0,0.0,1.0]])
                self.world2cam.append(w2c_mat)
                self.w2c_scaling.append(scale_mat)
        elif self.c_type==1:
            for n in range(self.N_cam):
                C = np.array([[self.c_params[0,n], self.c_params[2,n], 0.0, 0.0],
                    [0.0, self.c_params[1,n], 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
                q0 = self.c_params[3,n]
                q1 = self.c_params[4,n]
                q2 = self.c_params[5,n]
                q3 = self.c_params[6,n]
                R = np.array([[2.0*pow(q0,2)-1.0+2.0*pow(q1,2), 2.0*q1*q2+2.0*q0*q3,  2.0*q1*q3-2.0*q0*q2],
                    [2.0*q1*q2-2.0*q0*q3, 2.0*pow(q0,2)-1.0+2.0*pow(q2,2), 2.0*q2*q3+2.0*q0*q1],
                    [2.0*q1*q3+2.0*q0*q2, 2.0*q2*q3-2.0*q0*q1, 2.0*pow(q0,2)-1.0+2.0*pow(q3,2)]])
                T = np.array([self.c_params[7,n],self.c_params[8,n],self.c_params[9,n]])
                #T = np.array([0.0,0.0,0.0])
                K = np.array([[R[0,0], R[0,1], R[0,2], T[0]],
                    [R[1,0], R[1,1], R[1,2], T[1]],
                    [R[2,0], R[2,1], R[2,2], T[2]],
                    [0.0, 0.0, 0.0, 1.0]])
                W2C_mat = np.dot(C,K)
                uv_corr = np.array([self.c_params[14,n],self.c_params[15,n]])
                self.uv_shift.append(uv_corr)
                W2C_mat[0,3] = W2C_mat[0,3]-self.c_params[11,n]/2.0+self.c_params[13,n]/2.0+uv_corr[0]
                W2C_mat[1,3] = W2C_mat[1,3]-self.c_params[10,n]/2.0+self.c_params[12,n]/2.0+uv_corr[1]
                self.world2cam.append(W2C_mat)
                scale_mat = np.array([[0.0,0.0,0.0,1.0],
                    [0.0,0.0,0.0,1.0],
                    [0.0,0.0,0.0,1.0]])
                self.w2c_scaling.append(scale_mat)

    def project_2_uv(self,xyz_pts,cam_nr):
        if self.c_type==0:
            uv_pts = np.divide(np.dot(self.world2cam[cam_nr],xyz_pts),np.dot(self.w2c_scaling[cam_nr],xyz_pts))
        elif self.c_type==1:
            uv_pts = np.dot(self.world2cam[cam_nr],xyz_pts)
        return uv_pts

    def tri_function(self,x_in,b_in):
        uv_out = np.zeros(self.N_cam*2)
        xyz_vec = np.ones((4,1))
        xyz_vec[:3,0] = x_in
        for n in range(self.N_cam):
            uv_n = self.project_2_uv(xyz_vec,n)
            uv_out[(n*2):(n*2+2)] = uv_n[:2,0]
        res = b_in-uv_out
        return res

    def triangulation(self,uv_list_in):
        x_0 = np.array([0.0,0.0,0.0])
        # test projection
        b = np.zeros(self.N_cam*2)
        for n in range(self.N_cam):
            b[n*2] = uv_list_in[n][0]
            b[n*2+1] = uv_list_in[n][1]
        #print(self.tri_function(x_0,b))
        res = least_squares(self.tri_function,x_0,method='lm',args=(b,))
        coord = np.ones(4)
        coord[:3] = res.x
        # Calculate RMSE by back-projecting to uv coordinates:
        uv_out = np.zeros(self.N_cam*2)
        for n in range(self.N_cam):
            uv_n = self.project_2_uv(coord,n)
            uv_out[n*2] = uv_n[0]
            uv_out[n*2+1] = uv_n[1]
        uv_diff = b-uv_out
        RMSE = np.sqrt(np.mean(np.power(b-uv_out,2)))
        return coord, uv_out, RMSE, uv_diff

    def calculate_median_frame(self):
        self.frame_nr = self.start_frame
        frame_list = self.load_frame()
        N_med_frames = int(np.floor((self.end_frame-self.start_frame)/100))+1
        med_frames = np.zeros((frame_list[0].shape[0],frame_list[0].shape[1],self.N_cam,N_med_frames))
        cntr = 0
        for i in range(self.start_frame,self.end_frame,100):
            self.frame_nr = i
            frame_list = self.load_frame()
            for n in range(self.N_cam):
                blur = cv2.blur(frame_list[n],(7,7))
                med_frames[:,:,n,cntr] = blur
            cntr += 1
        self.median_frames = np.zeros((frame_list[0].shape[0],frame_list[0].shape[1],self.N_cam))
        for n in range(self.N_cam):
            self.median_frames[:,:,n] = np.median(med_frames[:,:,n,:],axis=2)

    def load_frame(self):
        frame_list = []
        if self.img_format == '.mraw':
            if self.trigger_mode == 'start':
                frame_ind = self.frame_nr
            elif self.trigger_mode == 'center':
                frame_ind = self.frame_nr
            elif self.trigger_mode == 'end':
                frame_ind = self.frame_nr
            else:
                print('error: invalid trigger mode')
            for i in range(self.N_cam):
                os.chdir(self.session_folder+'/'+self.cam_folders[i]+self.mov_folder)
                file_name = str(self.cam_folders[i]+self.mov_folder+'.mraw')
                raw_type = 'photron_mraw_mono_8bit'
                text_trap = io.StringIO()
                sys.stdout = text_trap
                data = ImageSequence(file_name,rawtype=raw_type,height=256,width=256,frames=(frame_ind,frame_ind+1))
                sys.stdout = sys.__stdout__
                img_data = np.asarray(data.arr)
                img_min = np.amin(img_data[0,:,:])
                img_max = np.amax(img_data[0,:,:])
                data = ((img_data[0,:,:]-img_min)/(img_max-img_min))*255
                img_scaled = data.astype(np.uint8)
                frame_list.append(img_scaled)
        elif self.img_format == '.bmp':
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
            for i in range(self.N_cam):
                os.chdir(self.session_folder+'/'+self.mov_folder+'/'+self.cam_folders[i])
                img_cv = cv2.imread(self.frame_name + str(frame_ind) +'.bmp',0)
                frame_list.append(img_cv/255.0)
        elif self.img_format == '.tif':
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
            for i in range(self.N_cam):
                os.chdir(self.session_folder+'/'+self.cam_folders[i]+self.mov_folder)
                frame_str = self.cam_folders[i]+self.mov_folder+f'{frame_ind:06}'+'.tif'
                img_cv = plt.imread(frame_str)
                img_min = np.amin(img_cv)
                img_max = np.amax(img_cv)
                data = ((img_cv-img_min)/((img_max-img_min)*1.0))*255
                frame_list.append(data)
                if len(self.window_size) == i:
                    self.window_size.append([data.shape[1],data.shape[0]])
        else:
            print('error: unknown image format')
        return frame_list

    def add_frame(self,start_frame_in,end_frame_in):
        self.start_frame = start_frame_in
        self.end_frame = end_frame_in
        self.frame_nr = start_frame_in
        self.N_frames = self.end_frame-self.start_frame
        frame_list = self.load_frame()
        self.frame_shape = frame_list[0].shape
        for i, frame in enumerate(frame_list):
            self.frame_list.append(frame)
            self.v_list.append(self.w_sub.addViewBox(row=1,col=i,lockAspect=True))
            frame_in = np.transpose(np.flipud(frame))
            self.img_list.append(pg.ImageItem(frame_in))
            self.v_list[i].addItem(self.img_list[i])
            self.v_list[i].disableAutoRange('xy')
            self.v_list[i].autoRange()

    def update_frame(self,frame_nr):
        self.frame_nr = frame_nr
        frame_list = self.load_frame()
        self.fast_mode = False
        if len(frame_list)==self.N_cam:
            if self.automatic_mode:
                img_contour_list = []
                COM_all_views = []
                N_COM = 0
                for n, frame in enumerate(frame_list):
                    frame_in = np.transpose(np.flipud(frame))
                    img_contour, fly_contours = self.find_contours(frame_in)
                    COM_list = self.find_COM(fly_contours,n)
                    N_COM_n = len(COM_list)
                    if N_COM_n>0:
                        for j in range(N_COM_n):
                            COM_all_views.append(COM_list[j])
                        N_COM += N_COM_n
                    img_contour_list.append(img_contour)
                # Compute combinations:
                comb_list = list(combinations(COM_all_views,3))
                self.fly_COM_coord = []
                self.fly_COM_uv = []
                id_cntr = 0
                for comb in comb_list:
                    uv_com = [None,None]*self.N_cam
                    cam_sum = 0
                    cam_views = np.zeros(self.N_cam)
                    for n in range(self.N_cam):
                        uv_com[comb[n][2]] = [comb[n][1]*1.0,self.frame_shape[0]-comb[n][0]*1.0]
                        cam_views[comb[n][2]] = 1
                    if np.sum(cam_views) == self.N_cam:
                        # triangulate points:
                        xyz_com, uv_xyz, RMSE, uv_diff = self.triangulation(uv_com)
                        if RMSE<4.0:
                            self.fly_COM_coord.append(xyz_com)
                            self.fly_COM_uv.append(uv_xyz)
                            for n in range(self.N_cam):
                                cv2.circle(img_contour_list[n],(int(self.frame_shape[0]-uv_xyz[n*2+1]),int(uv_xyz[n*2])),1,self.crop_window_clrs[id_cntr],-1)
                            id_cntr += 1
                if id_cntr>0:
                    # Set Crop windows
                    # apply mask on greyscale image
                    frame_list= self.update_mask_grey_image(frame_list)
                    self.update_crop_window(frame_list,img_contour_list)
                img_contour_list = self.update_mask_window(img_contour_list)
                for n in range(self.N_cam):
                    self.img_list[n].setImage(img_contour_list[n])
            elif self.manual_mode:
                img_contour_list = []
                COM_all_views = []
                N_COM = 0
                uv_xyz = np.zeros(self.N_cam*2)
                for n, frame in enumerate(frame_list):
                    frame_in = np.transpose(np.flipud(frame))
                    img_contour, fly_contours = self.find_contours(frame_in)
                    img_contour_list.append(img_contour)
                    uv_n = self.project_2_uv(self.crop_cntr_xyz,n)
                    uv_xyz[n*2] = uv_n[0]
                    uv_xyz[n*2+1] = uv_n[1]
                self.fly_COM_coord = []
                self.fly_COM_uv = []
                id_cntr=0
                #if RMSE<4.0:
                self.fly_COM_coord.append(self.crop_cntr_xyz)
                self.fly_COM_uv.append(uv_xyz)
                for n in range(self.N_cam):
                    cv2.circle(img_contour_list[n],(int(self.frame_shape[0]-uv_xyz[n*2+1]),int(uv_xyz[n*2])),1,self.crop_window_clrs[id_cntr],-1)
                #if id_cntr>0:
                # apply mask on greyscale image
                frame_list= self.update_mask_grey_image(frame_list)
                self.update_crop_window(frame_list,img_contour_list)
                # Color masked parts of the image blue:
                img_contour_list = self.update_mask_window(img_contour_list)
                for n in range(self.N_cam):
                    self.img_list[n].setImage(img_contour_list[n])
        self.frame_spin.setValue(self.frame_nr)
        self.frame_slider.setValue(self.frame_nr)

    def update_frame_fast(self,frame_nr):
        self.frame_nr = frame_nr
        frame_list = self.load_frame()
        self.fast_mode = True
        if len(frame_list)==self.N_cam:
            if self.automatic_mode:
                img_contour_list = []
                COM_all_views = []
                N_COM = 0
                for n, frame in enumerate(frame_list):
                    frame_in = np.transpose(np.flipud(frame))
                    img_contour, fly_contours = self.find_contours(frame_in)
                    COM_list = self.find_COM(fly_contours,n)
                    N_COM_n = len(COM_list)
                    if N_COM_n>0:
                        for j in range(N_COM_n):
                            COM_all_views.append(COM_list[j])
                        N_COM += N_COM_n
                    img_contour_list.append(img_contour)
                # Compute combinations:
                comb_list = list(combinations(COM_all_views,3))
                self.fly_COM_coord = []
                self.fly_COM_uv = []
                id_cntr = 0
                for comb in comb_list:
                    uv_com = [None,None]*self.N_cam
                    cam_sum = 0
                    cam_views = np.zeros(self.N_cam)
                    for n in range(self.N_cam):
                        uv_com[comb[n][2]] = [comb[n][1]*1.0,self.frame_shape[0]-comb[n][0]*1.0]
                        cam_views[comb[n][2]] = 1
                    if np.sum(cam_views) == self.N_cam:
                        # triangulate points:
                        xyz_com, uv_xyz, RMSE, uv_diff = self.triangulation(uv_com)
                        if RMSE<4.0:
                            self.fly_COM_coord.append(xyz_com)
                            self.fly_COM_uv.append(uv_xyz)
                            #for n in range(self.N_cam):
                            #    cv2.circle(img_contour_list[n],(int(self.frame_shape[0]-uv_xyz[n*2+1]),int(uv_xyz[n*2])),1,self.crop_window_clrs[id_cntr],-1)
                            id_cntr += 1
                if id_cntr>0:
                    # Set Crop windows
                    # apply mask on greyscale image
                    frame_list= self.update_mask_grey_image(frame_list)
                    self.update_crop_window(frame_list,img_contour_list)
            elif self.manual_mode:
                img_contour_list = []
                COM_all_views = []
                N_COM = 0
                uv_xyz = np.zeros(self.N_cam*2)
                for n, frame in enumerate(frame_list):
                    frame_in = np.transpose(np.flipud(frame))
                    uv_n = self.project_2_uv(self.crop_cntr_xyz,n)
                    uv_xyz[n*2] = uv_n[0]
                    uv_xyz[n*2+1] = uv_n[1]
                self.fly_COM_coord = []
                self.fly_COM_uv = []
                id_cntr=0
                self.fly_COM_coord.append(np.squeeze(self.crop_cntr_xyz))
                self.fly_COM_uv.append(uv_xyz)
                # apply mask on greyscale image
                frame_list= self.update_mask_grey_image(frame_list)
                self.update_crop_window(frame_list,img_contour_list)

    def set_frame_spin(self,frame_spin_in):
        self.frame_spin = frame_spin_in

    def set_frame_slider(self,frame_slider_in):
        self.frame_slider = frame_slider_in

    def set_body_thresh(self,body_thresh_in):
        self.body_thresh = body_thresh_in

    def set_min_seq_len(self,min_len_in):
        self.min_seq_len = min_len_in

    def set_max_COM_dist(self,max_COM_dist_in):
        self.max_COM_dist = max_COM_dist_in

    def update_crop_window(self,frame_list_in,contour_imgs):
        self.cropped_imgs = []
        for k in range(len(self.fly_COM_coord)):
            imgs_k = []
            for n in range(self.N_cam):
                i_min = int(self.frame_shape[0]-int(self.fly_COM_uv[k][n*2+1])-self.crop_window[1]/2.0)
                j_min = int(int(self.fly_COM_uv[k][n*2])-self.crop_window[0]/2.0)
                i_max = int(self.frame_shape[0]-int(self.fly_COM_uv[k][n*2+1])+self.crop_window[1]/2.0)
                j_max = int(int(self.fly_COM_uv[k][n*2])+self.crop_window[0]/2.0)
                if self.fast_mode==False:
                    cv2.rectangle(contour_imgs[n],(i_min,j_min),(i_max,j_max),self.crop_window_clrs[k],1) 
                img_kn = np.zeros((self.crop_window[0],self.crop_window[1]))
                img_med = np.zeros((self.crop_window[0],self.crop_window[1]))
                frame_n = np.flipud(frame_list_in[n])
                med_img_n = np.flipud(self.median_frames[:,:,n])
                if i_min<0:
                    if j_min<0:
                        img_kn[-i_max:,-j_max:] = frame_n[0:i_max,0:j_max]
                        img_med[-i_max:,-j_max:] = med_img_n[0:i_max,0:j_max]
                    elif j_max >= self.frame_shape[1]:
                        img_kn[-i_max:,:(self.frame_shape[1]-j_min)] = frame_n[0:i_max,j_min:self.frame_shape[1]]
                        img_med[-i_max:,:(self.frame_shape[1]-j_min)] = med_img_n[0:i_max,j_min:self.frame_shape[1]]
                    else:
                        img_kn[-i_max:,:] = frame_n[0:i_max,j_min:j_max]
                        img_med[-i_max:,:] = med_img_n[0:i_max,j_min:j_max]
                elif i_max >= self.frame_shape[0]:
                    if j_min<0:
                        img_kn[:(self.frame_shape[0]-i_min),-j_max:] = frame_n[i_min:self.frame_shape[0],0:j_max]
                        img_med[:(self.frame_shape[0]-i_min),-j_max:] = med_img_n[i_min:self.frame_shape[0],0:j_max]
                    elif j_max >= self.frame_shape[1]:
                        img_kn[:(self.frame_shape[0]-i_min),:(self.frame_shape[1]-j_min)] = frame_n[i_min:self.frame_shape[0],j_min:self.frame_shape[1]]
                        img_med[:(self.frame_shape[0]-i_min),:(self.frame_shape[1]-j_min)] = med_img_n[i_min:self.frame_shape[0],j_min:self.frame_shape[1]]
                    else:
                        img_kn[:(self.frame_shape[0]-i_min),:] = frame_n[i_min:self.frame_shape[0],j_min:j_max]
                        img_med[:(self.frame_shape[0]-i_min),:] = med_img_n[i_min:self.frame_shape[0],j_min:j_max]
                else:
                    if j_min<0:
                        img_kn[:,-j_max:] = frame_n[i_min:i_max,0:j_max]
                        img_med[:,-j_max:] = med_img_n[i_min:i_max,0:j_max]
                    elif j_max >= self.frame_shape[1]:
                        img_kn[:,:(self.frame_shape[1]-j_min)] = frame_n[i_min:i_max,j_min:self.frame_shape[1]]
                        img_med[:,:(self.frame_shape[1]-j_min)] = med_img_n[i_min:i_max,j_min:self.frame_shape[1]]
                    else:
                        img_kn = frame_n[i_min:i_max,j_min:j_max]
                        img_med = med_img_n[i_min:i_max,j_min:j_max]
                # Background subtraction
                if self.automatic_mode:
                    img_sub = img_kn-img_med
                    img_sub_min = np.amin(img_sub)
                    img_sub_max = np.amax(img_sub)
                    data = ((img_sub-img_sub_min)/(img_sub_max-img_sub_min))*255
                    #data = img_kn
                    img_out = np.copy(data.astype(np.uint8))
                    imgs_k.append(img_out)
                elif self.manual_mode:
                    img_sub = img_kn
                    img_sub_min = np.amin(img_sub)
                    img_sub_max = np.amax(img_sub)
                    data = ((img_sub-img_sub_min)/(img_sub_max-img_sub_min))*255
                    #data = img_kn
                    if data.shape[0] != self.crop_window[0]:
                        print('error data shape[0]: '+str(data.shape[0]))
                    if data.shape[1] != self.crop_window[0]:
                        print('error data shape[1]: '+str(data.shape[1]))
                    img_out = np.copy(data.astype(np.uint8))
                    imgs_k.append(img_out)
            self.cropped_imgs.append(imgs_k)

    def find_contours(self,frame_in): #,median_img):
        img_min = np.amin(frame_in)
        img_max = np.amax(frame_in)
        data = ((frame_in-img_min)/(img_max-img_min))*255
        img = data.astype(np.uint8)
        img_inv = 255-img
        blur = cv2.blur(img_inv,(7,7))
        ret,thresh = cv2.threshold(blur,self.body_thresh,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        color =    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # find contours with limited length:
        fly_contour = []
        for contour in contours:
            L_c = len(contour)
            if L_c>50 and L_c<1000:
                # Check if any of the points is on the edges
                contour_array = np.squeeze(np.asarray(contour))
                cu_min = np.amin(contour_array[:,0])
                cu_max = np.amax(contour_array[:,0])
                cv_min = np.amin(contour_array[:,1])
                cv_max = np.amax(contour_array[:,1])
                if cu_min>0 and cu_max<self.frame_shape[1] and cv_min>0 and cv_max<self.frame_shape[0]:
                    fly_contour.append(contour)
        cv2.drawContours(color,fly_contour,-1,(0,255,0),cv2.FILLED)
        return color,fly_contour

    def find_COM(self,contours_in,cam_nr):
        COM_list = []
        for contour in contours_in:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            COM_list.append([cX,cY,cam_nr])
        return COM_list

    def set_save_fldr(self,save_fldr_in):
        self.save_fldr = save_fldr_in
        self.load_flight_seqs()

    def set_ses_calc_progress(self,progress_in):
        self.save_progress = progress_in
        self.save_progress.setValue(0)

    def calculate_flight_seqs(self):
        print('calculating flight sequences')
        prog_val_prev = 0
        os.chdir(self.save_fldr)
        self.seq_file = h5py.File('seqs_'+str(self.session_name)+'.h5py','w')
        if self.automatic_mode:
            for m in range(self.N_mov):
                print('movie ' + str(m+1))
                self.set_movie_nr(m+1)
                seq_list = []
                seq_cntr = 0
                curr_ids = []
                prev_coms = []
                prev_ids = []
                for i in range(self.N_frames):
                    print(i)
                    self.update_frame_fast(self.start_frame+i)
                    N_flies = len(self.fly_COM_coord)
                    N_prev = len(prev_coms)
                    conn_ids = [-10]*N_flies
                    if N_flies>0:
                        curr_ids = range(N_flies)
                        if N_prev>0:
                            comb_list = list(product(range(N_prev),curr_ids))
                            for j,comb in enumerate(comb_list):
                                dist_comb = np.sqrt(np.sum(np.power(prev_coms[comb[0]]-self.fly_COM_coord[comb[1]],2)))
                                if dist_comb<(self.max_COM_dist*self.ds):
                                    conn_ids[comb[1]] = prev_ids[comb[0]]
                        for k in range(N_flies):
                            if conn_ids[k] >= 0:
                                # existing sequence
                                try:
                                    com_tag = 'com_m_'+str(m+1)+'_f_'+str(i+1)+'_s_'+str(conn_ids[k]+1)
                                    self.seq_file.create_dataset(com_tag,data=self.fly_COM_coord[k])
                                    for n in range(self.N_cam):
                                        frame_tag = 'frame_m_'+str(m+1)+'_f_'+str(i+1)+'_s_'+str(conn_ids[k]+1)+'_c_'+str(n+1)
                                        self.seq_file.create_dataset(frame_tag,data=self.cropped_imgs[k][n])
                                except:
                                    print('could not save:')
                                    print(com_tag)
                            else:
                                # new sequence
                                seq_cntr += 1
                                conn_ids[k] = seq_cntr
                                try:
                                    com_tag = 'com_m_'+str(m+1)+'_f_'+str(i+1)+'_s_'+str(conn_ids[k]+1)
                                    self.seq_file.create_dataset(com_tag,data=self.fly_COM_coord[k])
                                    for n in range(self.N_cam):
                                        frame_tag = 'frame_m_'+str(m+1)+'_f_'+str(i+1)+'_s_'+str(conn_ids[k]+1)+'_c_'+str(n+1)
                                        self.seq_file.create_dataset(frame_tag,data=self.cropped_imgs[k][n])
                                except:
                                    print('could not save:')
                                    print(com_tag)
                        prev_coms = self.fly_COM_coord
                        prev_ids = conn_ids
                    else:
                        prev_coms = []
                        prev_ids = []
                    if int(100.0*(m*self.N_frames+i+1)/(self.N_frames*self.N_mov))>prog_val_prev:
                        prog_val_prev = int(100.0*(m*self.N_frames+i+1)/(self.N_frames*self.N_mov))
                        self.save_progress.setValue(prog_val_prev)
        elif self.manual_mode:
            for m in range(self.N_mov):
                print('movie ' + str(m+1))
                self.set_movie_nr(m+1)
                for i in range(self.N_frames):
                    print(i)
                    self.update_frame_fast(self.start_frame+i)
                    com_tag = 'com_m_'+str(m+1)+'_f_'+str(i+1)+'_s_'+str(1)
                    self.seq_file.create_dataset(com_tag,data=self.fly_COM_coord[0])
                    for n in range(self.N_cam):
                        frame_tag = 'frame_m_'+str(m+1)+'_f_'+str(i+1)+'_s_'+str(1)+'_c_'+str(n+1)
                        self.seq_file.create_dataset(frame_tag,data=self.cropped_imgs[0][n])
                    if int(100.0*(m*self.N_frames+i+1)/(self.N_frames*self.N_mov))>prog_val_prev:
                        prog_val_prev = int(100.0*(m*self.N_frames+i+1)/(self.N_frames*self.N_mov))
                        self.save_progress.setValue(prog_val_prev)
        self.seq_file.close()
        self.load_flight_seqs()
        print('finished')

    def set_flight_seq_disp(self,seq_disp_in):
        self.seq_disp = seq_disp_in

    def load_flight_seqs(self):
        for root, dirs, files in os.walk(self.save_fldr):
            if len(files)>0:
                for file in files:
                    if self.session_name in file:
                        self.seq_file_name = file
                        self.seq_disp.setText(self.seq_file_name)

    def set_free_flight_toggle(self,toggle_in):
        self.free_flight_toggle = toggle_in
        self.free_flight_toggle.setChecked(True)
        self.free_flight_mode = True
        self.free_flight_toggle.toggled.connect(self.free_flight_toggled)

    def free_flight_toggled(self):
        if self.free_flight_toggle.isChecked() == True:
            self.free_flight_mode = True
            self.teth_flight_mode = False

    def set_teth_flight_toggle(self,toggle_in):
        self.teth_flight_toggle = toggle_in
        self.teth_flight_toggle.setChecked(False)
        self.teth_flight_mode = False
        self.teth_flight_toggle.toggled.connect(self.teth_flight_toggled)

    def teth_flight_toggled(self):
        if self.teth_flight_toggle.isChecked() == True:
            self.free_flight_mode = False
            self.teth_flight_mode = True

    def set_Nu_crop_spin(self,spin_in):
        self.Nu_crop_spin = spin_in
        self.Nu_crop_spin.setMinimum(1)
        self.Nu_crop_spin.setMaximum(1024)
        self.Nu_crop_spin.setValue(224)
        self.set_Nu_crop(224)
        self.Nu_crop_spin.valueChanged.connect(self.set_Nu_crop)

    def set_Nu_crop(self,Nu_in):
        self.Nu_crop = Nu_in
        self.crop_window[1] = self.Nu_crop
        try:
            self.update_frame(self.frame_nr)
        except:
            time.sleep(0.001)

    def set_Nv_crop_spin(self,spin_in):
        self.Nv_crop_spin = spin_in
        self.Nv_crop_spin.setMinimum(1)
        self.Nv_crop_spin.setMaximum(1024)
        self.Nv_crop_spin.setValue(224)
        self.set_Nv_crop(224)
        self.Nv_crop_spin.valueChanged.connect(self.set_Nv_crop)

    def set_Nv_crop(self,Nv_in):
        self.Nv_crop = Nv_in
        self.crop_window[0] = self.Nv_crop
        try:
            self.update_frame(self.frame_nr)
        except:
            time.sleep(0.001)

    def set_manual_mode_toggle(self,toggle_in):
        self.manual_toggle = toggle_in
        self.manual_toggle.setChecked(False)
        self.manual_mode = False
        self.manual_toggle.toggled.connect(self.manual_mode_toggled)

    def manual_mode_toggled(self):
        if self.manual_toggle.isChecked() == True:
            self.manual_mode = True
            self.automatic_mode = False

    def set_automatic_mode_toggle(self,toggle_in):
        self.automatic_toggle = toggle_in
        self.automatic_toggle.setChecked(True)
        self.automatic_mode = True
        self.automatic_toggle.toggled.connect(self.automatic_mode_toggled)

    def automatic_mode_toggled(self):
        if self.automatic_toggle.isChecked() == True:
            self.automatic_mode = True
            self.manual_mode = False

    def set_crop_x_spin(self,spin_in):
        self.crop_x_spin = spin_in
        self.crop_x_spin.setMinimum(-10.0)
        self.crop_x_spin.setMaximum(10.0)
        self.crop_x_spin.setDecimals(2)
        self.crop_x_spin.setSingleStep(0.01)
        self.crop_x_spin.setValue(0.0)
        self.set_crop_x(0.0)
        self.crop_x_spin.valueChanged.connect(self.set_crop_x)

    def set_crop_x(self,val_in):
        self.crop_x = val_in
        self.crop_cntr_xyz[0] = self.crop_x
        try:
            self.update_frame(self.frame_nr)
        except:
            time.sleep(0.001)

    def set_crop_y_spin(self,spin_in):
        self.crop_y_spin = spin_in
        self.crop_y_spin.setMinimum(-10.0)
        self.crop_y_spin.setMaximum(10.0)
        self.crop_y_spin.setDecimals(2)
        self.crop_y_spin.setSingleStep(0.01)
        self.crop_y_spin.setValue(0.0)
        self.set_crop_y(0.0)
        self.crop_y_spin.valueChanged.connect(self.set_crop_y)

    def set_crop_y(self,val_in):
        self.crop_y = val_in
        self.crop_cntr_xyz[1] = self.crop_y
        try:
            self.update_frame(self.frame_nr)
        except:
            time.sleep(0.001)

    def set_crop_z_spin(self,spin_in):
        self.crop_z_spin = spin_in
        self.crop_z_spin.setMinimum(-10.0)
        self.crop_z_spin.setMaximum(10.0)
        self.crop_z_spin.setDecimals(2)
        self.crop_z_spin.setSingleStep(0.01)
        self.crop_z_spin.setValue(0.0)
        self.set_crop_z(0.0)
        self.crop_z_spin.valueChanged.connect(self.set_crop_z)

    def set_crop_z(self,val_in):
        self.crop_z = val_in
        self.crop_cntr_xyz[2] = self.crop_z
        try:
            self.update_frame(self.frame_nr)
        except:
            time.sleep(0.001)

    def set_cam_mask_spin(self,spin_in):
        self.cam_mask_spin = spin_in
        self.cam_mask_spin.setMinimum(1)
        self.cam_mask_spin.setMaximum(self.N_cam)
        self.cam_mask_spin.setValue(1)
        self.set_cam_mask_nr(1)
        self.cam_mask_spin.valueChanged.connect(self.set_cam_mask_nr)
        for n in range(self.N_cam):
            self.mask_ROIS.append([])
            self.mask_positions.append([])
            self.mask_polys.append([])
            self.mask_polys_grey.append([])

    def set_cam_mask_nr(self,nr_in):
        self.cam_mask_nr = nr_in-1

    def set_add_mask_btn(self,btn_in):
        self.add_mask_btn = btn_in
        self.add_mask_btn.clicked.connect(self.add_mask)

    def add_mask(self):
        cp_cntr_u = int(self.crop_window[1]/2.0)
        cp_cntr_v = int(self.crop_window[0]/2.0)
        mask_start_pos = [[cp_cntr_u-10,cp_cntr_v-10],[cp_cntr_u-10,cp_cntr_v+10],[cp_cntr_u+10,cp_cntr_v+10],[cp_cntr_u+10,cp_cntr_v-10]]
        self.mask_positions[self.cam_mask_nr].append(mask_start_pos)
        roi_lines_pen = pg.mkPen(color='y', width=1.0)
        roi_handles_pen = pg.mkPen(color='y')
        pr = pg.PolyLineROI(self.mask_positions[self.cam_mask_nr][-1],pen=roi_lines_pen,handlePen=roi_handles_pen,closed=True)
        self.mask_ROIS[self.cam_mask_nr].append(pr)
        self.v_list[self.cam_mask_nr].addItem(self.mask_ROIS[self.cam_mask_nr][-1])
        self.mask_ROIS[self.cam_mask_nr][-1].sigRegionChangeFinished.connect(self.update_poly_masks)

    def set_remove_masks_btn(self,btn_in):
        self.remove_masks_btn = btn_in
        self.remove_masks_btn.clicked.connect(self.remove_masks)

    def remove_masks(self):
        for mask in self.mask_ROIS[self.cam_mask_nr]:
            self.v_list[self.cam_mask_nr].removeItem(mask)
        self.mask_ROIS[self.cam_mask_nr] = []
        self.mask_positions[self.cam_mask_nr] = []
        self.mask_polys[self.cam_mask_nr] = []
        self.mask_polys_grey[self.cam_mask_nr] = []
        self.update_frame(self.frame_nr)

    def update_mask_window(self,img_list_in):
        img_list_out = img_list_in
        for n in range(self.N_cam):
            for i,mask in enumerate(self.mask_ROIS[n]):
                cv2.fillPoly(img_list_out[n], self.mask_polys[n][i],(0,0,255))
        return img_list_out

    def update_mask_grey_image(self,img_list_in):
        img_list_out = img_list_in
        fill_val = 0
        for n in range(self.N_cam):
            fill_val = int(np.amax(img_list_out[n]))
            for i,mask in enumerate(self.mask_ROIS[n]):
                if len(self.mask_polys_grey[n][i]) > 0:
                    cv2.fillPoly(img_list_out[n],self.mask_polys_grey[n][i],[fill_val])
        return img_list_out

    def update_poly_masks(self):
        for n in range(self.N_cam):
            self.mask_polys[n] = []
            self.mask_polys_grey[n] = []
            for i,mask in enumerate(self.mask_ROIS[n]):
                mask_state = mask.getState()
                pt_list = []
                for pt in mask_state['points']:
                    pt_x = pt.x()
                    pt_y = pt.y()
                    pt_list.append([pt_x,pt_y])
                nds = np.array(pt_list)
                nds[:,[0, 1]] = nds[:,[1, 0]]
                nds = np.int32([nds])
                gds = np.array(pt_list)
                gds[:,1] = self.window_size[1]-gds[:,1]
                gds = np.int32([gds])
                self.mask_polys[n].append(nds)
                self.mask_polys_grey[n].append(gds)
        self.update_frame(self.frame_nr)

    def print_mask_positions(self):
        for n in range(self.N_cam):
            print('masks cam ' + str(n+1))
            for i,mask in enumerate(self.mask_ROIS[n]):
                print("mask " + str(i+1))
                mask_state = mask.getState()
                print(mask_state['points'])
                for pt in mask_state['points']:
                    print(pt.x())
                    print(pt.y())
