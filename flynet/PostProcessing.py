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
from scipy.signal import medfilt, find_peaks, savgol_filter
from scipy.stats import zscore
import scipy.special
from sklearn.preprocessing import normalize
from itertools import combinations, product
from sklearn.decomposition import PCA

#sys.path.append('/home/flynet/Documents/FlyNet4/filter/build')
#import Kalman_lib
import flynet_kalman


#sys.path.append('/home/flynet/Documents/FlyNet4/filter')
from .KF_filter import KF
from .EKF_filter import EKF

class PostProcessing():

    def __init__(self):
        self.N_mov = 1
        self.N_seq = 1
        self.N_frames = 1
        self.mov_nr = 0
        self.seq_nr = 0
        self.frame_nr = 0
        self.start_frame = 0
        self.end_frame   = 1
        self.dt = 1.0/15000.0
        self.comp_ind = 0
        self.raw_trace_view =False
        self.raw_pred_trace_view = False
        self.raw_filt_trace_view = False
        self.filt_trace_view = False
        self.stroke_trace_view = False
        self.kalman_comp_ind = 0
        self.q_cov_init = [0.0001,0.0001,0.0001,0.01,0.01]
        self.t_cov_init = [0.0001,0.0001,0.0001,0.01,0.01]
        self.x_cov_init = [0.0001,0.0001,0.0001,0.01,0.01]
        self.q_cov = [0.0001,0.0001,0.0001,0.01,0.01]
        self.t_cov = [0.0001,0.0001,0.0001,0.01,0.01]
        self.x_cov = [0.0001,0.0001,0.0001,0.01,0.01]
        self.N_theta = 20
        self.N_eta = 24
        self.N_phi = 16
        self.N_xi = 20
        self.n_deriv_view = 0
        self.Euler_angles = []
        self.SRF_traces = []
        self.wb_intervals = []
        self.wb_T = []
        self.wb_freq = []
        self.N_const = 2
        self.outlier_frames = []
        self.thorax_corr_batch = 2000
        self.N_components = 5
        self.mdl_components = ['...','head','thorax','abdomen','wing left','wing right']
        # Kalman filter:
        self.EKF_filter = Kalman_lib.Kalman()

    def set_seq_file(self,seq_file_in,seq_name_in):
        self.seq_name = seq_name_in
        self.seq_file = seq_file_in    

    def load_seq_file(self,file_loc,seq_name_in):
        os.chdir(file_loc)
        self.seq_name = seq_name_in
        self.seq_file = h5py.File(self.seq_name,'r+')

    def close_seq_file(self):
        self.seq_file.close()

    def load_data(self):
        self.keys = list(self.seq_file.keys())
        self.batch_keys = [key for key in self.keys if 'batch_fit_' in key]
        batch_codes = []
        for i,batch_key in enumerate(self.batch_keys):
            split_str = batch_key.split('_')
            batch_code = [int(split_str[3]),int(split_str[5]),int(split_str[7]),i]
            batch_codes.append(batch_code)
        batch_code_array = np.array(batch_codes)
        self.N_mov = np.amax(batch_code_array[:,0])
        print('N_mov: ' + str(self.N_mov))
        self.seq_indices = []
        self.frame_nr_data = []
        self.pred_data = []
        self.fit_data = []
        self.com_data = []
        for i in range(self.N_mov):
            # Get number of sequences:
            batch_code_i = batch_code_array[batch_code_array[:,0]==(i+1),:]
            nr_of_seqs = np.amax(batch_code_i[:,1])
            n_batches_seq = []
            frame_nr_seq = []
            pred_data_seq = []
            fit_data_seq = []
            com_data_seq = []
            for j in range(nr_of_seqs):
                batch_code_j = batch_code_i[batch_code_i[:,1]==(j+1),:]
                nr_of_batches = np.amax(batch_code_j[:,2])
                n_batches_seq.append(nr_of_batches)
                # Get frame numbers:
                f_start = np.squeeze(np.copy(self.seq_file['batch_fit_m_'+str(i+1)+'_s_'+str(j+1)+'_b_'+str(1)]['f_start']))
                f_end = np.squeeze(np.copy(self.seq_file['batch_fit_m_'+str(i+1)+'_s_'+str(j+1)+'_b_'+str(nr_of_batches)]['f_end']))
                frame_nr_seq.append([f_start,f_end])
                pred_data_j = None
                fit_data_j = None
                com_data_j = None
                for k in range(nr_of_batches):
                    if k==0:
                        pred_data_j = np.copy(self.seq_file['batch_fit_m_'+str(i+1)+'_s_'+str(j+1)+'_b_'+str(k+1)]['pred_state'])
                        fit_data_j = np.copy(self.seq_file['batch_fit_m_'+str(i+1)+'_s_'+str(j+1)+'_b_'+str(k+1)]['opt_state'])
                        com_data_j = np.copy(self.seq_file['batch_fit_m_'+str(i+1)+'_s_'+str(j+1)+'_b_'+str(k+1)]['com_batch'])
                    else:
                        pred_data_j = np.append(pred_data_j,np.copy(self.seq_file['batch_fit_m_'+str(i+1)+'_s_'+str(j+1)+'_b_'+str(k+1)]['pred_state']),axis=1)
                        fit_data_j = np.append(fit_data_j,np.copy(self.seq_file['batch_fit_m_'+str(i+1)+'_s_'+str(j+1)+'_b_'+str(k+1)]['opt_state']),axis=1)
                        com_data_j = np.append(com_data_j,np.copy(self.seq_file['batch_fit_m_'+str(i+1)+'_s_'+str(j+1)+'_b_'+str(k+1)]['com_batch']),axis=1)
                pred_data_seq.append(pred_data_j)
                fit_data_seq.append(fit_data_j)
                com_data_seq.append(com_data_j)
            self.seq_indices.append(n_batches_seq)
            self.frame_nr_data.append(frame_nr_seq)
            self.pred_data.append(pred_data_seq)
            self.fit_data.append(fit_data_seq)
            self.com_data.append(com_data_seq)

    def continuous_quaternions(self,q_data_in):
        N_data = q_data_in.shape[1]
        q_data_out = q_data_in
        q_mean = np.mean(q_data_out,axis=1)
        for i in range(N_data):
            #q_err_p = np.sum(np.power(q_data_out[1:4,i]-q_mean[1:4],2))
            #q_err_m = np.sum(np.power(-q_data_out[1:4,i]-q_mean[1:4],2))
            q_err_p = np.sum(np.power(q_data_out[0:4,i]-q_mean[0:4],2))
            q_err_m = np.sum(np.power(-q_data_out[0:4,i]-q_mean[0:4],2))
            if q_err_p>q_err_m:
                q_data_out[:,i] = -q_data_out[:,i]
        return q_data_out

    def set_mov_spin(self,mov_spin_in):
        self.mov_spin = mov_spin_in
        self.mov_spin.setMinimum(1)
        self.mov_spin.setMaximum(self.N_mov)
        self.mov_spin.setValue(1)
        #self.set_mov_nr(1)
        self.mov_spin.valueChanged.connect(self.set_mov_nr)

    def set_mov_nr(self,mov_nr_in):
        self.mov_nr = mov_nr_in-1
        self.N_seq = len(self.frame_nr_data[self.mov_nr])
        self.update_seq_spin()

    def set_seq_spin(self,seq_spin_in):
        self.seq_spin = seq_spin_in
        self.seq_spin.setMinimum(1)
        self.seq_spin.setMaximum(self.N_seq)
        self.seq_spin.setValue(1)
        #self.set_seq_nr(1)
        self.seq_spin.valueChanged.connect(self.set_seq_nr)

    def update_seq_spin(self):
        self.seq_spin.setMinimum(1)
        self.seq_spin.setMaximum(self.N_seq)
        self.seq_spin.setValue(1)
        self.set_seq_nr(1)

    def set_seq_nr(self,seq_nr_in):
        self.seq_nr = seq_nr_in-1
        self.N_frames = self.frame_nr_data[self.mov_nr][self.seq_nr][1]-self.frame_nr_data[self.mov_nr][self.seq_nr][0]
        self.update_frame_spin()
        self.update_frame_slider()

    def set_frame_spin(self,frame_spin_in):
        self.frame_spin = frame_spin_in
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(self.N_frames)
        self.frame_spin.setValue(0)
        self.frame_spin.valueChanged.connect(self.set_frame_nr)

    def update_frame_spin(self):
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(self.N_frames)
        self.frame_spin.setValue(0)
        self.set_frame_nr(0)
        self.set_data_q_plot()
        self.set_data_t_plot()
        self.set_data_x_plot()
        self.set_data_cost_plot()

    def set_frame_nr(self,frame_nr_in):
        self.frame_nr = frame_nr_in
        self.frame_spin.setValue(frame_nr_in)
        self.frame_slider.setValue(frame_nr_in)
        self.q_plot.set_time_line(frame_nr_in)
        self.q_plot_zoom.set_time_line(frame_nr_in)
        frame_range = [frame_nr_in-200,frame_nr_in+200]
        self.q_plot_zoom.set_x_range(frame_range)
        self.t_plot.set_time_line(frame_nr_in)
        self.t_plot_zoom.set_time_line(frame_nr_in)
        frame_range = [frame_nr_in-200,frame_nr_in+200]
        self.t_plot_zoom.set_x_range(frame_range)
        self.x_plot.set_time_line(frame_nr_in)
        self.x_plot_zoom.set_time_line(frame_nr_in)
        frame_range = [frame_nr_in-200,frame_nr_in+200]
        self.x_plot_zoom.set_x_range(frame_range)
        self.cost_plot.set_time_line(frame_nr_in)
        self.cost_plot_zoom.set_time_line(frame_nr_in)
        frame_range = [frame_nr_in-200,frame_nr_in+200]
        cost_range = [0.0,2000.0]
        self.cost_plot.set_y_range(cost_range)
        self.cost_plot_zoom.set_x_range(frame_range)
        self.cost_plot_zoom.set_y_range(cost_range)

    def set_frame_slider(self,frame_slider_in):
        self.frame_slider = frame_slider_in
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.N_frames)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.set_frame_nr)

    def update_frame_slider(self):
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.N_frames)
        self.frame_slider.setValue(0)
        self.set_frame_nr(0)

    def set_raw_trace_toggle(self,toggle_in):
        self.raw_trace_toggle = toggle_in
        self.raw_trace_toggle.setChecked(False)
        self.raw_trace_toggle.toggled.connect(self.raw_trace_toggled)

    def raw_trace_toggled(self):
        if self.raw_trace_toggle.isChecked() == True:
            self.raw_trace_view =True
            self.raw_pred_trace_view = False
            self.raw_filt_trace_view = False
            self.filt_trace_view = False
            self.stroke_trace_view = False
            self.update_seq_spin()

    def set_raw_pred_trace_toggle(self,toggle_in):
        self.raw_pred_trace_toggle = toggle_in
        self.raw_pred_trace_toggle.setChecked(False)
        self.raw_pred_trace_toggle.toggled.connect(self.raw_pred_trace_toggled)

    def raw_pred_trace_toggled(self):
        if self.raw_pred_trace_toggle.isChecked() == True:
            self.raw_trace_view =False
            self.raw_pred_trace_view = True
            self.raw_filt_trace_view = False
            self.filt_trace_view = False
            self.stroke_trace_view = False
            self.update_seq_spin()

    def set_raw_filt_trace_toggle(self,toggle_in):
        self.raw_filt_trace_toggle = toggle_in
        self.raw_filt_trace_toggle.setChecked(False)
        self.raw_filt_trace_toggle.toggled.connect(self.raw_filt_trace_toggled)

    def raw_filt_trace_toggled(self):
        if self.raw_filt_trace_toggle.isChecked() == True:
            self.raw_trace_view =False
            self.raw_pred_trace_view = False
            self.raw_filt_trace_view = True
            self.filt_trace_view = False
            self.stroke_trace_view = False
            self.update_seq_spin()

    def set_filt_trace_toggle(self,toggle_in):
        self.filt_trace_toggle = toggle_in
        self.filt_trace_toggle.setChecked(False)
        self.filt_trace_toggle.toggled.connect(self.filt_trace_toggled)

    def filt_trace_toggled(self):
        if self.filt_trace_toggle.isChecked() == True:
            self.raw_trace_view =False
            self.raw_pred_trace_view = False
            self.raw_filt_trace_view = False
            self.filt_trace_view = True
            self.stroke_trace_view = False
            self.update_seq_spin()

    def set_stroke_trace_toggle(self,toggle_in):
        self.stroke_trace_toggle = toggle_in
        self.stroke_trace_toggle.setChecked(False)
        self.stroke_trace_toggle.toggled.connect(self.stroke_trace_toggled)

    def stroke_trace_toggled(self):
        if self.stroke_trace_toggle.isChecked() == True:
            self.raw_trace_view =False
            self.raw_pred_trace_view = False
            self.raw_filt_trace_view = False
            self.filt_trace_view = False
            self.stroke_trace_view = True
            self.update_seq_spin()

    def set_q_plot(self,q_plot_in):
        self.q_plot = q_plot_in
        labels = ['q','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        if self.comp_ind==0:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][0:4,:])
        elif self.comp_ind==1:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][7:11,:])
        elif self.comp_ind==2:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][14:18,:])
        elif self.comp_ind==3:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][21:25,:])
        elif self.comp_ind==4:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][29:33,:])
        print(q_trace)
        curves.append([frame_trace,q_trace[0,:]])
        curves_pen.append((255,0,0))
        curves_legends.append('q0')
        curves.append([frame_trace,q_trace[1,:]])
        curves_pen.append((0,255,0))
        curves_legends.append('q1')
        curves.append([frame_trace,q_trace[2,:]])
        curves_pen.append((0,0,255))
        curves_legends.append('q2')
        curves.append([frame_trace,q_trace[3,:]])
        curves_pen.append((255,0,255))
        curves_legends.append('q3')
        self.q_plot.set_curves(labels,curves,curves_pen,curves_legends)

    def set_data_q_plot(self):
        labels = ['q','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        # Append to curves:
        if self.raw_trace_view:
            if self.comp_ind==0:
                q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][0:4,:])
            elif self.comp_ind==1:
                q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][7:11,:])
            elif self.comp_ind==2:
                q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][14:18,:])
            elif self.comp_ind==3:
                q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][21:25,:])
            elif self.comp_ind==4:
                q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][29:33,:])
            curves.append([frame_trace,q_trace[0,:]])
            curves_pen.append((255,0,0))
            curves_legends.append('q0')
            curves.append([frame_trace,q_trace[1,:]])
            curves_pen.append((0,255,0))
            curves_legends.append('q1')
            curves.append([frame_trace,q_trace[2,:]])
            curves_pen.append((0,0,255))
            curves_legends.append('q2')
            curves.append([frame_trace,q_trace[3,:]])
            curves_pen.append((255,0,255))
            curves_legends.append('q3')
        elif self.raw_pred_trace_view:
            if self.comp_ind==0:
                q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][0:4,:])
                q_pred = self.continuous_quaternions(self.pred_data[self.mov_nr][self.seq_nr][0:4,:])
            elif self.comp_ind==1:
                q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][7:11,:])
                q_pred = self.continuous_quaternions(self.pred_data[self.mov_nr][self.seq_nr][7:11,:])
            elif self.comp_ind==2:
                q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][14:18,:])
                q_pred = self.continuous_quaternions(self.pred_data[self.mov_nr][self.seq_nr][14:18,:])
            elif self.comp_ind==3:
                q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][21:25,:])
                q_pred = self.continuous_quaternions(self.pred_data[self.mov_nr][self.seq_nr][21:25,:])
            elif self.comp_ind==4:
                q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][29:33,:])
                q_pred = self.continuous_quaternions(self.pred_data[self.mov_nr][self.seq_nr][29:33,:])
            curves.append([frame_trace,q_fit[0,:]])
            curves.append([frame_trace,q_pred[0,:]])
            curves_pen.append((128,128,128))
            curves_pen.append((255,0,0))
            curves_legends.append('q0 raw')
            curves_legends.append('q0 pred')
            curves.append([frame_trace,q_fit[1,:]])
            curves.append([frame_trace,q_pred[1,:]])
            curves_pen.append((128,128,128))
            curves_pen.append((255,0,0))
            curves_legends.append('q1 raw')
            curves_legends.append('q1 pred')
            curves.append([frame_trace,q_fit[2,:]])
            curves.append([frame_trace,q_pred[2,:]])
            curves_pen.append((128,128,128))
            curves_pen.append((255,0,0))
            curves_legends.append('q2 raw')
            curves_legends.append('q2 pred')
            curves.append([frame_trace,q_fit[3,:]])
            curves.append([frame_trace,q_pred[3,:]])
            curves_pen.append((128,128,128))
            curves_pen.append((255,0,0))
            curves_legends.append('q3 raw')
            curves_legends.append('q3 pred')
        elif self.raw_filt_trace_view:
            if self.n_deriv_view==0:
                if self.comp_ind==0:
                    q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][0:4,:])
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][0:4,:]
                elif self.comp_ind==1:
                    q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][7:11,:])
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][0:4,:]
                elif self.comp_ind==2:
                    q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][14:18,:])
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][0:4,:]
                elif self.comp_ind==3:
                    q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][21:25,:])
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][0:4,:]
                elif self.comp_ind==4:
                    q_fit = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][29:33,:])
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][0:4,:]
                curves.append([frame_trace,q_fit[0,:]])
                curves.append([frame_trace,q_filt[0,:]])
                curves_pen.append((128,128,128))
                curves_pen.append((255,0,0))
                curves_legends.append('q0 raw')
                curves_legends.append('q0 filt')
                curves.append([frame_trace,q_fit[1,:]])
                curves.append([frame_trace,q_filt[1,:]])
                curves_pen.append((128,128,128))
                curves_pen.append((0,255,0))
                curves_legends.append('q1 raw')
                curves_legends.append('q1 filt')
                curves.append([frame_trace,q_fit[2,:]])
                curves.append([frame_trace,q_filt[2,:]])
                curves_pen.append((128,128,128))
                curves_pen.append((0,0,255))
                curves_legends.append('q2 raw')
                curves_legends.append('q2 filt')
                curves.append([frame_trace,q_fit[3,:]])
                curves.append([frame_trace,q_filt[3,:]])
                curves_pen.append((128,128,128))
                curves_pen.append((255,0,255))
                curves_legends.append('q3 raw')
                curves_legends.append('q3 filt')
            elif self.n_deriv_view==1:
                if self.comp_ind==0:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7:10,:]
                elif self.comp_ind==1:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7:10,:]
                elif self.comp_ind==2:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7:10,:]
                elif self.comp_ind==3:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][8:11,:]
                elif self.comp_ind==4:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][8:11,:]
                curves.append([frame_trace,q_filt[0,:]])
                curves_pen.append((255,0,0))
                curves_legends.append('wx')
                curves.append([frame_trace,q_filt[1,:]])
                curves_pen.append((0,255,0))
                curves_legends.append('wy')
                curves.append([frame_trace,q_filt[2,:]])
                curves_pen.append((0,0,255))
                curves_legends.append('wz')
            elif self.n_deriv_view==2:
                if self.comp_ind==0:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][13:16,:]
                elif self.comp_ind==1:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][13:16,:]
                elif self.comp_ind==2:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][13:16,:]
                elif self.comp_ind==3:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][15:18,:]
                elif self.comp_ind==4:
                    q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][15:18,:]
                curves.append([frame_trace,q_filt[0,:]])
                curves_pen.append((255,0,0))
                curves_legends.append('w dot x')
                curves.append([frame_trace,q_filt[1,:]])
                curves_pen.append((0,255,0))
                curves_legends.append('w dot y')
                curves.append([frame_trace,q_filt[2,:]])
                curves_pen.append((0,0,255))
                curves_legends.append('w dot z')
        elif self.filt_trace_view:
            if self.comp_ind==0:
                q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][0:4,:]
            elif self.comp_ind==1:
                q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7:11,:]
            elif self.comp_ind==2:
                q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][14:18,:]
            elif self.comp_ind==3:
                q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][21:25,:]
            elif self.comp_ind==4:
                q_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][29:33,:]
            curves.append([frame_trace,q_filt[0,:]])
            curves_pen.append((255,0,0))
            curves_legends.append('q0 filt')
            curves.append([frame_trace,q_filt[1,:]])
            curves_pen.append((0,255,0))
            curves_legends.append('q1 filt')
            curves.append([frame_trace,q_filt[2,:]])
            curves_pen.append((0,0,255))
            curves_legends.append('q2 filt')
            curves.append([frame_trace,q_filt[3,:]])
            curves_pen.append((255,0,255))
            curves_legends.append('q3 filt')
        elif self.stroke_trace_view:
            curves.append([frame_trace,self.Euler_angles[self.mov_nr][self.seq_nr][self.comp_ind][1,:]*(180.0/np.pi)])
            curves_pen.append((255,0,0))
            curves_legends.append('theta')
            curves.append([frame_trace,self.Euler_angles[self.mov_nr][self.seq_nr][self.comp_ind][2,:]*(180.0/np.pi)])
            curves_pen.append((0,255,0))
            curves_legends.append('eta')
            curves.append([frame_trace,self.Euler_angles[self.mov_nr][self.seq_nr][self.comp_ind][0,:]*(180.0/np.pi)])
            curves_pen.append((0,0,255))
            curves_legends.append('phi')
        self.q_plot.set_curves(labels,curves,curves_pen,curves_legends)
        self.q_plot_zoom.set_curves(labels,curves,curves_pen,curves_legends)
        if len(self.wb_intervals)>0:
            if self.wb_intervals[self.mov_nr][self.seq_nr].shape[0]>1:
                #self.q_plot.set_wb_lines(self.wb_intervals[self.mov_nr][self.seq_nr])
                self.q_plot_zoom.set_wb_lines(self.wb_intervals[self.mov_nr][self.seq_nr])

    def set_q_plot_zoom(self,q_plot_zoom_in):
        self.q_plot_zoom = q_plot_zoom_in
        labels = ['q','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        if self.comp_ind==0:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][0:4,:])
        elif self.comp_ind==1:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][7:11,:])
        elif self.comp_ind==2:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][14:18,:])
        elif self.comp_ind==3:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][21:25,:])
        elif self.comp_ind==4:
            q_trace = self.continuous_quaternions(self.fit_data[self.mov_nr][self.seq_nr][29:33,:])
        curves.append([frame_trace,q_trace[0,:]])
        curves_pen.append((255,0,0))
        curves_legends.append('q0')
        curves.append([frame_trace,q_trace[1,:]])
        curves_pen.append((0,255,0))
        curves_legends.append('q1')
        curves.append([frame_trace,q_trace[2,:]])
        curves_pen.append((0,0,255))
        curves_legends.append('q2')
        curves.append([frame_trace,q_trace[3,:]])
        curves_pen.append((255,0,255))
        curves_legends.append('q3')
        self.q_plot_zoom.set_curves(labels,curves,curves_pen,curves_legends)

    def set_t_plot(self,t_plot_zoom_in):
        self.t_plot = t_plot_zoom_in
        labels = ['t','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        if self.comp_ind==0:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][4,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][5,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][6,:]
        elif self.comp_ind==1:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][11,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][12,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][13,:]
        elif self.comp_ind==2:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][18,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][19,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][20,:]
        elif self.comp_ind==3:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][25,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][26,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][27,:]
        elif self.comp_ind==4:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][33,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][34,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][35,:]
        curves.append([frame_trace,t0_trace])
        curves_pen.append((255,0,0))
        curves_legends.append('tx')
        curves.append([frame_trace,t1_trace])
        curves_pen.append((0,255,0))
        curves_legends.append('ty')
        curves.append([frame_trace,t2_trace])
        curves_pen.append((0,0,255))
        curves_legends.append('tz')
        self.t_plot.set_curves(labels,curves,curves_pen,curves_legends)

    def set_data_t_plot(self):
        labels = ['t','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        # Append to curves:
        if self.raw_trace_view:
            if self.comp_ind==0:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][4,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][5,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][6,:]
            elif self.comp_ind==1:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][11,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][12,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][13,:]
            elif self.comp_ind==2:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][18,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][19,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][20,:]
            elif self.comp_ind==3:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][25,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][26,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][27,:]
            elif self.comp_ind==4:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][33,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][34,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][35,:]
            curves.append([frame_trace,t0_trace])
            curves_pen.append((255,0,0))
            curves_legends.append('tx')
            curves.append([frame_trace,t1_trace])
            curves_pen.append((0,255,0))
            curves_legends.append('ty')
            curves.append([frame_trace,t2_trace])
            curves_pen.append((0,0,255))
            curves_legends.append('tz')
        elif self.raw_pred_trace_view:
            if self.comp_ind==0:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][4,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][5,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][6,:]
                t0_pred = self.pred_data[self.mov_nr][self.seq_nr][4,:]
                t1_pred = self.pred_data[self.mov_nr][self.seq_nr][5,:]
                t2_pred = self.pred_data[self.mov_nr][self.seq_nr][6,:]
            elif self.comp_ind==1:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][11,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][12,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][13,:]
                t0_pred = self.pred_data[self.mov_nr][self.seq_nr][11,:]
                t1_pred = self.pred_data[self.mov_nr][self.seq_nr][12,:]
                t2_pred = self.pred_data[self.mov_nr][self.seq_nr][13,:]
            elif self.comp_ind==2:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][18,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][19,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][20,:]
                t0_pred = self.pred_data[self.mov_nr][self.seq_nr][18,:]
                t1_pred = self.pred_data[self.mov_nr][self.seq_nr][19,:]
                t2_pred = self.pred_data[self.mov_nr][self.seq_nr][20,:]
            elif self.comp_ind==3:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][25,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][26,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][27,:]
                t0_pred = self.pred_data[self.mov_nr][self.seq_nr][25,:]
                t1_pred = self.pred_data[self.mov_nr][self.seq_nr][26,:]
                t2_pred = self.pred_data[self.mov_nr][self.seq_nr][27,:]
            elif self.comp_ind==4:
                t0_trace = self.fit_data[self.mov_nr][self.seq_nr][33,:]
                t1_trace = self.fit_data[self.mov_nr][self.seq_nr][34,:]
                t2_trace = self.fit_data[self.mov_nr][self.seq_nr][35,:]
                t0_pred = self.pred_data[self.mov_nr][self.seq_nr][33,:]
                t1_pred = self.pred_data[self.mov_nr][self.seq_nr][34,:]
                t2_pred = self.pred_data[self.mov_nr][self.seq_nr][35,:]
            curves.append([frame_trace,t0_trace])
            curves.append([frame_trace,t0_pred])
            curves_pen.append((255,0,0))
            curves_pen.append((128,128,128))
            curves_legends.append('tx raw')
            curves_legends.append('tx pred')
            curves.append([frame_trace,t1_trace])
            curves.append([frame_trace,t1_pred])
            curves_pen.append((0,255,0))
            curves_pen.append((128,128,128))
            curves_legends.append('ty raw')
            curves_legends.append('ty pred')
            curves.append([frame_trace,t2_trace])
            curves.append([frame_trace,t2_pred])
            curves_pen.append((0,0,255))
            curves_pen.append((128,128,128))
            curves_legends.append('tz raw')
            curves_legends.append('tz pred')
        elif self.raw_filt_trace_view:
            if self.n_deriv_view==0:
                if self.comp_ind==0:
                    t0_trace = self.fit_data[self.mov_nr][self.seq_nr][4,:]
                    t1_trace = self.fit_data[self.mov_nr][self.seq_nr][5,:]
                    t2_trace = self.fit_data[self.mov_nr][self.seq_nr][6,:]
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
                elif self.comp_ind==1:
                    t0_trace = self.fit_data[self.mov_nr][self.seq_nr][11,:]
                    t1_trace = self.fit_data[self.mov_nr][self.seq_nr][12,:]
                    t2_trace = self.fit_data[self.mov_nr][self.seq_nr][13,:]
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
                elif self.comp_ind==2:
                    t0_trace = self.fit_data[self.mov_nr][self.seq_nr][18,:]
                    t1_trace = self.fit_data[self.mov_nr][self.seq_nr][19,:]
                    t2_trace = self.fit_data[self.mov_nr][self.seq_nr][20,:]
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
                elif self.comp_ind==3:
                    t0_trace = self.fit_data[self.mov_nr][self.seq_nr][25,:]
                    t1_trace = self.fit_data[self.mov_nr][self.seq_nr][26,:]
                    t2_trace = self.fit_data[self.mov_nr][self.seq_nr][27,:]
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
                elif self.comp_ind==4:
                    t0_trace = self.fit_data[self.mov_nr][self.seq_nr][33,:]
                    t1_trace = self.fit_data[self.mov_nr][self.seq_nr][34,:]
                    t2_trace = self.fit_data[self.mov_nr][self.seq_nr][35,:]
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
                curves.append([frame_trace,t0_trace])
                curves.append([frame_trace,t0_filt])
                curves_pen.append((128,128,128))
                curves_pen.append((255,0,0))
                curves_legends.append('tx raw')
                curves_legends.append('tx filt')
                curves.append([frame_trace,t1_trace])
                curves.append([frame_trace,t1_filt])
                curves_pen.append((128,128,128))
                curves_pen.append((0,255,0))
                curves_legends.append('ty raw')
                curves_legends.append('ty filt')
                curves.append([frame_trace,t2_trace])
                curves.append([frame_trace,t2_filt])
                curves_pen.append((128,128,128))
                curves_pen.append((0,0,255))
                curves_legends.append('tz raw')
                curves_legends.append('tz filt')
            elif self.n_deriv_view==1:
                if self.comp_ind==0:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][10,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][11,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][12,:]
                elif self.comp_ind==1:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][10,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][11,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][12,:]
                elif self.comp_ind==2:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][10,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][11,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][12,:]
                elif self.comp_ind==3:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][11,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][12,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][13,:]
                elif self.comp_ind==4:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][11,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][12,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][13,:]
                curves.append([frame_trace,t0_filt])
                curves_pen.append((255,0,0))
                curves_legends.append('vx')
                curves.append([frame_trace,t1_filt])
                curves_pen.append((0,255,0))
                curves_legends.append('vy')
                curves.append([frame_trace,t2_filt])
                curves_pen.append((0,0,255))
                curves_legends.append('vz')
            elif self.n_deriv_view==2:
                if self.comp_ind==0:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][16,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][17,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][18,:]
                elif self.comp_ind==1:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][16,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][17,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][18,:]
                elif self.comp_ind==2:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][16,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][17,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][18,:]
                elif self.comp_ind==3:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][18,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][19,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][20,:]
                elif self.comp_ind==4:
                    t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][18,:]
                    t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][19,:]
                    t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][20,:]
                curves.append([frame_trace,t0_filt])
                curves_pen.append((255,0,0))
                curves_legends.append('ax')
                curves.append([frame_trace,t1_filt])
                curves_pen.append((0,255,0))
                curves_legends.append('ay')
                curves.append([frame_trace,t2_filt])
                curves_pen.append((0,0,255))
                curves_legends.append('az')
        elif self.filt_trace_view:
            if self.comp_ind==0:
                t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
            elif self.comp_ind==1:
                t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
            elif self.comp_ind==2:
                t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
            elif self.comp_ind==3:
                t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
            elif self.comp_ind==4:
                t0_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][4,:]
                t1_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][5,:]
                t2_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][6,:]
            curves.append([frame_trace,t0_filt])
            curves_pen.append((255,0,0))
            curves_legends.append('tx filt')
            curves.append([frame_trace,t1_filt])
            curves_pen.append((0,255,0))
            curves_legends.append('ty filt')
            curves.append([frame_trace,t2_filt])
            curves_pen.append((0,0,255))
            curves_legends.append('tz filt')
        elif self.stroke_trace_view:
            curves.append([frame_trace,self.SRF_traces[self.mov_nr][self.seq_nr][self.comp_ind][4,:]])
            curves_pen.append((255,0,0))
            curves_legends.append('x root')
            curves.append([frame_trace,self.SRF_traces[self.mov_nr][self.seq_nr][self.comp_ind][5,:]])
            curves_pen.append((0,255,0))
            curves_legends.append('y root')
            curves.append([frame_trace,self.SRF_traces[self.mov_nr][self.seq_nr][self.comp_ind][6,:]])
            curves_pen.append((0,0,255))
            curves_legends.append('z root')
        self.t_plot.set_curves(labels,curves,curves_pen,curves_legends)
        self.t_plot_zoom.set_curves(labels,curves,curves_pen,curves_legends)
        if len(self.wb_intervals)>0:
            if self.wb_intervals[self.mov_nr][self.seq_nr].shape[0]>1:
                #self.t_plot.set_wb_lines(self.wb_intervals[self.mov_nr][self.seq_nr])
                self.t_plot_zoom.set_wb_lines(self.wb_intervals[self.mov_nr][self.seq_nr])

    def set_t_plot_zoom(self,t_plot_zoom_in):
        self.t_plot_zoom = t_plot_zoom_in
        labels = ['t','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        if self.comp_ind==0:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][4,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][5,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][6,:]
        elif self.comp_ind==1:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][11,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][12,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][13,:]
        elif self.comp_ind==2:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][18,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][19,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][20,:]
        elif self.comp_ind==3:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][25,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][26,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][27,:]
        elif self.comp_ind==4:
            t0_trace = self.fit_data[self.mov_nr][self.seq_nr][33,:]
            t1_trace = self.fit_data[self.mov_nr][self.seq_nr][34,:]
            t2_trace = self.fit_data[self.mov_nr][self.seq_nr][35,:]
        curves.append([frame_trace,t0_trace])
        curves_pen.append((255,0,0))
        curves_legends.append('tx')
        curves.append([frame_trace,t1_trace])
        curves_pen.append((0,255,0))
        curves_legends.append('ty')
        curves.append([frame_trace,t2_trace])
        curves_pen.append((0,0,255))
        curves_legends.append('tz')
        self.t_plot_zoom.set_curves(labels,curves,curves_pen,curves_legends)

    def set_x_plot(self,x_plot_in):
        self.x_plot = x_plot_in
        labels = ['x','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        x_trace = self.fit_data[self.mov_nr][self.seq_nr][28,:]
        if self.comp_ind==0:
            x_trace = np.zeros(frame_trace.shape)
        elif self.comp_ind==1:
            x_trace = np.zeros(frame_trace.shape)
        elif self.comp_ind==2:
            x_trace = np.zeros(frame_trace.shape)
        elif self.comp_ind==3:
            x_trace = self.fit_data[self.mov_nr][self.seq_nr][28,:]
        elif self.comp_ind==4:
            x_trace = self.fit_data[self.mov_nr][self.seq_nr][36,:]
        curves.append([frame_trace,x_trace])
        curves_pen.append((0,255,0))
        curves_legends.append('xi')
        self.x_plot.set_curves(labels,curves,curves_pen,curves_legends)

    def set_data_x_plot(self):
        labels = ['x','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        # Append to curves:
        if self.raw_trace_view:
            if self.comp_ind==0:
                x_trace = np.zeros(frame_trace.shape)
            elif self.comp_ind==1:
                x_trace = np.zeros(frame_trace.shape)
            elif self.comp_ind==2:
                x_trace = np.zeros(frame_trace.shape)
            elif self.comp_ind==3:
                x_trace = self.fit_data[self.mov_nr][self.seq_nr][28,:]
            elif self.comp_ind==4:
                x_trace = self.fit_data[self.mov_nr][self.seq_nr][36,:]
            curves.append([frame_trace,x_trace])
            curves_pen.append((0,255,0))
            curves_legends.append('xi')
        elif self.raw_pred_trace_view:
            if self.comp_ind==0:
                x_trace = np.zeros(frame_trace.shape)
                x_pred = np.zeros(frame_trace.shape)
            elif self.comp_ind==1:
                x_trace = np.zeros(frame_trace.shape)
                x_pred = np.zeros(frame_trace.shape)
            elif self.comp_ind==2:
                x_trace = np.zeros(frame_trace.shape)
                x_pred = np.zeros(frame_trace.shape)
            elif self.comp_ind==3:
                x_trace = self.fit_data[self.mov_nr][self.seq_nr][28,:]
                x_pred = self.pred_data[self.mov_nr][self.seq_nr][28,:]
            elif self.comp_ind==4:
                x_trace = self.fit_data[self.mov_nr][self.seq_nr][36,:]
                x_pred = self.pred_data[self.mov_nr][self.seq_nr][36,:]
            curves.append([frame_trace,x_trace])
            curves.append([frame_trace,x_pred])
            curves_pen.append((128,128,128))
            curves_pen.append((0,255,0))
            curves_legends.append('xi raw')
            curves_legends.append('xi pred')
        elif self.raw_filt_trace_view:
            if self.n_deriv_view==0:
                if self.comp_ind==0:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==1:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==2:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==3:
                    x_trace = self.fit_data[self.mov_nr][self.seq_nr][28,:]
                    x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7,:]
                elif self.comp_ind==4:
                    x_trace = self.fit_data[self.mov_nr][self.seq_nr][36,:]
                    x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7,:]
            elif self.n_deriv_view==1:
                if self.comp_ind==0:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==1:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==2:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==3:
                    x_trace = np.gradient(self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7,:])/self.dt
                    x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][14,:]
                elif self.comp_ind==4:
                    x_trace = np.gradient(self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7,:])/self.dt
                    x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][14,:]
            elif self.n_deriv_view==2:
                if self.comp_ind==0:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==1:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==2:
                    x_trace = np.zeros(frame_trace.shape)
                    x_filt = np.zeros(frame_trace.shape)
                elif self.comp_ind==3:
                    x_trace = np.gradient(np.gradient(self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][21,:]))/np.power(self.dt,2)
                    x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][21,:]
                elif self.comp_ind==4:
                    x_trace = np.gradient(np.gradient(self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][21,:]))/np.power(self.dt,2)
                    x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][21,:]
            curves.append([frame_trace,x_trace])
            curves.append([frame_trace,x_filt])
            curves_pen.append((128,128,128))
            curves_pen.append((0,255,0))
            curves_legends.append('xi raw')
            curves_legends.append('xi filt')
        elif self.filt_trace_view:
            if self.comp_ind==0:
                x_filt = np.zeros(frame_trace.shape)
            elif self.comp_ind==1:
                x_filt = np.zeros(frame_trace.shape)
            elif self.comp_ind==2:
                x_filt = np.zeros(frame_trace.shape)
            elif self.comp_ind==3:
                x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7,:]
            elif self.comp_ind==4:
                x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7,:]
            curves.append([frame_trace,x_filt])
            curves_pen.append((0,255,0))
            curves_legends.append('xi filt')
        elif self.stroke_trace_view:
            if self.comp_ind==0:
                x_filt = np.zeros(frame_trace.shape)
            elif self.comp_ind==1:
                x_filt = np.zeros(frame_trace.shape)
            elif self.comp_ind==2:
                x_filt = np.zeros(frame_trace.shape)
            elif self.comp_ind==3:
                x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7,:]
            elif self.comp_ind==4:
                x_filt = self.filtered_data[self.mov_nr][self.seq_nr][self.comp_ind][7,:]
            curves.append([frame_trace,x_filt])
            curves_pen.append((0,255,0))
            curves_legends.append('xi filt')
        self.x_plot.set_curves(labels,curves,curves_pen,curves_legends)
        self.x_plot_zoom.set_curves(labels,curves,curves_pen,curves_legends)
        if len(self.wb_intervals)>0:
            if self.wb_intervals[self.mov_nr][self.seq_nr].shape[0]>1:
                #self.x_plot.set_wb_lines(self.wb_intervals[self.mov_nr][self.seq_nr])
                self.x_plot_zoom.set_wb_lines(self.wb_intervals[self.mov_nr][self.seq_nr])

    def set_x_plot_zoom(self,x_plot_zoom_in):
        self.x_plot_zoom = x_plot_zoom_in
        labels = ['x','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        if self.comp_ind==0:
            x_trace = np.zeros(frame_trace.shape)
        elif self.comp_ind==1:
            x_trace = np.zeros(frame_trace.shape)
        elif self.comp_ind==2:
            x_trace = np.zeros(frame_trace.shape)
        elif self.comp_ind==3:
            x_trace = self.fit_data[self.mov_nr][self.seq_nr][28,:]
        elif self.comp_ind==4:
            x_trace = self.fit_data[self.mov_nr][self.seq_nr][36,:]
        curves.append([frame_trace,x_trace])
        curves_pen.append((0,255,0))
        curves_legends.append('xi')
        self.x_plot_zoom.set_curves(labels,curves,curves_pen,curves_legends)

    def set_cost_plot(self,cost_plot_in):
        self.cost_plot = cost_plot_in
        labels = ['cost','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        if self.comp_ind==0:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][37,:]
        elif self.comp_ind==1:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][38,:]
        elif self.comp_ind==2:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][39,:]
        elif self.comp_ind==3:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][40,:]
        elif self.comp_ind==4:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][41,:]
        curves.append([frame_trace,cost_trace])
        curves_pen.append((0,255,0))
        curves_legends.append('cost')
        self.cost_plot.set_curves(labels,curves,curves_pen,curves_legends)

    def set_data_cost_plot(self):
        labels = ['cost','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        if self.comp_ind==0:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][37,:]
        elif self.comp_ind==1:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][38,:]
        elif self.comp_ind==2:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][39,:]
        elif self.comp_ind==3:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][40,:]
        elif self.comp_ind==4:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][41,:]
        curves.append([frame_trace,cost_trace])
        curves_pen.append((0,255,0))
        curves_legends.append('cost')
        self.cost_plot.set_curves(labels,curves,curves_pen,curves_legends)
        self.cost_plot_zoom.set_curves(labels,curves,curves_pen,curves_legends)
        if len(self.wb_intervals)>0:
            if self.wb_intervals[self.mov_nr][self.seq_nr].shape[0]>1:
                #self.cost_plot.set_wb_lines(self.wb_intervals[self.mov_nr][self.seq_nr])
                self.cost_plot_zoom.set_wb_lines(self.wb_intervals[self.mov_nr][self.seq_nr])

    def set_cost_plot_zoom(self,cost_plot_zoom_in):
        self.cost_plot_zoom = cost_plot_zoom_in
        labels = ['cost','frame nr']
        frame_trace = np.arange(self.frame_nr_data[self.mov_nr][self.seq_nr][0],self.frame_nr_data[self.mov_nr][self.seq_nr][1])
        curves = []
        curves_pen = []
        curves_legends = []
        if self.comp_ind==0:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][37,:]
        elif self.comp_ind==1:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][38,:]
        elif self.comp_ind==2:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][39,:]
        elif self.comp_ind==3:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][40,:]
        elif self.comp_ind==4:
            cost_trace = self.fit_data[self.mov_nr][self.seq_nr][41,:]
        curves.append([frame_trace,cost_trace])
        curves_pen.append((0,255,0))
        curves_legends.append('cost')
        self.cost_plot_zoom.set_curves(labels,curves,curves_pen,curves_legends)

    def set_component_combo(self,combo_in,mdl_comp_in):
        self.component_combo = combo_in
        self.mdl_components = mdl_comp_in
        for comp in self.mdl_components:
            self.component_combo.addItem(comp)
        self.N_components = len(self.mdl_components)-1
        self.component_combo.currentIndexChanged.connect(self.component_selected)

    def component_selected(self,ind):
        self.comp_ind = ind-1
        self.set_data_q_plot()
        self.set_data_t_plot()
        self.set_data_x_plot()
        self.set_data_cost_plot()

    def set_kalman_combo(self,combo_in):
        self.kalman_combo = combo_in
        self.q_cov = []
        self.t_cov = []
        self.x_cov = []
        for i,comp in enumerate(self.mdl_components):
            self.kalman_combo.addItem(comp)
            if comp != '...':
                self.q_cov.append(self.q_cov_init[i-1])
                self.t_cov.append(self.t_cov_init[i-1])
                self.x_cov.append(self.x_cov_init[i-1])
        self.kalman_combo.currentIndexChanged.connect(self.kalman_comp_selected)

    def kalman_comp_selected(self,ind):
        self.kalman_comp_ind = ind-1
        self.q_kalman_spin.setValue(self.q_cov[self.kalman_comp_ind])
        self.t_kalman_spin.setValue(self.t_cov[self.kalman_comp_ind])
        self.x_kalman_spin.setValue(self.x_cov[self.kalman_comp_ind])

    def set_n_deriv_kalman_spin(self,spin_in):
        self.n_deriv_spin = spin_in
        self.n_deriv_spin.setMinimum(0)
        self.n_deriv_spin.setMaximum(2)
        self.n_deriv_spin.setValue(2)
        self.set_n_deriv_kalman(2)
        self.n_deriv_spin.valueChanged.connect(self.set_n_deriv_kalman)

    def set_n_deriv_kalman(self,n_in):
        self.n_deriv_kalman = n_in

    def set_fps_spin(self,spin_in):
        self.fps_spin = spin_in
        self.fps_spin.setMinimum(1)
        self.fps_spin.setMaximum(50000)
        self.fps_spin.setValue(15000)
        self.set_fps(15000)
        self.fps_spin.valueChanged.connect(self.set_fps)

    def set_fps(self,fps_in):
        self.fps = fps_in

    def set_q_kalman_spin(self,spin_in):
        self.q_kalman_spin = spin_in
        self.q_kalman_spin.setMinimum(0.0)
        self.q_kalman_spin.setMaximum(10.0)
        self.q_kalman_spin.setDecimals(6)
        self.q_kalman_spin.setSingleStep(0.000001)
        self.q_kalman_spin.setValue(1.0)
        self.q_kalman_spin.valueChanged.connect(self.set_q_covariance)

    def set_q_covariance(self,q_var_in):
        self.q_cov[self.kalman_comp_ind] = q_var_in

    def set_t_kalman_spin(self,spin_in):
        self.t_kalman_spin = spin_in
        self.t_kalman_spin.setMinimum(0.0)
        self.t_kalman_spin.setMaximum(10.0)
        self.t_kalman_spin.setDecimals(6)
        self.t_kalman_spin.setSingleStep(0.000001)
        self.t_kalman_spin.setValue(1.0)
        self.t_kalman_spin.valueChanged.connect(self.set_t_covariance)

    def set_t_covariance(self,t_var_in):
        self.t_cov[self.kalman_comp_ind] = t_var_in

    def set_x_kalman_spin(self,spin_in):
        self.x_kalman_spin = spin_in
        self.x_kalman_spin.setMinimum(0.0)
        self.x_kalman_spin.setMaximum(10.0)
        self.x_kalman_spin.setDecimals(6)
        self.x_kalman_spin.setSingleStep(0.000001)
        self.x_kalman_spin.setValue(1.0)
        self.x_kalman_spin.valueChanged.connect(self.set_x_covariance)

    def set_x_covariance(self,x_var_in):
        self.x_cov[self.kalman_comp_ind] = x_var_in

    def set_kalman_progress_bar(self,prog_bar_in):
        self.kalman_progress = prog_bar_in
        self.kalman_progress.setValue(0)

    def set_kalman_filter_btn(self,btn_in):
        self.filter_btn = btn_in
        self.filter_btn.clicked.connect(self.filter_data)

    def set_n_deriv_view_spin(self,spin_in):
        self.n_deriv_view_spin = spin_in
        self.n_deriv_view_spin.setMinimum(0)
        self.n_deriv_view_spin.setMaximum(2)
        self.n_deriv_view_spin.setValue(0)
        self.set_n_deriv_view(0)
        self.n_deriv_view_spin.valueChanged.connect(self.set_n_deriv_view)

    def set_n_deriv_view(self,n_deriv_in):
        self.n_deriv_view = n_deriv_in
        self.set_data_q_plot()
        self.set_data_t_plot()
        self.set_data_x_plot()

    '''
    def filter_data(self):
        self.filtered_data = []
        self.kalman_progress.setValue(0)
        for i in range(self.N_mov):
            filt_data_mov = []
            for j in range(len(self.frame_nr_data[i])):
                filt_data_seq = []
                for k,comp in enumerate(self.mdl_components):
                    if k==0:
                        Y = self.fit_data[i][j][0:7,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        N_samples = Y.shape[1]
                        self.EKF_filter.initialize(self.n_deriv_kalman,self.dt,False)
                        self.EKF_filter.set_Y(Y,N_samples)
                        self.EKF_filter.set_RQ(self.q_cov[k],self.t_cov[k],self.x_cov[k])
                        self.EKF_filter.filter_data()
                        Y_filt = self.EKF_filter.results()
                        filt_data_seq.append(Y_filt)
                    elif k==1:
                        Y = self.fit_data[i][j][7:14,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        N_samples = Y.shape[1]
                        self.EKF_filter.initialize(self.n_deriv_kalman,self.dt,False)
                        self.EKF_filter.set_Y(Y,N_samples)
                        self.EKF_filter.set_RQ(self.q_cov[k],self.t_cov[k],self.x_cov[k])
                        self.EKF_filter.filter_data()
                        Y_filt = self.EKF_filter.results()
                        filt_data_seq.append(Y_filt)
                    elif k==2:
                        Y = self.fit_data[i][j][14:21,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        N_samples = Y.shape[1]
                        self.EKF_filter.initialize(self.n_deriv_kalman,self.dt,False)
                        self.EKF_filter.set_Y(Y,N_samples)
                        self.EKF_filter.set_RQ(self.q_cov[k],self.t_cov[k],self.x_cov[k])
                        self.EKF_filter.filter_data()
                        Y_filt = self.EKF_filter.results()
                        filt_data_seq.append(Y_filt)
                    elif k==3:
                        Y = self.fit_data[i][j][21:29,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        self.EKF_filter.initialize(self.n_deriv_kalman,self.dt,True)
                        self.EKF_filter.set_Y(Y,N_samples)
                        self.EKF_filter.set_RQ(self.q_cov[k],self.t_cov[k],self.x_cov[k])
                        self.EKF_filter.filter_data()
                        Y_filt = self.EKF_filter.results()
                        filt_data_seq.append(Y_filt)
                    elif k==4:
                        Y = self.fit_data[i][j][29:37,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        self.EKF_filter.initialize(self.n_deriv_kalman,self.dt,True)
                        self.EKF_filter.set_Y(Y,N_samples)
                        self.EKF_filter.set_RQ(self.q_cov[k],self.t_cov[k],self.x_cov[k])
                        self.EKF_filter.filter_data()
                        Y_filt = self.EKF_filter.results()
                        filt_data_seq.append(Y_filt)
                filt_data_mov.append(filt_data_seq)
            self.kalman_progress.setValue(int(100.0*(i+1)/(1.0*self.N_mov)))
            self.filtered_data.append(filt_data_mov)
        self.set_mov_nr(self.mov_nr+1)
    '''

    def filter_data(self):
        self.filtered_data = []
        try:
            self.kalman_progress.setValue(0)
        except:
            pass
        for i in range(self.N_mov):
            filt_data_mov = []
            for j in range(len(self.frame_nr_data[i])):
                filt_data_seq = []
                for k,comp in enumerate(self.mdl_components):
                    if k==0:
                        Y = self.fit_data[i][j][0:7,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        N_samples = Y.shape[1]
                        Y_filt = np.zeros((19,N_samples))
                        # filter quaternion
                        EKF_filter = EKF(self.dt)
                        q_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[0,0],Y[1,0],Y[2,0],Y[3,0]])
                        Q_vec = self.q_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0,1.0])
                        q_smooth = EKF_filter.filter_data(Y[0:4,:],q_init,Q_vec)
                        # filter position
                        K_filter = KF(3,3,4,self.dt)
                        Q_vec = self.t_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0])
                        t_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[4,0],Y[5,0],Y[6,0]])
                        t_smooth = K_filter.filter_data(Y[4:7,:],t_init,Q_vec)
                        Y_filt[0:4,:] = q_smooth[9:13,:]
                        Y_filt[4:7,:] = t_smooth[9:12]
                        Y_filt[7:10,:] = q_smooth[6:9,:]
                        Y_filt[10:13,:] = t_smooth[6:9,:]
                        Y_filt[13:16,:] = q_smooth[3:6,:]
                        Y_filt[16:19,:] = t_smooth[3:6,:]
                        filt_data_seq.append(Y_filt)
                    elif k==1:
                        Y = self.fit_data[i][j][7:14,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        N_samples = Y.shape[1]
                        Y_filt = np.zeros((19,N_samples))
                        # filter quaternion
                        EKF_filter = EKF(self.dt)
                        q_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[0,0],Y[1,0],Y[2,0],Y[3,0]])
                        Q_vec = self.q_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0,1.0])
                        q_smooth = EKF_filter.filter_data(Y[0:4,:],q_init,Q_vec)
                        # filter position
                        K_filter = KF(3,3,4,self.dt)
                        Q_vec = self.t_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0])
                        t_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[4,0],Y[5,0],Y[6,0]])
                        t_smooth = K_filter.filter_data(Y[4:7,:],t_init,Q_vec)
                        Y_filt[0:4,:] = q_smooth[9:13,:]
                        Y_filt[4:7,:] = t_smooth[9:12]
                        Y_filt[7:10,:] = q_smooth[6:9,:]
                        Y_filt[10:13,:] = t_smooth[6:9,:]
                        Y_filt[13:16,:] = q_smooth[3:6,:]
                        Y_filt[16:19,:] = t_smooth[3:6,:]
                        filt_data_seq.append(Y_filt)
                    elif k==2:
                        Y = self.fit_data[i][j][14:21,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        N_samples = Y.shape[1]
                        Y_filt = np.zeros((19,N_samples))
                        # filter quaternion
                        EKF_filter = EKF(self.dt)
                        q_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[0,0],Y[1,0],Y[2,0],Y[3,0]])
                        Q_vec = self.q_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0,1.0])
                        q_smooth = EKF_filter.filter_data(Y[0:4,:],q_init,Q_vec)
                        # filter position
                        K_filter = KF(3,3,4,self.dt)
                        Q_vec = self.t_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0])
                        t_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[4,0],Y[5,0],Y[6,0]])
                        t_smooth = K_filter.filter_data(Y[4:7,:],t_init,Q_vec)
                        Y_filt[0:4,:] = q_smooth[9:13,:]
                        Y_filt[4:7,:] = t_smooth[9:12]
                        Y_filt[7:10,:] = q_smooth[6:9,:]
                        Y_filt[10:13,:] = t_smooth[6:9,:]
                        Y_filt[13:16,:] = q_smooth[3:6,:]
                        Y_filt[16:19,:] = t_smooth[3:6,:]
                        filt_data_seq.append(Y_filt)
                    elif k==3:
                        Y = self.fit_data[i][j][21:29,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        N_samples = Y.shape[1]
                        Y_filt = np.zeros((22,N_samples))
                        # filter quaternion
                        EKF_filter = EKF(self.dt)
                        q_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[0,0],Y[1,0],Y[2,0],Y[3,0]])
                        Q_vec = self.q_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0,1.0])
                        q_smooth = EKF_filter.filter_data(Y[0:4,:],q_init,Q_vec)
                        # filter position
                        K_filter = KF(3,3,4,self.dt)
                        Q_vec = self.t_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0])
                        t_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[4,0],Y[5,0],Y[6,0]])
                        t_smooth = K_filter.filter_data(Y[4:7,:],t_init,Q_vec)
                        # filter bending
                        K_filter = KF(1,1,4,self.dt)
                        x_init = np.array([0.0,0.0,0.0,Y[7,0]])
                        Q_vec = self.x_cov[k]*np.array([0.0,1/self.dt**4,1/self.dt**2,1.0])
                        x_smooth = K_filter.filter_data(Y[7,:],x_init,Q_vec)
                        Y_filt[0:4,:] = q_smooth[9:13,:]
                        Y_filt[4:7,:] = t_smooth[9:12]
                        Y_filt[7,:] = x_smooth[3,:]
                        Y_filt[8:11,:] = q_smooth[6:9,:]
                        Y_filt[11:14,:] = t_smooth[6:9,:]
                        Y_filt[14,:] = x_smooth[2,:]
                        Y_filt[15:18,:] = q_smooth[3:6,:]
                        Y_filt[18:21,:] = t_smooth[3:6,:]
                        Y_filt[21,:] = x_smooth[1,:]
                        filt_data_seq.append(Y_filt)
                    elif k==4:
                        Y = self.fit_data[i][j][29:37,:]
                        Y[0:4,:] = self.continuous_quaternions(Y[0:4,:])
                        N_samples = Y.shape[1]
                        Y_filt = np.zeros((22,N_samples))
                        # filter quaternion
                        EKF_filter = EKF(self.dt)
                        q_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[0,0],Y[1,0],Y[2,0],Y[3,0]])
                        Q_vec = self.q_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0,1.0])
                        q_smooth = EKF_filter.filter_data(Y[0:4,:],q_init,Q_vec)
                        # filter position
                        K_filter = KF(3,3,4,self.dt)
                        Q_vec = self.t_cov[k]*np.array([0.0,0.0,0.0,1/self.dt**4,1/self.dt**4,1/self.dt**4,1/self.dt**2,1/self.dt**2,1/self.dt**2,1.0,1.0,1.0])
                        t_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,Y[4,0],Y[5,0],Y[6,0]])
                        t_smooth = K_filter.filter_data(Y[4:7,:],t_init,Q_vec)
                        # filter bending
                        K_filter = KF(1,1,4,self.dt)
                        x_init = np.array([0.0,0.0,0.0,Y[7,0]])
                        Q_vec = self.x_cov[k]*np.array([0.0,1/self.dt**4,1/self.dt**2,1.0])
                        x_smooth = K_filter.filter_data(Y[7,:],x_init,Q_vec)
                        Y_filt[0:4,:] = q_smooth[9:13,:]
                        Y_filt[4:7,:] = t_smooth[9:12]
                        Y_filt[7,:] = x_smooth[3,:]
                        Y_filt[8:11,:] = q_smooth[6:9,:]
                        Y_filt[11:14,:] = t_smooth[6:9,:]
                        Y_filt[14,:] = x_smooth[2,:]
                        Y_filt[15:18,:] = q_smooth[3:6,:]
                        Y_filt[18:21,:] = t_smooth[3:6,:]
                        Y_filt[21,:] = x_smooth[1,:]
                        filt_data_seq.append(Y_filt)
                filt_data_mov.append(filt_data_seq)
            try:
                self.kalman_progress.setValue(int(100.0*(i+1)/(1.0*self.N_mov)))
            except:
                pass
            self.filtered_data.append(filt_data_mov)
        try:
            self.set_mov_nr(self.mov_nr+1)
        except:
            pass
        # Save filtered data:
        key_list = list(self.seq_file.keys())
        # Iterate over movies & sequences:
        for i in range(self.N_mov):
            n_seq = len(self.filtered_data[i])
            for j in range(n_seq):
                for k,comp in enumerate(self.mdl_components):
                    if k==0:
                        group_key = 'traces_m_'+str(i+1)+'_s_'+str(j+1)+'_head'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['raw']
                            temp_data[...] = self.fit_data[i][j][0:7,:]
                            temp_data = grp['predicted']
                            temp_data[...] = self.pred_data[i][j][0:7,:]
                            temp_data = grp['filtered']
                            temp_data[...] = self.filtered_data[i][j][k]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('raw', data=self.fit_data[i][j][0:7,:])
                            grp.create_dataset('predicted', data=self.pred_data[i][j][0:7,:])
                            grp.create_dataset('filtered', data=self.filtered_data[i][j][k])
                    elif k==1:
                        group_key = 'traces_m_'+str(i+1)+'_s_'+str(j+1)+'_thorax'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['raw']
                            temp_data[...] = self.fit_data[i][j][7:14,:]
                            temp_data = grp['predicted']
                            temp_data[...] = self.pred_data[i][j][7:14,:]
                            temp_data = grp['filtered']
                            temp_data[...] = self.filtered_data[i][j][k]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('raw', data=self.fit_data[i][j][7:14,:])
                            grp.create_dataset('predicted', data=self.pred_data[i][j][7:14,:])
                            grp.create_dataset('filtered', data=self.filtered_data[i][j][k])
                    elif k==2:
                        group_key = 'traces_m_'+str(i+1)+'_s_'+str(j+1)+'_abdomen'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['raw']
                            temp_data[...] = self.fit_data[i][j][14:21,:]
                            temp_data = grp['predicted']
                            temp_data[...] = self.pred_data[i][j][14:21,:]
                            temp_data = grp['filtered']
                            temp_data[...] = self.filtered_data[i][j][k]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('raw', data=self.fit_data[i][j][14:21,:])
                            grp.create_dataset('predicted', data=self.pred_data[i][j][14:21,:])
                            grp.create_dataset('filtered', data=self.filtered_data[i][j][k])
                    elif k==3:
                        group_key = 'traces_m_'+str(i+1)+'_s_'+str(j+1)+'_wing_L'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['raw']
                            temp_data[...] = self.fit_data[i][j][21:29,:]
                            temp_data = grp['predicted']
                            temp_data[...] = self.pred_data[i][j][21:29,:]
                            temp_data = grp['filtered']
                            temp_data[...] = self.filtered_data[i][j][k]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('raw', data=self.fit_data[i][j][21:29,:])
                            grp.create_dataset('predicted', data=self.pred_data[i][j][21:29,:])
                            grp.create_dataset('filtered', data=self.filtered_data[i][j][k])
                    elif k==4:
                        group_key = 'traces_m_'+str(i+1)+'_s_'+str(j+1)+'_wing_R'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['raw']
                            temp_data[...] = self.fit_data[i][j][29:37,:]
                            temp_data = grp['predicted']
                            temp_data[...] = self.pred_data[i][j][29:37,:]
                            temp_data = grp['filtered']
                            temp_data[...] = self.filtered_data[i][j][k]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('raw', data=self.fit_data[i][j][29:37,:])
                            grp.create_dataset('predicted', data=self.pred_data[i][j][29:37,:])
                            grp.create_dataset('filtered', data=self.filtered_data[i][j][k])
        print('saved filtered data')

    def set_N_phi_spin(self,spin_in):
        self.N_phi_spin = spin_in
        self.N_phi_spin.setMinimum(1)
        self.N_phi_spin.setMaximum(50)
        self.N_phi_spin.setValue(16)
        self.set_N_phi(16)
        self.N_phi_spin.valueChanged.connect(self.set_N_phi)

    def set_N_phi(self,N_in):
        self.N_phi = N_in

    def set_N_eta_spin(self,spin_in):
        self.N_eta_spin = spin_in
        self.N_eta_spin.setMinimum(1)
        self.N_eta_spin.setMaximum(50)
        self.N_eta_spin.setValue(24)
        self.set_N_eta(24)
        self.N_eta_spin.valueChanged.connect(self.set_N_eta)

    def set_N_eta(self,N_in):
        self.N_eta = N_in

    def set_N_theta_spin(self,spin_in):
        self.N_theta_spin = spin_in
        self.N_theta_spin.setMinimum(1)
        self.N_theta_spin.setMaximum(50)
        self.N_theta_spin.setValue(20)
        self.set_N_theta(20)
        self.N_theta_spin.valueChanged.connect(self.set_N_theta)

    def set_N_theta(self,N_in):
        self.N_theta = N_in

    def set_N_xi_spin(self,spin_in):
        self.N_xi_spin = spin_in
        self.N_xi_spin.setMinimum(1)
        self.N_xi_spin.setMaximum(50)
        self.N_xi_spin.setValue(20)
        self.set_N_xi(20)
        self.N_xi_spin.valueChanged.connect(self.set_N_xi)

    def set_N_xi(self,N_in):
        self.N_xi = N_in

    def set_fit_wbs_btn(self,btn_in):
        self.fit_wbs_btn = btn_in
        self.fit_wbs_btn.clicked.connect(self.fit_wbs)

    def set_strk_angle_spin(self,spin_in):
        self.strk_angle_spin = spin_in
        self.strk_angle_spin.setMinimum(-180)
        self.strk_angle_spin.setMaximum(180)
        self.strk_angle_spin.setValue(-45)
        self.set_SRF_angle(-45)
        self.strk_angle_spin.valueChanged.connect(self.set_SRF_angle)

    def set_SRF_angle(self,angle_in):
        self.SRF_angle = np.pi*angle_in/180.0

    def set_phi_thresh_spin(self,spin_in):
        self.phi_thresh_spin = spin_in
        self.phi_thresh_spin.setMinimum(-180)
        self.phi_thresh_spin.setMaximum(180)
        self.phi_thresh_spin.setValue(45)
        self.set_phi_thresh(45)
        self.phi_thresh_spin.valueChanged.connect(self.set_phi_thresh)

    def set_phi_thresh(self,thresh_in):
        self.phi_thresh = np.pi*thresh_in/180.0

    def set_Legendre_progress(self,progress_in):
        self.fit_wbs_progress = progress_in
        self.fit_wbs_progress.setValue(0)

    def set_batch_size(self,N_batch_in):
        self.N_batch = N_batch_in

    def fit_wbs(self):
        # Iterate over movies & sequences:
        self.Euler_angles = []
        self.SRF_traces = []
        self.wb_intervals = []
        self.wb_T = []
        self.wb_freq = []
        self.trace_intervals = []
        self.wingkin_intervals = []
        self.Legendre_fits = []
        try:
            self.fit_wbs_progress.setValue(0)
        except:
            pass
        key_list = list(self.seq_file.keys())
        self.thorax_corr_batch = 1000
        for i in range(self.N_mov):
            n_seq = len(self.filtered_data[i])
            Euler_angles_mov = []
            SRF_traces_mov = []
            wb_intervals_mov = []
            wb_T_mov = []
            wb_freq_mov = []
            for j in range(n_seq):
                # get nr of frames:
                n_frames = self.filtered_data[i][j][0].shape[1]
                #phi_L = np.zeros(n_frames)
                #phi_R = np.zeros(n_frames)
                # take the median value of all frames:
                #q_cg = np.transpose(np.matlib.repmat(np.median(self.filtered_data[i][j][1][0:4,:],axis=1),n_frames,1))
                #t_cg = np.transpose(np.matlib.repmat(np.median(self.filtered_data[i][j][1][4:7,:],axis=1),n_frames,1))
                N_mean = 2000

                q_mean = np.mean(self.filtered_data[i][j][1][0:4,:N_mean],axis=1)
                q_mean = q_mean/np.linalg.norm(q_mean)
                q_cg = np.transpose(np.matlib.repmat(q_mean,n_frames,1))
                #t_mean = np.mean(self.filtered_data[i][j][1][4:7,:],axis=1)
                t_mean = np.mean(self.filtered_data[i][j][1][4:7,:N_mean],axis=1)
                root_L = np.copy(self.filtered_data[i][j][3][4:7,:N_mean])
                root_R = np.copy(self.filtered_data[i][j][4][4:7,:N_mean])
                #t_mean = np.mean(np.concatenate((root_L,root_R),axis=1),axis=1)
                t_cg = np.transpose(np.matlib.repmat(t_mean,n_frames,1))
                #if np.mod(self.N_batch,2)==0:
                #    q_cg = medfilt(self.filtered_data[i][j][1][0:4,:],[1,self.thorax_corr_batch+1])
                #    t_cg = medfilt(self.filtered_data[i][j][1][4:7,:],[1,self.thorax_corr_batch+1])
                #else:
                #    q_cg = medfilt(self.filtered_data[i][j][1][0:4,:],[1,self.thorax_corr_batch])
                #    t_cg = medfilt(self.filtered_data[i][j][1][4:7,:],[1,self.thorax_corr_batch])
                # Apply correction thorax
                #self.thorax_corr_batch = n_frames
                #N_b2 = math.floor(self.thorax_corr_batch/2)
                #root_L = self.filtered_data[i][j][3][4:7,:]-t_cg
                #root_R = self.filtered_data[i][j][4][4:7,:]-t_cg
                q_cg   = self.thorax_roll_angle_correction(q_cg,root_L,root_R) # use only the first 2000 frames for SRF computation
                #for k in range(0,n_frames,N_b2):
                #    if (k+self.thorax_corr_batch)>n_frames:
                #        root_L = self.filtered_data[i][j][3][4:7,(n_frames-self.thorax_corr_batch):n_frames]
                #        root_R = self.filtered_data[i][j][4][4:7,(n_frames-self.thorax_corr_batch):n_frames]
                #        q_thorax = q_cg[:,(n_frames-self.thorax_corr_batch):n_frames]
                #        q_cg_corr = self.thorax_roll_angle_correction(q_thorax,root_L,root_R)
                #        q_cg[:,(n_frames-self.thorax_corr_batch):n_frames] = q_cg_corr
                #    else:
                #        root_L = self.filtered_data[i][j][3][4:7,k:(k+self.thorax_corr_batch)]
                #        root_R = self.filtered_data[i][j][4][4:7,k:(k+self.thorax_corr_batch)]
                #        q_thorax = q_cg[:,k:(k+self.thorax_corr_batch)]
                #        q_cg_corr = self.thorax_roll_angle_correction(q_thorax,root_L,root_R)
                #        q_cg[:,k:(k+self.thorax_corr_batch)] = q_cg_corr
                Euler_angles_seq = []
                SRF_traces_seq = []
                for k in range(self.N_components):
                    q_filt = np.copy(self.filtered_data[i][j][k][0:4,:])
                    t_filt = np.copy(self.filtered_data[i][j][k][4:7,:])
                    q_beta = np.array([np.cos(self.SRF_angle/2.0),0.0,np.sin(self.SRF_angle/2.0),0.0])
                    if k == 1:
                        # thorax
                        q_srf = q_cg
                        R_srf = self.R_matrix(q_srf)
                        t_srf = self.t_multiply(R_srf,t_filt,t_cg)
                    else:
                        #q_srf = q_cg
                        q_srf = self.q_multiply(q_beta,self.q_diff(q_cg,q_filt))
                        #q_srf = self.q_multiply(self.q_diff(q_cg,q_filt),q_beta)
                        q_beta = np.array([np.cos(-self.SRF_angle/2.0),0.0,np.sin(-self.SRF_angle/2.0),0.0])
                        q_root = self.q_multiply(q_cg,q_beta)
                        R_root = self.R_matrix(q_root)
                        t_srf = self.t_multiply(R_root,t_filt,t_cg)
                    e_angles = np.zeros((3,n_frames))
                    if k==0:
                        # head
                        R_srf = self.R_matrix(q_srf)
                        e_angles = self.Euler_x(R_srf)
                        SRF_traces_seq.append(np.append(q_srf,t_srf,axis=0))
                    elif k==1:
                        # thorax
                        R_srf = self.R_matrix(q_srf)
                        e_angles = self.Euler_x(R_srf)
                        SRF_traces_seq.append(np.append(q_srf,t_srf,axis=0))
                    elif k==2:
                        # abdomen
                        R_srf = self.R_matrix(q_srf)
                        e_angles = self.Euler_x(R_srf)
                        SRF_traces_seq.append(np.append(q_srf,t_srf,axis=0))
                    elif k==3:
                        # wing L
                        R_srf = self.R_matrix(q_srf)
                        e_angles = self.Euler_yL(R_srf)
                        phi_L = e_angles[0,:]
                        SRF_traces_seq.append(np.append(q_srf,t_srf,axis=0))
                    elif k==4:
                        # wing R
                        R_srf = self.R_matrix(q_srf)
                        e_angles = self.Euler_yR(R_srf)
                        phi_R = e_angles[0,:]
                        SRF_traces_seq.append(np.append(q_srf,t_srf,axis=0))
                    Euler_angles_seq.append(e_angles)
                Euler_angles_mov.append(Euler_angles_seq)
                SRF_traces_mov.append(SRF_traces_seq)
                # Parse wingbeats:
                #wb_inter, wb_t, wb_f = self.parse_wingbeats(phi_L,phi_R,20,self.phi_thresh,50,150,np.pi/4.0)
                root_SRF_L = SRF_traces_seq[3][5,:]
                root_SRF_R = SRF_traces_seq[4][5,:]
                wb_inter, wb_t, wb_f = self.parse_wingbeats(phi_L,phi_R,root_SRF_L,root_SRF_R,50,50,150)
                wb_intervals_mov.append(np.array(wb_inter))
                wb_T_mov.append(np.array(wb_t))
                wb_freq_mov.append(np.array(wb_f))
                # Iterate over wingbeats and perform Legendre fitting:
                N_wbs = len(wb_inter)
                for k in range(N_wbs):
                    try:
                        b1 = wb_inter[k][0]
                        b2 = wb_inter[k][1]
                        n_b = self.N_const+1
                        # Left wing:
                        theta_L     = Euler_angles_seq[3][1,b1:b2]
                        theta_b1_L     = Euler_angles_seq[3][1,b1-n_b:b1+n_b]
                        theta_b2_L     = Euler_angles_seq[3][1,b2-n_b:b2+n_b]
                        eta_L         = Euler_angles_seq[3][2,b1:b2]
                        eta_b1_L     = Euler_angles_seq[3][2,b1-n_b:b1+n_b]
                        eta_b2_L     = Euler_angles_seq[3][2,b2-n_b:b2+n_b]
                        phi_L         = Euler_angles_seq[3][0,b1:b2]
                        phi_b1_L     = Euler_angles_seq[3][0,b1-n_b:b1+n_b]
                        phi_b2_L     = Euler_angles_seq[3][0,b2-n_b:b2+n_b]
                        xi_L         = self.filtered_data[i][j][3][7,b1:b2]
                        xi_b1_L     = self.filtered_data[i][j][3][7,b1-n_b:b1+n_b]
                        xi_b2_L     = self.filtered_data[i][j][3][7,b2-n_b:b2+n_b]
                        x_L         = SRF_traces_seq[3][4,b1:b2]
                        x_b1_L         = SRF_traces_seq[3][4,b1-n_b:b1+n_b]
                        x_b2_L         = SRF_traces_seq[3][4,b2-n_b:b2+n_b]
                        y_L         = SRF_traces_seq[3][5,b1:b2]
                        y_b1_L         = SRF_traces_seq[3][5,b1-n_b:b1+n_b]
                        y_b2_L         = SRF_traces_seq[3][5,b2-n_b:b2+n_b]
                        z_L         = SRF_traces_seq[3][6,b1:b2]
                        z_b1_L         = SRF_traces_seq[3][6,b1-n_b:b1+n_b]
                        z_b2_L         = SRF_traces_seq[3][6,b2-n_b:b2+n_b]
                        a_theta_L     = self.LegendreFit(theta_L,theta_b1_L,theta_b2_L,self.N_theta,self.N_const)
                        a_eta_L     = self.LegendreFit(eta_L,eta_b1_L,eta_b2_L,self.N_eta,self.N_const)
                        a_phi_L     = self.LegendreFit(phi_L,phi_b1_L,phi_b2_L,self.N_phi,self.N_const)
                        a_xi_L      = self.LegendreFit(xi_L,xi_b1_L,xi_b2_L,self.N_xi,self.N_const)
                        a_x_L       = self.LegendreFit(x_L,x_b1_L,x_b2_L,self.N_xi,self.N_const)
                        a_y_L       = self.LegendreFit(y_L,y_b1_L,y_b2_L,self.N_xi,self.N_const)
                        a_z_L       = self.LegendreFit(z_L,z_b1_L,z_b2_L,self.N_xi,self.N_const)
                        # Right wing:
                        theta_R     = Euler_angles_seq[4][1,b1:b2]
                        theta_b1_R     = Euler_angles_seq[4][1,b1-n_b:b1+n_b]
                        theta_b2_R     = Euler_angles_seq[4][1,b2-n_b:b2+n_b]
                        eta_R         = Euler_angles_seq[4][2,b1:b2]
                        eta_b1_R     = Euler_angles_seq[4][2,b1-n_b:b1+n_b]
                        eta_b2_R     = Euler_angles_seq[4][2,b2-n_b:b2+n_b]
                        phi_R         = Euler_angles_seq[4][0,b1:b2]
                        phi_b1_R     = Euler_angles_seq[4][0,b1-n_b:b1+n_b]
                        phi_b2_R     = Euler_angles_seq[4][0,b2-n_b:b2+n_b]
                        xi_R         = self.filtered_data[i][j][4][7,b1:b2]
                        xi_b1_R     = self.filtered_data[i][j][4][7,b1-n_b:b1+n_b]
                        xi_b2_R     = self.filtered_data[i][j][4][7,b2-n_b:b2+n_b]
                        x_R         = SRF_traces_seq[4][4,b1:b2]
                        x_b1_R         = SRF_traces_seq[4][4,b1-n_b:b1+n_b]
                        x_b2_R         = SRF_traces_seq[4][4,b2-n_b:b2+n_b]
                        y_R         = SRF_traces_seq[4][5,b1:b2]
                        y_b1_R         = SRF_traces_seq[4][5,b1-n_b:b1+n_b]
                        y_b2_R         = SRF_traces_seq[4][5,b2-n_b:b2+n_b]
                        z_R         = SRF_traces_seq[4][6,b1:b2]
                        z_b1_R         = SRF_traces_seq[4][6,b1-n_b:b1+n_b]
                        z_b2_R         = SRF_traces_seq[4][6,b2-n_b:b2+n_b]
                        a_theta_R     = self.LegendreFit(theta_R,theta_b1_R,theta_b2_R,self.N_theta,self.N_const)
                        a_eta_R     = self.LegendreFit(eta_R,eta_b1_R,eta_b2_R,self.N_eta,self.N_const)
                        a_phi_R     = self.LegendreFit(phi_R,phi_b1_R,phi_b2_R,self.N_phi,self.N_const)
                        a_xi_R      = self.LegendreFit(xi_R,xi_b1_R,xi_b2_R,self.N_xi,self.N_const)
                        a_x_R       = self.LegendreFit(x_R,x_b1_R,x_b2_R,self.N_xi,self.N_const)
                        a_y_R          = self.LegendreFit(y_R,y_b1_R,y_b2_R,self.N_xi,self.N_const)
                        a_z_R        = self.LegendreFit(z_R,z_b1_R,z_b2_R,self.N_xi,self.N_const)
                        # save fits:
                        group_key = 'wb_m_'+str(i+1)+'_s_'+str(j+1)+'_n_'+str(k+1)
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            grp_keys = grp.keys()
                            temp_data = grp['b1_b2']
                            temp_data[...] = np.array([b1,b2])
                            #temp_data = grp['time']
                            #temp_data[...] = np.arange(b1,b2+1)*self.dt
                            if 'time' in grp_keys:
                                del grp['time']
                            grp.create_dataset('time', data=np.arange(b1,b2+1)*self.dt)
                            temp_data = grp['T']
                            temp_data[...] = wb_t[k]
                            temp_data = grp['freq']
                            temp_data[...] = wb_f[k]
                            if 'theta_L' in grp_keys:
                                del grp['theta_L']
                            grp.create_dataset('theta_L', data=theta_L)
                            if 'eta_L' in grp_keys:
                                del grp['eta_L']
                            grp.create_dataset('eta_L', data=eta_L)
                            if 'phi_L' in grp_keys:
                                del grp['phi_L']
                            grp.create_dataset('phi_L', data=phi_L)
                            if 'xi_L' in grp_keys:
                                del grp['xi_L']
                            grp.create_dataset('xi_L', data=xi_L)
                            if 'x_L' in grp_keys:
                                del grp['x_L']
                            grp.create_dataset('x_L', data=x_L)
                            if 'y_L' in grp_keys:
                                del grp['y_L']
                            grp.create_dataset('y_L', data=y_L)
                            if 'z_L' in grp_keys:
                                del grp['z_L']
                            grp.create_dataset('z_L', data=z_L)
                            if 'theta_R' in grp_keys:
                                del grp['theta_R']
                            grp.create_dataset('theta_R', data=theta_R)
                            if 'eta_R' in grp_keys:
                                del grp['eta_R']
                            grp.create_dataset('eta_R', data=eta_R)
                            if 'phi_R' in grp_keys:
                                del grp['phi_R']
                            grp.create_dataset('phi_R', data=phi_R)
                            if 'xi_R' in grp_keys:
                                del grp['xi_R']
                            grp.create_dataset('xi_R', data=xi_R)
                            if 'x_R' in grp_keys:
                                del grp['x_R']
                            grp.create_dataset('x_R', data=x_R)
                            if 'y_R' in grp_keys:
                                del grp['y_R']
                            grp.create_dataset('y_R', data=y_R)
                            if 'z_R' in grp_keys:
                                del grp['z_R']
                            grp.create_dataset('z_R', data=z_R)
                            temp_data = grp['a_theta_L']
                            temp_data[...] = a_theta_L
                            temp_data = grp['a_eta_L']
                            temp_data[...] = a_eta_L
                            temp_data = grp['a_phi_L']
                            temp_data[...] = a_phi_L
                            temp_data = grp['a_xi_L']
                            temp_data[...] = a_xi_L
                            temp_data = grp['a_x_L']
                            temp_data[...] = a_x_L
                            temp_data = grp['a_y_L']
                            temp_data[...] = a_y_L
                            temp_data = grp['a_z_L']
                            temp_data[...] = a_z_L
                            temp_data = grp['a_theta_R']
                            temp_data[...] = a_theta_R
                            temp_data = grp['a_eta_R']
                            temp_data[...] = a_eta_R
                            temp_data = grp['a_phi_R']
                            temp_data[...] = a_phi_R
                            temp_data = grp['a_xi_R']
                            temp_data[...] = a_xi_R
                            temp_data = grp['a_x_R']
                            temp_data[...] = a_x_R
                            temp_data = grp['a_y_R']
                            temp_data[...] = a_y_R
                            temp_data = grp['a_z_R']
                            temp_data[...] = a_z_R
                            if 'filt_trace_head' in grp_keys:
                                del grp['filt_trace_head']
                            grp.create_dataset('filt_trace_head', data=self.filtered_data[i][j][0][:,b1:(b2+1)])
                            if 'filt_trace_thorax' in grp_keys:
                                del grp['filt_trace_thorax']
                            grp.create_dataset('filt_trace_thorax', data=self.filtered_data[i][j][1][:,b1:(b2+1)])
                            if 'filt_trace_abdomen' in grp_keys:
                                del grp['filt_trace_abdomen']
                            grp.create_dataset('filt_trace_abdomen', data=self.filtered_data[i][j][2][:,b1:(b2+1)])
                            if 'filt_trace_wing_L' in grp_keys:
                                del grp['filt_trace_wing_L']
                            grp.create_dataset('filt_trace_wing_L', data=self.filtered_data[i][j][3][:,b1:(b2+1)])
                            if 'filt_trace_wing_R' in grp_keys:
                                del grp['filt_trace_wing_R']
                            grp.create_dataset('filt_trace_wing_R', data=self.filtered_data[i][j][4][:,b1:(b2+1)])
                            if 'srf_trace_head' in grp_keys:
                                del grp['srf_trace_head']
                            srf_trace_head = np.concatenate((Euler_angles_seq[0][:,b1:(b2+1)],SRF_traces_seq[0][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_head',data=srf_trace_head)
                            if 'srf_trace_thorax' in grp_keys:
                                del grp['srf_trace_thorax']
                            srf_trace_thorax = np.concatenate((Euler_angles_seq[1][:,b1:(b2+1)],SRF_traces_seq[1][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_thorax',data=srf_trace_thorax)
                            if 'srf_trace_abdomen' in grp_keys:
                                del grp['srf_trace_abdomen']
                            srf_trace_abdomen = np.concatenate((Euler_angles_seq[2][:,b1:(b2+1)],SRF_traces_seq[2][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_abdomen',data=srf_trace_abdomen)
                            if 'srf_trace_wing_L' in grp_keys:
                                del grp['srf_trace_wing_L']
                            srf_trace_wing_L = np.concatenate((Euler_angles_seq[3][:,b1:(b2+1)],SRF_traces_seq[3][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_wing_L',data=srf_trace_wing_L)
                            if 'srf_trace_wing_R' in grp_keys:
                                del grp['srf_trace_wing_R']
                            srf_trace_wing_R = np.concatenate((Euler_angles_seq[4][:,b1:(b2+1)],SRF_traces_seq[4][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_wing_R',data=srf_trace_wing_R)
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('b1_b2', data=np.array([b1,b2]))
                            grp.create_dataset('time', data=np.arange(b1,b2+1)*self.dt)
                            grp.create_dataset('T', data=wb_t[k])
                            grp.create_dataset('freq', data=wb_f[k])
                            grp.create_dataset('theta_L', data=theta_L)
                            grp.create_dataset('eta_L', data=eta_L)
                            grp.create_dataset('phi_L', data=phi_L)
                            grp.create_dataset('xi_L', data=xi_L)
                            grp.create_dataset('x_L', data=x_L)
                            grp.create_dataset('y_L', data=y_L)
                            grp.create_dataset('z_L', data=z_L)
                            grp.create_dataset('theta_R', data=theta_R)
                            grp.create_dataset('eta_R', data=eta_R)
                            grp.create_dataset('phi_R', data=phi_R)
                            grp.create_dataset('xi_R', data=xi_R)
                            grp.create_dataset('x_R', data=x_R)
                            grp.create_dataset('y_R', data=y_R)
                            grp.create_dataset('z_R', data=z_R)
                            grp.create_dataset('a_theta_L', data=a_theta_L)
                            grp.create_dataset('a_eta_L', data=a_eta_L)
                            grp.create_dataset('a_phi_L', data=a_phi_L)
                            grp.create_dataset('a_xi_L', data=a_xi_L)
                            grp.create_dataset('a_x_L', data=a_x_L)
                            grp.create_dataset('a_y_L', data=a_y_L)
                            grp.create_dataset('a_z_L', data=a_z_L)
                            grp.create_dataset('a_theta_R', data=a_theta_R)
                            grp.create_dataset('a_eta_R', data=a_eta_R)
                            grp.create_dataset('a_phi_R', data=a_phi_R)
                            grp.create_dataset('a_xi_R', data=a_xi_R)
                            grp.create_dataset('a_x_R', data=a_x_R)
                            grp.create_dataset('a_y_R', data=a_y_R)
                            grp.create_dataset('a_z_R', data=a_z_R)
                            grp.create_dataset('filt_trace_head', data=self.filtered_data[i][j][0][:,b1:(b2+1)])
                            grp.create_dataset('filt_trace_thorax', data=self.filtered_data[i][j][1][:,b1:(b2+1)])
                            grp.create_dataset('filt_trace_abdomen', data=self.filtered_data[i][j][2][:,b1:(b2+1)])
                            grp.create_dataset('filt_trace_wing_L', data=self.filtered_data[i][j][3][:,b1:(b2+1)])
                            grp.create_dataset('filt_trace_wing_R', data=self.filtered_data[i][j][4][:,b1:(b2+1)])
                            srf_trace_head = np.concatenate((Euler_angles_seq[0][:,b1:(b2+1)],SRF_traces_seq[0][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_head', data=srf_trace_head)
                            srf_trace_thorax = np.concatenate((Euler_angles_seq[1][:,b1:(b2+1)],SRF_traces_seq[1][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_thorax', data=srf_trace_thorax)
                            srf_trace_abdomen = np.concatenate((Euler_angles_seq[2][:,b1:(b2+1)],SRF_traces_seq[2][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_abdomen', data=srf_trace_abdomen)
                            srf_trace_wing_L = np.concatenate((Euler_angles_seq[3][:,b1:(b2+1)],SRF_traces_seq[3][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_wing_L', data=srf_trace_wing_L)
                            srf_trace_wing_R = np.concatenate((Euler_angles_seq[4][:,b1:(b2+1)],SRF_traces_seq[4][:,b1:(b2+1)]),axis=0)
                            grp.create_dataset('srf_trace_wing_R', data=srf_trace_wing_R)
                    except:
                        print('could not create wingbeat: '+str(k))
                # Save Euler angles:
                for k in range(5):
                    if k==0:
                        group_key = 'srf_m_'+str(i+1)+'_s_'+str(j+1)+'_head'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['phi']
                            temp_data[...] = Euler_angles_seq[0][0,:]
                            temp_data = grp['theta']
                            temp_data[...] = Euler_angles_seq[0][1,:]
                            temp_data = grp['eta']
                            temp_data[...] = Euler_angles_seq[0][2,:]
                            temp_data = grp['x']
                            temp_data[...] = SRF_traces_seq[0][4,:]
                            temp_data = grp['y']
                            temp_data[...] = SRF_traces_seq[0][5,:]
                            temp_data = grp['z']
                            temp_data[...] = SRF_traces_seq[0][6,:]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('phi', data=Euler_angles_seq[0][0,:])
                            grp.create_dataset('theta', data=Euler_angles_seq[0][1,:])
                            grp.create_dataset('eta', data=Euler_angles_seq[0][2,:])
                            grp.create_dataset('x', data=SRF_traces_seq[0][4,:])
                            grp.create_dataset('y', data=SRF_traces_seq[0][5,:])
                            grp.create_dataset('z', data=SRF_traces_seq[0][6,:])
                    elif k==1:
                        group_key = 'srf_m_'+str(i+1)+'_s_'+str(j+1)+'_thorax'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['phi']
                            temp_data[...] = Euler_angles_seq[1][0,:]
                            temp_data = grp['theta']
                            temp_data[...] = Euler_angles_seq[1][1,:]
                            temp_data = grp['eta']
                            temp_data[...] = Euler_angles_seq[1][2,:]
                            temp_data = grp['x']
                            temp_data[...] = SRF_traces_seq[1][4,:]
                            temp_data = grp['y']
                            temp_data[...] = SRF_traces_seq[1][5,:]
                            temp_data = grp['z']
                            temp_data[...] = SRF_traces_seq[1][6,:]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('phi', data=Euler_angles_seq[1][0,:])
                            grp.create_dataset('theta', data=Euler_angles_seq[1][1,:])
                            grp.create_dataset('eta', data=Euler_angles_seq[1][2,:])
                            grp.create_dataset('x', data=SRF_traces_seq[1][4,:])
                            grp.create_dataset('y', data=SRF_traces_seq[1][5,:])
                            grp.create_dataset('z', data=SRF_traces_seq[1][6,:])
                    elif k==2:
                        group_key = 'srf_m_'+str(i+1)+'_s_'+str(j+1)+'_abdomen'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['phi']
                            temp_data[...] = Euler_angles_seq[2][0,:]
                            temp_data = grp['theta']
                            temp_data[...] = Euler_angles_seq[2][1,:]
                            temp_data = grp['eta']
                            temp_data[...] = Euler_angles_seq[2][2,:]
                            temp_data = grp['x']
                            temp_data[...] = SRF_traces_seq[2][4,:]
                            temp_data = grp['y']
                            temp_data[...] = SRF_traces_seq[2][5,:]
                            temp_data = grp['z']
                            temp_data[...] = SRF_traces_seq[2][6,:]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('phi', data=Euler_angles_seq[2][0,:])
                            grp.create_dataset('theta', data=Euler_angles_seq[2][1,:])
                            grp.create_dataset('eta', data=Euler_angles_seq[2][2,:])
                            grp.create_dataset('x', data=SRF_traces_seq[2][4,:])
                            grp.create_dataset('y', data=SRF_traces_seq[2][5,:])
                            grp.create_dataset('z', data=SRF_traces_seq[2][6,:])
                    elif k==3:
                        group_key = 'srf_m_'+str(i+1)+'_s_'+str(j+1)+'_wing_L'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['phi']
                            temp_data[...] = Euler_angles_seq[3][0,:]
                            temp_data = grp['theta']
                            temp_data[...] = Euler_angles_seq[3][1,:]
                            temp_data = grp['eta']
                            temp_data[...] = Euler_angles_seq[3][2,:]
                            temp_data = grp['xi']
                            temp_data[...] = self.filtered_data[i][j][3][7,:]
                            temp_data = grp['x']
                            temp_data[...] = SRF_traces_seq[3][4,:]
                            temp_data = grp['y']
                            temp_data[...] = SRF_traces_seq[3][5,:]
                            temp_data = grp['z']
                            temp_data[...] = SRF_traces_seq[3][6,:]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('phi', data=Euler_angles_seq[3][0,:])
                            grp.create_dataset('theta', data=Euler_angles_seq[3][1,:])
                            grp.create_dataset('eta', data=Euler_angles_seq[3][2,:])
                            grp.create_dataset('xi', data=self.filtered_data[i][j][3][7,:])
                            grp.create_dataset('x', data=SRF_traces_seq[3][4,:])
                            grp.create_dataset('y', data=SRF_traces_seq[3][5,:])
                            grp.create_dataset('z', data=SRF_traces_seq[3][6,:])
                    elif k==4:
                        group_key = 'srf_m_'+str(i+1)+'_s_'+str(j+1)+'_wing_R'
                        if group_key in key_list:
                            grp = self.seq_file[group_key]
                            temp_data = grp['phi']
                            temp_data[...] = Euler_angles_seq[4][0,:]
                            temp_data = grp['theta']
                            temp_data[...] = Euler_angles_seq[4][1,:]
                            temp_data = grp['eta']
                            temp_data[...] = Euler_angles_seq[4][2,:]
                            temp_data = grp['xi']
                            temp_data[...] = self.filtered_data[i][j][4][7,:]
                            temp_data = grp['x']
                            temp_data[...] = SRF_traces_seq[4][4,:]
                            temp_data = grp['y']
                            temp_data[...] = SRF_traces_seq[4][5,:]
                            temp_data = grp['z']
                            temp_data[...] = SRF_traces_seq[4][6,:]
                        else:
                            grp = self.seq_file.create_group(group_key)
                            grp.create_dataset('phi', data=Euler_angles_seq[4][0,:])
                            grp.create_dataset('theta', data=Euler_angles_seq[4][1,:])
                            grp.create_dataset('eta', data=Euler_angles_seq[4][2,:])
                            grp.create_dataset('xi', data=self.filtered_data[i][j][4][7,:])
                            grp.create_dataset('x', data=SRF_traces_seq[4][4,:])
                            grp.create_dataset('y', data=SRF_traces_seq[4][5,:])
                            grp.create_dataset('z', data=SRF_traces_seq[4][6,:])
            self.Euler_angles.append(Euler_angles_mov)
            self.SRF_traces.append(SRF_traces_mov)
            try:
                self.fit_wbs_progress.setValue(int(100.0*(i+1)/(1.0*self.N_mov)))
            except:
                pass
            self.wb_intervals.append(wb_intervals_mov)
            self.wb_T.append(wb_T_mov)
            self.wb_freq.append(wb_freq_mov)

    def q_multiply(self,qA,qB):
        try:
            NA = qA.shape[1]
        except:
            NA = 1
        try:
            NB = qB.shape[1]
        except:
            NB = 1
        if NA == 1:
            QA = np.array([[qA[0],-qA[1],-qA[2],-qA[3]],
                [qA[1],qA[0],-qA[3],qA[2]],
                [qA[2],qA[3],qA[0],-qA[1]],
                [qA[3],-qA[2],qA[1],qA[0]]])
            if NB == 1:
                qC = np.dot(QA,qB)
                qC /= np.linalg.norm(qC)
            else:
                qC = np.zeros((4,NB))
                for i in range(NB):
                    qC[:,i] = np.dot(QA,qB[:,i])
                    qC[:,i] /= np.linalg.norm(qC[:,i])
        else:
            qC = np.zeros((4,NA))
            for i in range(NA):
                QA = np.array([[qA[0,i],-qA[1,i],-qA[2,i],-qA[3,i]],
                    [qA[1,i],qA[0,i],-qA[3,i],qA[2,i]],
                    [qA[2,i],qA[3,i],qA[0,i],-qA[1,i]],
                    [qA[3,i],-qA[2,i],qA[1,i],qA[0,i]]])
                if NB == 1:
                    qC[:,i] = np.dot(QA,qB)
                    qC[:,i] /= np.linalg.norm(qC[:,i])
                else:
                    qC[:,i] = np.dot(QA,qB[:,i])
                    qC[:,i] /= np.linalg.norm(qC[:,i])
        return qC

    def q_conj_multiply(self,qA,qB):
        try:
            NA = qA.shape[1]
        except:
            NA = 1
        try:
            NB = qB.shape[1]
        except:
            NB = 1
        if NA == 1:
            QA = np.array([[qA[0],-qA[1],-qA[2],-qA[3]],
                [qA[1],qA[0],qA[3],-qA[2]],
                [qA[2],-qA[3],qA[0],qA[1]],
                [qA[3],qA[2],-qA[1],qA[0]]])
            qC = np.dot(QA,qB)
            qC /= np.linalg.norm(qC)
        else:
            qC = np.zeros((4,NA))
            for i in range(NA):
                QA = np.array([[qA[0,i],qA[1,i],qA[2,i],qA[3,i]],
                    [-qA[1,i],qA[0,i],-qA[3,i],qA[2,i]],
                    [-qA[2,i],qA[3,i],qA[0,i],-qA[1,i]],
                    [-qA[3,i],-qA[2,i],qA[1,i],qA[0,i]]])
                if NB == 1:
                    qC[:,i] = np.dot(QA,qB)
                    qC[:,i] /= np.linalg.norm(qC[:,i])
                else:
                    qC[:,i] = np.dot(QA,qB[:,i])
                    qC[:,i] /= np.linalg.norm(qC[:,i])
        return qC

    def t_multiply(self,R,tA,tB):
        try:
            NA = tA.shape[1]
        except:
            NA = 1
        try:
            NB = tB.shape[1]
        except:
            NB = 1
        if NA == 1:
            tC = np.dot(R,tA-tB)
        else:
            tC = np.zeros((3,NA))
            if NB == 1:
                for i in range(NA):
                    tC[:,i] = np.dot(np.transpose(R[:,:,i]),tA[:,i]-tB)
            else:
                for i in range(NA):
                    tC[:,i] = np.dot(np.transpose(R[:,:,i]),tA[:,i]-tB[:,i])
        return tC

    def R_matrix(self,qA):
        try:
            NA = qA.shape[1]
        except:
            NA = 1
        if NA==1:
            R = np.array([[2*qA[0]**2-1+2*qA[1]**2, 2*qA[1]*qA[2]-2*qA[0]*qA[3], 2*qA[1]*qA[3]+2*qA[0]*qA[2]],
                [2*qA[1]*qA[2]+2*qA[0]*qA[3], 2*qA[0]**2-1+2*qA[2]**2, 2*qA[2]*qA[3]-2*qA[0]*qA[1]],
                [2*qA[1]*qA[3]-2*qA[0]*qA[2], 2*qA[2]*qA[3]+2*qA[0]*qA[1], 2*qA[0]**2-1+2*qA[3]**2]])
        else:
            R = np.zeros((3,3,NA))
            for i in range(NA):
                R[:,:,i] = np.array([[2*qA[0,i]**2-1+2*qA[1,i]**2, 2*qA[1,i]*qA[2,i]-2*qA[0,i]*qA[3,i], 2*qA[1,i]*qA[3,i]+2*qA[0,i]*qA[2,i]],
                    [2*qA[1,i]*qA[2,i]+2*qA[0,i]*qA[3,i], 2*qA[0,i]**2-1+2*qA[2,i]**2, 2*qA[2,i]*qA[3,i]-2*qA[0,i]*qA[1,i]],
                    [2*qA[1,i]*qA[3,i]-2*qA[0,i]*qA[2,i], 2*qA[2,i]*qA[3,i]+2*qA[0,i]*qA[1,i], 2*qA[0,i]**2-1+2*qA[3,i]**2]])
        return R

    def q_diff(self,qA,qB):
        try:
            NA = qA.shape[1]
        except:
            NA = 1
        if NA == 1:
            QB_conj = np.array([[qB[0],qB[1],qB[2],qB[3]],
                [qB[1],-qB[0],-qB[3],qB[2]],
                [qB[2],qB[3],-qB[0],-qB[1]],
                [qB[3],-qB[2],qB[1],-qB[0]]])
            qC = np.dot(QB_conj,qA)
            qC /= np.linalg.norm(qC)
        else:
            qC = np.zeros((4,NA))
            for i in range(NA):
                QB_conj = np.array([[qB[0,i],qB[1,i],qB[2,i],qB[3,i]],
                    [qB[1,i],-qB[0,i],-qB[3,i],qB[2,i]],
                    [qB[2,i],qB[3,i],-qB[0,i],-qB[1,i]],
                    [qB[3,i],-qB[2,i],qB[1,i],-qB[0,i]]])
                qC[:,i] = np.dot(QB_conj,qA[:,i])
                qC[:,i] /= np.linalg.norm(qC[:,i])
        return qC

    def continuous_angle(self,trace_in):
        N_trace = trace_in.shape[0]
        trace_out = trace_in
        for i in range(N_trace):
            if trace_in[i]>np.pi:
                trace_out[i] = trace_in[i]-2.0*np.pi
            elif trace_in[i]<-np.pi:
                trace_out[i] = trace_in[i]+2.0*np.pi
        return trace_out


    def thorax_roll_angle_correction(self,q_thorax,root_L,root_R):
        N_pts = q_thorax.shape[1]
        root_pts = np.transpose(np.concatenate((root_L,root_R),axis=1))
        root_cg = np.mean(root_pts,axis=0)
        root_pts = root_pts-np.matlib.repmat(root_cg,2*root_L.shape[1],1)
        pca = PCA(n_components=3)
        pca.fit(root_pts)
        q_srf_rot = np.array([np.cos(self.SRF_angle/2.0),0.0,np.sin(self.SRF_angle/2.0),0.0])
        #q_srf_now = self.q_multiply(q_srf_rot,q_thorax[:,0])
        q_srf_now = self.q_multiply(q_thorax[:,0],q_srf_rot)
        # obtain rotation matrix
        x_thorax_axis = np.squeeze(np.dot(self.R_matrix(q_srf_now),np.array([[1.0],[0.0],[0.0]])))
        y_thorax_axis = np.squeeze(np.dot(self.R_matrix(q_srf_now),np.array([[0.0],[1.0],[0.0]])))
        z_thorax_axis = np.squeeze(np.dot(self.R_matrix(q_srf_now),np.array([[0.0],[0.0],[1.0]])))
        x_SRF_axis = pca.components_[2]
        x_angle = np.arccos(np.squeeze(np.dot(x_thorax_axis,x_SRF_axis)))
        if x_angle>np.pi/2.0:
            x_SRF_axis = -x_SRF_axis
        y_SRF_axis = pca.components_[0]
        y_angle = np.arccos(np.squeeze(np.dot(y_thorax_axis,y_SRF_axis)))
        if y_angle>np.pi/2.0:
            y_SRF_axis = -y_SRF_axis
        z_SRF_axis = pca.components_[1]
        z_angle = np.arccos(np.squeeze(np.dot(z_thorax_axis,z_SRF_axis)))
        if z_angle>np.pi/2.0:
            z_SRF_axis = -z_SRF_axis
        # correct data:
        RB = np.zeros((3,3))
        RB[0,0] = x_SRF_axis[0]
        RB[0,1] = x_SRF_axis[1]
        RB[0,2] = x_SRF_axis[2]
        RB[1,0] = y_SRF_axis[0]
        RB[1,1] = y_SRF_axis[1]
        RB[1,2] = y_SRF_axis[2]
        RB[2,0] = z_SRF_axis[0]
        RB[2,1] = z_SRF_axis[1]
        RB[2,2] = z_SRF_axis[2]
        RB_trace = RB[0,0]+RB[1,1]+RB[2,2]+1.0
        if RB_trace>=0.0:
            q0 = 0.5*np.sqrt(RB_trace)
            q1 = (RB[1,2]-RB[2,1])/(4.0*q0)
            q2 = (RB[2,0]-RB[0,2])/(4.0*q0)
            q3 = (RB[0,1]-RB[1,0])/(4.0*q0)
            q_corr = np.array([q0,q1,q2,q3])
            q_corr /= np.linalg.norm(q_corr)
        else:
            q0 = 0.5*np.sqrt(np.abs(RB_trace))
            q1 = -(RB[1,2]-RB[2,1])/(4.0*q0)
            q2 = -(RB[2,0]-RB[0,2])/(4.0*q0)
            q3 = -(RB[0,1]-RB[1,0])/(4.0*q0)
            q_corr = np.array([q0,q1,q2,q3])
            q_corr /= np.linalg.norm(q_corr)
        # Apply srf rotation:
        q_srf_rot_inv = np.array([np.cos(-self.SRF_angle/2.0),0.0,np.sin(-self.SRF_angle/2.0),0.0])
        #q_thorax = self.q_multiply(q_srf_rot_inv,q_corr)
        q_thorax = self.q_multiply(q_corr,q_srf_rot_inv)
        #q_thorax = q_corr
        q_out = np.transpose(np.matlib.repmat(q_thorax,N_pts,1))
        return q_out        

    '''
    def thorax_roll_angle_correction(self,q_thorax,root_L,root_R):
        q_out = q_thorax
        N_pts = q_thorax.shape[1]
        root_pts = np.transpose(np.concatenate((root_L,root_R),axis=1))
        root_cg = np.mean(root_pts,axis=0)
        q_srf = np.array([np.cos(self.SRF_angle/2.0),0.0,np.sin(self.SRF_angle/2.0),0.0])
        pca = PCA(n_components=3)
        pca.fit(root_pts)
        #y_axis_root = np.zeros((3,1))
        y_axis_root = np.array(pca.components_[0])
        print(y_axis_root)
        #y_axis_root /= np.linalg.norm(y_axis_root)
        y_axis = np.array([[0.0],[1.0],[0.0]])
        #z_axis_root = np.zeros((3,1))
        z_axis_root = np.array(pca.components_[2])
        print(z_axis_root)
        #z_axis_root /= np.linalg.norm(z_axis_root)
        z_axis = np.array([[0.0],[0.0],[1.0]])
        for i in range(N_pts):
            #y_axis_thorax = np.dot(self.R_matrix(q_out[:,i]),y_axis)
            # cross product
            #x_corr = np.cross(np.squeeze(y_axis_root),np.squeeze(y_axis_thorax))
            #theta_x = np.arcsin(np.sum(x_corr))
            #x_dot = np.dot(np.squeeze(y_axis_root),np.squeeze(y_axis_thorax))
            #print(x_dot)
            #theta_x = np.arccos(x_dot)
            # correction quaternion
            #e_norm = np.linalg.norm(x_corr)
            #if e_norm > 0.01:
            #    e_corr = x_corr/e_norm
            #    if theta_x > np.pi/2.0:
            #        theta_x = np.pi-theta_x
            #    elif theta_x < -np.pi/2.0:
            #        theta_x = -np.pi-theta_x
            #    q_corr = np.array([np.cos(theta_x/2.0),np.sin(theta_x/2.0)*e_corr[0],np.sin(theta_x/2.0)*e_corr[1],np.sin(theta_x/2.0)*e_corr[2]])
            #    # apply correction
            #    q_out[:,i] = self.q_multiply(q_corr,q_out[:,i])
            z_axis_thorax = np.dot(self.R_matrix(q_out[:,i]),z_axis)
            # cross product
            z_corr = np.cross(np.squeeze(z_axis_root),np.squeeze(z_axis_thorax))
            z_dot = np.dot(np.squeeze(z_axis_root),np.squeeze(z_axis_thorax))
            #print(z_dot)
            theta_z = np.arccos(z_dot)
            #theta_z = np.arcsin(np.sum(z_corr))
            # correction quaternion
            e_norm = np.linalg.norm(z_corr)
            if e_norm > 0.01:
                e_corr = z_corr/e_norm
                if theta_z > np.pi/2.0:
                    theta_z = np.pi-theta_z
                elif theta_z < -np.pi/2.0:
                    theta_z = -np.pi-theta_z
                q_corr = np.array([np.cos(theta_z/2.0),np.sin(theta_z/2.0)*e_corr[0],np.sin(theta_z/2.0)*e_corr[1],np.sin(theta_z/2.0)*e_corr[2]])
                # apply correction
                q_out[:,i] = self.q_multiply(q_corr,q_out[:,i])
                q_out[:,i] = self.q_multiply(q_srf,q_out[:,i])
        return q_out
    '''

    '''
    def parse_wingbeats(self,phi_L,phi_R,peak_dist,peak_height,T_wb_min,T_wb_max,prom):
        # combine left and right wing motion take the mean:
        #phi_LR = np.zeros((phi_L.shape[0],2))
        #phi_LR[:,0] = phi_L
        #phi_LR[:,1] = phi_R
        #phi_LR = np.squeeze(np.mean(phi_LR,axis=1))
        phi_LR = phi_L # make left wing motion dominant for now
        peaks, _ = find_peaks(phi_LR,height=peak_height,distance=peak_dist,prominence=prom)
        N_peaks = peaks.shape[0]
        wb_intervals = []
        wb_t = []
        wb_f = []
        for i in range(N_peaks-1):
            T_wb_i = peaks[i+1]-peaks[i]
            if T_wb_i>T_wb_min & T_wb_i<T_wb_max:
                wb_intervals.append([peaks[i],peaks[i+1]])
                wb_t.append(T_wb_i*self.dt)
                f_i = 1.0/(T_wb_i*self.dt)
                wb_f.append(f_i)
        return wb_intervals, wb_t, wb_f
    '''

    def parse_wingbeats(self,phi_L,phi_R,root_L,root_R,peak_dist,T_wb_min,T_wb_max):
        # central difference filter of phi_L, phi_R, root_L, root_R
        phi_L     = savgol_filter(phi_L,21,3)
        phi_R     = savgol_filter(phi_R,21,3)
        root_L     = savgol_filter(root_L,21,3)
        root_R     = savgol_filter(root_R,21,3)
        phi_LR_plus = ((phi_L>self.phi_thresh)&(phi_R>self.phi_thresh))
        root_y_L = -root_L+np.mean(root_L)
        root_y_R = root_R-np.mean(root_R)
        root_LR = np.multiply(phi_LR_plus,np.abs(root_y_L+root_y_R))
        #peaks, _ = find_peaks(root_LR,distance=peak_dist)
        peaks, _ = find_peaks(phi_L,distance=peak_dist)
        N_peaks = peaks.shape[0]
        wb_intervals = []
        wb_t = []
        wb_f = []
        for i in range(N_peaks-1):
            T_wb_i = peaks[i+1]-peaks[i]
            if T_wb_i>T_wb_min & T_wb_i<T_wb_max:
                wb_intervals.append([peaks[i],peaks[i+1]])
                wb_t.append(T_wb_i*self.dt)
                f_i = 1.0/(T_wb_i*self.dt)
                wb_f.append(f_i)
        return wb_intervals, wb_t, wb_f

    def Euler_x(self,RA):
        x_axis = np.array([[1.0],[0.0],[0.0]])
        y_axis = np.array([[0.0],[1.0],[0.0]])
        z_axis = np.array([[0.0],[0.0],[1.0]])
        try:
            NA = RA.shape[2]
        except:
            NA = 1
        if NA == 1:
            angles_out = np.zeros(3)
            x_r = np.squeeze(np.dot(RA,x_axis))
            phi = np.arctan2(x_r[1],x_r[0])
            theta = np.arcsin(x_r[2])
            R_phi = np.array([[np.cos(phi),-np.sin(phi),0.0],[np.sin(phi),np.cos(phi),0.0],[0.0,0.0,1.0]])
            R_theta = np.array([[np.cos(theta),0.0,np.sin(theta)],[0.0,1.0,0.0],[-np.sin(theta),0.0,np.cos(theta)]])
            R_eta = np.dot(np.transpose(R_phi),np.dot(np.transpose(R_theta),RA))
            eta = np.arctan2(R_eta[2,1],R_eta[2,2])
            angles_out[0] = phi
            angles_out[1] = theta
            angles_out[2] = eta
        else:
            angles_out = np.zeros((3,NA))
            for i in range(NA):
                x_r = np.squeeze(np.dot(RA[:,:,i],x_axis))
                phi = np.arctan2(x_r[1],x_r[0])
                theta = np.arcsin(x_r[2])
                R_phi = np.array([[np.cos(phi),-np.sin(phi),0.0],[np.sin(phi),np.cos(phi),0.0],[0.0,0.0,1.0]])
                R_theta = np.array([[np.cos(theta),0.0,np.sin(theta)],[0.0,1.0,0.0],[-np.sin(theta),0.0,np.cos(theta)]])
                R_eta = np.dot(np.transpose(R_phi),np.dot(np.transpose(R_theta),RA[:,:,i]))
                eta = np.arctan2(R_eta[2,1],R_eta[2,2])
                angles_out[0,i] = phi
                angles_out[1,i] = theta
                angles_out[2,i] = eta
            #angles_out[0,:] = self.continuous_angle(angles_out[0,:])
            #angles_out[1,:] = self.continuous_angle(angles_out[1,:])
            #angles_out[2,:] = self.continuous_angle(angles_out[2,:])
        return angles_out

    def Euler_yL(self,RA):
        x_axis = np.array([[1.0],[0.0],[0.0]])
        y_axis = np.array([[0.0],[1.0],[0.0]])
        z_axis = np.array([[0.0],[0.0],[1.0]])
        try:
            NA = RA.shape[2]
        except:
            NA = 1
        if NA == 1:
            angles_out = np.zeros(3)
            y_r = np.squeeze(np.dot(RA,y_axis))
            phi = np.arctan2(-y_r[0],y_r[1])
            theta = np.arcsin(y_r[2])
            R_phi = np.array([[np.cos(phi),-np.sin(phi),0.0],[np.sin(phi),np.cos(phi),0.0],[0.0,0.0,1.0]])
            R_theta = np.array([[1.0,0.0,0.0],[0.0,np.cos(theta),-np.sin(theta)],[0.0,np.sin(theta),np.cos(theta)]])
            R_eta = np.dot(np.transpose(R_phi),np.dot(np.transpose(R_theta),RA))
            eta = np.arctan2(R_eta[0,2],R_eta[0,0])+np.pi/2.0
            angles_out[0] = phi
            angles_out[1] = theta
            angles_out[2] = eta
        else:
            angles_out = np.zeros((3,NA))
            for i in range(NA):
                y_r = np.squeeze(np.dot(RA[:,:,i],y_axis))
                phi = np.arctan2(-y_r[0],y_r[1])
                theta = np.arcsin(y_r[2])
                R_phi = np.array([[np.cos(phi),-np.sin(phi),0.0],[np.sin(phi),np.cos(phi),0.0],[0.0,0.0,1.0]])
                R_theta = np.array([[1.0,0.0,0.0],[0.0,np.cos(theta),-np.sin(theta)],[0.0,np.sin(theta),np.cos(theta)]])
                R_eta = np.dot(np.transpose(R_phi),np.dot(np.transpose(R_theta),RA[:,:,i]))
                eta = np.arctan2(R_eta[0,2],R_eta[0,0])+np.pi/2.0
                angles_out[0,i] = phi
                angles_out[1,i] = theta
                angles_out[2,i] = eta
            angles_out[0,:] = self.continuous_angle(angles_out[0,:])
            angles_out[1,:] = self.continuous_angle(angles_out[1,:])
            angles_out[2,:] = self.continuous_angle(angles_out[2,:])
        return angles_out

    def Euler_yR(self,RA):
        x_axis = np.array([[1.0],[0.0],[0.0]])
        y_axis = np.array([[0.0],[-1.0],[0.0]])
        z_axis = np.array([[0.0],[0.0],[1.0]])
        try:
            NA = RA.shape[2]
        except:
            NA = 1
        if NA == 1:
            angles_out = np.zeros(3)
            y_r = np.squeeze(np.dot(RA,y_axis))
            phi = np.arctan2(-y_r[0],-y_r[1])
            theta = np.arcsin(y_r[2])
            R_phi = np.array([[np.cos(-phi),-np.sin(-phi),0.0],[np.sin(-phi),np.cos(-phi),0.0],[0.0,0.0,1.0]])
            R_theta = np.array([[1.0,0.0,0.0],[0.0,np.cos(-theta),-np.sin(-theta)],[0.0,np.sin(-theta),np.cos(-theta)]])
            R_eta = np.dot(np.transpose(R_phi),np.dot(np.transpose(R_theta),RA))
            #eta = np.arctan2(R_eta[2,0],R_eta[2,2])+np.pi/2.0
            eta = np.arctan2(R_eta[0,2],R_eta[0,0])+np.pi/2.0
            angles_out[0] = phi
            angles_out[1] = theta
            angles_out[2] = eta
        else:
            angles_out = np.zeros((3,NA))
            for i in range(NA):
                y_r = np.squeeze(np.dot(RA[:,:,i],y_axis))
                phi = np.arctan2(-y_r[0],-y_r[1])
                theta = np.arcsin(y_r[2])
                R_phi = np.array([[np.cos(-phi),-np.sin(-phi),0.0],[np.sin(-phi),np.cos(-phi),0.0],[0.0,0.0,1.0]])
                R_theta = np.array([[1.0,0.0,0.0],[0.0,np.cos(-theta),-np.sin(-theta)],[0.0,np.sin(-theta),np.cos(-theta)]])
                R_eta = np.dot(np.transpose(R_phi),np.dot(np.transpose(R_theta),RA[:,:,i]))
                #eta = -np.arctan2(R_eta[2,0],R_eta[2,2])+np.pi/2.0
                eta = np.arctan2(R_eta[0,2],R_eta[0,0])+np.pi/2.0
                angles_out[0,i] = phi
                angles_out[1,i] = theta
                angles_out[2,i] = eta
            angles_out[0,:] = self.continuous_angle(angles_out[0,:])
            angles_out[1,:] = self.continuous_angle(angles_out[1,:])
            angles_out[2,:] = self.continuous_angle(angles_out[2,:])
        return angles_out

    def LegendreFit(self,trace_in,b1_in,b2_in,N_pol,N_const):
        N_pts = trace_in.shape[0]
        X_Legendre = self.LegendrePolynomials(N_pts,N_pol,N_const)
        A = X_Legendre[:,:,0]
        B = np.zeros((2*N_const,N_pol))
        B[:N_const,:] = np.transpose(X_Legendre[0,:,:])
        B[N_const:,:] = np.transpose(X_Legendre[-1,:,:])
        # data points:
        b = np.transpose(trace_in)
        # restriction vector (add zeros to smooth the connection!!!!!)
        d = np.zeros(2*N_const)
        d_gradient_1 = b1_in
        d_gradient_2 = b2_in
        for j in range(N_const):
            d[j] = d_gradient_1[3-j]*np.power(N_pts/2.0,j)
            d[N_const+j] = d_gradient_2[3-j]*np.power(N_pts/2.0,j)
            d_gradient_1 = np.diff(d_gradient_1)
            d_gradient_2 = np.diff(d_gradient_2)
        # Restricted least-squares fit:
        ATA = np.dot(np.transpose(A),A)
        ATA_inv = np.linalg.inv(ATA)
        AT = np.transpose(A)
        BT = np.transpose(B)
        BATABT     = np.dot(B,np.dot(ATA_inv,BT))
        c_ls     = np.linalg.solve(ATA,np.dot(AT,b))
        c_rls     = c_ls-np.dot(ATA_inv,np.dot(BT,np.linalg.solve(BATABT,np.dot(B,c_ls)-d)))
        return c_rls

    def LegendrePolynomials(self,N_pts,N_pol,n_deriv):
        L_basis = np.zeros((N_pts,N_pol,n_deriv))
        x_basis = np.linspace(-1.0,1.0,N_pts,endpoint=True)
        for i in range(n_deriv):
            if i==0:
                # Legendre basis:
                for n in range(N_pol):
                    if n==0:
                        L_basis[:,n,i] = 1.0
                    elif n==1:
                        L_basis[:,n,i] = x_basis
                    else:
                        for k in range(n+1):
                            L_basis[:,n,i] += (1.0/np.power(2.0,n))*np.power(scipy.special.binom(n,k),2)*np.multiply(np.power(x_basis-1.0,n-k),np.power(x_basis+1.0,k))
            else:
                # Derivatives:
                for n in range(N_pol):
                    if n>=i:
                        L_basis[:,n,i] = n*L_basis[:,n-1,i-1]+np.multiply(x_basis,L_basis[:,n-1,i])
        return L_basis

    def TemporalBC(self,a_c,N_pol,N_const):
        X_Legendre = self.LegendrePolynomials(100,N_pol,N_const)
        trace = np.dot(X_Legendre[:,:,0],a_c)
        b_L = np.zeros(9)
        b_L[0:4] = trace[-5:-1]
        b_L[4] = 0.5*(trace[0]+trace[-1])
        b_L[5:9] = trace[1:5]
        b_R = np.zeros(9)
        b_R[0:4] = trace[-5:-1]
        b_R[4] = 0.5*(trace[0]+trace[-1])
        b_R[5:9] = trace[1:5]
        c_per = self.LegendreFit(trace,b_L,b_R,N_pol,N_const)
        return c_per

    def set_find_outliers_btn(self,btn_in):
        self.find_outliers_btn = btn_in
        self.find_outliers_btn.clicked.connect(self.find_outliers)

    def find_outliers(self):
        # Subtract filtered traces from raw traces
        # Compute mean and standard deviation
        # Select all frames exceeding 6 times the standard deviation
        self.outlier_frames = []
        self.outlier_frames_mat = np.zeros((1,3))
        self.outlier_frames_mat.astype(int)
        for i in range(self.N_mov):
            n_seq = len(self.filtered_data[i])
            outliers_seq = []
            for j in range(n_seq):
                # get nr of frames:
                n_frames = self.filtered_data[i][j][0].shape[1]
                raw_data = np.zeros((37,n_frames))
                filt_data = np.zeros((37,n_frames))
                for k in range(5):
                    if k==0:
                        raw_data[0:4,:] = self.continuous_quaternions(self.fit_data[i][j][0:4,:])
                        raw_data[4:7,:] = self.fit_data[i][j][4:7,:]
                        filt_data[0:7,:] = self.filtered_data[i][j][k][0:7,:]
                    elif k==1:
                        raw_data[7:11,:] = self.continuous_quaternions(self.fit_data[i][j][7:11,:])
                        raw_data[11:14,:] = self.fit_data[i][j][11:14,:]
                        filt_data[7:14,:] = self.filtered_data[i][j][k][0:7,:]
                    elif k==2:
                        raw_data[14:18,:] = self.continuous_quaternions(self.fit_data[i][j][14:18,:])
                        raw_data[18:21,:] = self.fit_data[i][j][18:21,:]
                        filt_data[14:21,:] = self.filtered_data[i][j][k][0:7,:]
                    elif k==3:
                        raw_data[21:25,:] = self.continuous_quaternions(self.fit_data[i][j][21:25,:])
                        raw_data[25:29,:] = self.fit_data[i][j][25:29,:]
                        filt_data[21:29,:] = self.filtered_data[i][j][k][0:8,:]
                    elif k==4:
                        raw_data[29:33,:] = self.continuous_quaternions(self.fit_data[i][j][29:33,:])
                        raw_data[33:37,:] = self.fit_data[i][j][33:37,:]
                        filt_data[29:37,:] = self.filtered_data[i][j][k][0:8,:]
                raw_m_filt_z = np.abs(zscore(raw_data-filt_data,axis=1))
                outlier_inds = np.sum((raw_m_filt_z>6.0),axis=0)
                frame_range = np.arange(n_frames)
                outliers = frame_range[outlier_inds>0]
                n_outliers = len(outliers)
                print(outliers)
                print(n_outliers)
                outliers_seq.append(outliers)
                if n_outliers>0:
                    outlier_mat = np.zeros((n_outliers,3))
                    outlier_mat[:,0] = i+1
                    outlier_mat[:,1] = j+1
                    outlier_mat[:,2] = np.array(outliers)
                    self.outlier_frames_mat = np.append(self.outlier_frames_mat,outlier_mat,axis=0)
            self.outlier_frames.append(outliers_seq)
        # Update table
        self.update_fix_table()
        self.update_outlier_frame_table()
        self.update_outlier_spin()

    def set_fix_frames_table(self,table_in):
        self.fix_frames_table = table_in
        self.fix_frames_table.setRowCount(1)
        self.fix_frames_table.setColumnCount(3)
        self.fix_frames_table.setItem(0,0,QTableWidgetItem('mov nr:'))
        self.fix_frames_table.setItem(0,1,QTableWidgetItem('seq nr:'))
        self.fix_frames_table.setItem(0,2,QTableWidgetItem('frame nr:'))

    def update_fix_table(self):
        n_outliers = self.outlier_frames_mat.shape[0]
        self.fix_frames_table.setRowCount(n_outliers)
        self.fix_frames_table.setColumnCount(3)
        self.fix_frames_table.setItem(0,0,QTableWidgetItem('mov nr:'))
        self.fix_frames_table.setItem(0,1,QTableWidgetItem('seq nr:'))
        self.fix_frames_table.setItem(0,2,QTableWidgetItem('frame nr:'))
        if n_outliers>1:
            for i in range(1,n_outliers):
                self.fix_frames_table.setItem(i,0,QTableWidgetItem(str(self.outlier_frames_mat[i,0])))
                self.fix_frames_table.setItem(i,1,QTableWidgetItem(str(self.outlier_frames_mat[i,1])))
                self.fix_frames_table.setItem(i,2,QTableWidgetItem(str(self.outlier_frames_mat[i,2])))
        self.fix_frames_table.resizeColumnsToContents()

    def set_add_outlier_frame_btn(self,btn_in):
        self.add_outlier_frame_btn = btn_in
        self.add_outlier_frame_btn.clicked.connect(self.add_outlier_frame)

    def add_outlier_frame(self):
        frame_append = np.zeros((1,3))
        frame_append.astype(int)
        frame_append[0,0] = self.mov_nr+1
        frame_append[0,1] = self.seq_nr+1
        frame_append[0,2] = self.frame_nr
        self.outlier_frames_mat = np.append(self.outlier_frames_mat,frame_append,axis=0)
        # Update table
        self.update_fix_table()
        self.update_outlier_frame_table()
        self.update_outlier_spin()

    def set_outlier_fix_progress(self,progress_in):
        self.outlier_fix_progress = progress_in
        self.outlier_fix_progress.setValue(0)

    def set_outlier_frame_table(self,table_in):
        self.outlier_frame_table = table_in
        self.outlier_frame_table.setRowCount(1)
        self.outlier_frame_table.setColumnCount(4)
        self.outlier_frame_table.setItem(0,0,QTableWidgetItem('outlier:'))
        self.outlier_frame_table.setItem(0,1,QTableWidgetItem('mov nr:'))
        self.outlier_frame_table.setItem(0,2,QTableWidgetItem('seq nr:'))
        self.outlier_frame_table.setItem(0,3,QTableWidgetItem('frame nr:'))

    def update_outlier_frame_table(self):
        n_outliers = self.outlier_frames_mat.shape[0]
        self.outlier_frame_table.setRowCount(n_outliers)
        self.outlier_frame_table.setColumnCount(4)
        self.outlier_frame_table.setItem(0,0,QTableWidgetItem('outlier:'))
        self.outlier_frame_table.setItem(0,1,QTableWidgetItem('mov nr:'))
        self.outlier_frame_table.setItem(0,2,QTableWidgetItem('seq nr:'))
        self.outlier_frame_table.setItem(0,3,QTableWidgetItem('frame nr:'))
        if n_outliers>1:
            for i in range(1,n_outliers):
                self.outlier_frame_table.setItem(i,0,QTableWidgetItem(str(i)))
                self.outlier_frame_table.setItem(i,1,QTableWidgetItem(str(self.outlier_frames_mat[i,0])))
                self.outlier_frame_table.setItem(i,2,QTableWidgetItem(str(self.outlier_frames_mat[i,1])))
                self.outlier_frame_table.setItem(i,3,QTableWidgetItem(str(self.outlier_frames_mat[i,2])))
        self.outlier_frame_table.resizeColumnsToContents()

    def set_auto_fix_btn(self,btn_in):
        self.auto_fix_btn = btn_in
        self.auto_fix_btn.clicked.connect(self.fix_outliers)

    def set_model(self,mdl_in):
        self.mdl = mdl_in

    def set_img_viewer(self,viewer_in):
        self.img_viewer = viewer_in

    def fix_outliers(self):
        self.outlier_fix_progress.setValue(0)
        n_outliers = self.outlier_frames_mat.shape[0]
        if n_outliers>1:
            for i in range(1,n_outliers):
                out_mov_nr = int(self.outlier_frames_mat[i,0]-1)
                out_seq_nr = int(self.outlier_frames_mat[i,1]-1)
                out_frame_nr = int(self.outlier_frames_mat[i,2])
                print('outlier: '+str(i)+', mov nr: '+str(out_mov_nr)+', seq nr: '+str(out_seq_nr)+', frame_nr: '+str(out_frame_nr))
                if i==1:
                    state_i = self.mdl.fit_outlier(out_mov_nr,out_seq_nr,out_frame_nr,True)
                else:
                    state_i = self.mdl.fit_outlier(out_mov_nr,out_seq_nr,out_frame_nr,False)
                # Overwrite frame in hdf5
                self.fit_data[out_mov_nr-1][out_seq_nr][:,out_frame_nr-1] = np.squeeze(state_i)
                self.outlier_fix_progress.setValue(int(100.0*(i-1)/(n_outliers-2)))
        else:
            print('no outliers')

    def set_update_outlier_btn(self,btn_in):
        self.update_outlier_btn = btn_in

    def set_outlier_spin(self,spin_in):
        self.outlier_spin = spin_in

    def update_outlier_spin(self):
        n_outliers = self.outlier_frames_mat.shape[0]
        self.outlier_spin.setMinimum(1)
        self.outlier_spin.setMaximum(n_outliers)
        self.outlier_spin.setValue(1)
        self.outlier_nr = 1
        self.outlier_spin.valueChanged.connect(self.set_outlier_now)

    def set_outlier_now(self,outlier_ind):
        self.outlier_nr = outlier_ind-1
        out_mov_nr = int(self.outlier_frames_mat[self.outlier_nr,0])
        out_seq_nr = int(self.outlier_frames_mat[self.outlier_nr,1]-1)
        out_frame_nr = int(self.outlier_frames_mat[self.outlier_nr,2]+1)
        print('outlier: '+str(self.outlier_nr)+', mov nr: '+str(out_mov_nr)+', seq nr: '+str(out_seq_nr)+', frame_nr: '+str(out_frame_nr))
        self.img_viewer.set_movie_nr(out_mov_nr)
        self.img_viewer.set_seq_nr(out_seq_nr)
        self.img_viewer.load_frame(out_frame_nr)

# -------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    wingkin_data_loc = '/media/flythreads/FlyamiDataB/flyami/results'

    wingkin_files = [
        'seqs_Session_08_12_2020_14_53.h5py',
        'seqs_Session_04_01_2021_16_14.h5py',
        'seqs_Session_04_01_2021_16_30.h5py',
        'seqs_Session_08_02_2021_15_14.h5py',
        'seqs_Session_17_02_2021_13_18.h5py',
        'seqs_Session_19_02_2021_11_49.h5py',
        'seqs_Session_02_03_2021_11_54.h5py',     # 7
        'seqs_Session_28_10_2020_13_18.h5py',     # 8
        'seqs_Session_06_11_2020_10_55.h5py',     # 9
        'seqs_Session_13_11_2020_11_31.h5py',     # 10
        'seqs_Session_17_11_2020_12_38.h5py',     # 11
        'seqs_Session_06_01_2021_14_02.h5py',     # 12
        'seqs_Session_10_02_2021_12_59.h5py',     # 13
        'seqs_Session_05_03_2021_14_32.h5py',     # 14
        'seqs_Session_31_03_2021_16_07.h5py',     # 15
        'seqs_Session_02_11_2020_11_15.h5py',     # 16
        'seqs_Session_08_12_2020_14_40.h5py',     # 17
        'seqs_Session_05_01_2021_15_21.h5py',      # 18
        'seqs_Session_19_02_2021_13_00.h5py',     # 19
        'seqs_Session_22_02_2021_13_45.h5py',     # 20
        'seqs_Session_02_03_2021_12_50.h5py',     # 21
        'seqs_Session_04_03_2021_11_17.h5py',     # 22
        'seqs_Session_04_03_2021_15_06.h5py',     # 23
        'seqs_Session_28_10_2020_11_58.h5py',     # 24
        'seqs_Session_06_11_2020_11_36.h5py',     # 25
        'seqs_Session_05_01_2021_14_55.h5py',     # 26
        'seqs_Session_04_02_2021_12_44.h5py',     # 27
        'seqs_Session_08_02_2021_14_31.h5py',     # 28
        'seqs_Session_27_02_2021_13_19.h5py',     # 29
        'seqs_Session_01_03_2021_14_58.h5py',     # 30
        'seqs_Session_03_03_2021_14_57.h5py',     # 31
        'seqs_Session_11_03_2021_10_51.h5py',     # 32
        'seqs_Session_08_12_2020_15_30.h5py',     # 33
        'seqs_Session_17_02_2021_14_16.h5py',     # 34
        'seqs_Session_22_02_2021_12_45.h5py',     # 35
        'seqs_Session_27_02_2021_14_57.h5py',     # 36
        'seqs_Session_27_02_2021_15_12.h5py',     # 37
        'seqs_Session_22_03_2021_11_19.h5py',     # 38
        'seqs_Session_24_03_2021_13_39.h5py',     # 39
        'seqs_Session_08_12_2020_16_14.h5py',     # 40
        'seqs_Session_13_11_2020_10_23.h5py',     # 41
        'seqs_Session_17_11_2020_14_34.h5py',     # 42
        'seqs_Session_17_02_2021_12_27.h5py',     # 43
        'seqs_Session_01_03_2021_11_14.h5py',    # 44
        'seqs_Session_22_03_2021_12_23.h5py',     # 45
        'seqs_Session_08_11_2020_11_56.h5py',     # 46
        'seqs_Session_17_11_2020_13_51.h5py',     # 47
        'seqs_Session_04_01_2021_14_42.h5py',     # 48
        'seqs_Session_06_01_2021_15_32.h5py',     # 49
        'seqs_Session_07_01_2021_14_17.h5py',     # 50
        'seqs_Session_04_02_2021_13_51.h5py',    # 51
        'seqs_Session_19_02_2021_15_47.h5py',     # 52
        'seqs_Session_22_02_2021_11_36.h5py',     # 53
        'seqs_Session_25_02_2021_13_35.h5py',    # 54
        'seqs_Session_25_02_2021_14_05.h5py',     # 55
        'seqs_Session_25_02_2021_14_05.h5py',     # 56
        'seqs_Session_28_10_2020_14_05.h5py',    # 57
        'seqs_Session_08_11_2020_12_24.h5py',     # 58
        'seqs_Session_15_01_2021_15_35.h5py',     # 59
        'seqs_Session_18_02_2021_14_16.h5py',    # 60
        'seqs_Session_04_03_2021_12_45.h5py',    # 61
        'seqs_Session_05_03_2021_12_59.h5py',    # 62
        'seqs_Session_11_03_2021_11_51.h5py',
        'seqs_Session_22_03_2021_13_10.h5py',
        'seqs_Session_23_03_2021_13_19.h5py',
        'seqs_Session_04_11_2020_13_43.h5py',
        'seqs_Session_05_01_2021_13_40.h5py',
        'seqs_Session_15_01_2021_16_34.h5py',
        'seqs_Session_10_02_2021_12_08.h5py',
        'seqs_Session_01_03_2021_11_42.h5py',
        'seqs_Session_03_03_2021_15_32.h5py',
        'seqs_Session_13_03_2021_13_53.h5py',
        'seqs_Session_24_03_2021_14_10.h5py',
        'seqs_Session_08_11_2020_11_24.h5py', 
        'seqs_Session_04_01_2021_15_30.h5py', 
        'seqs_Session_05_01_2021_14_17.h5py', 
        'seqs_Session_07_01_2021_15_55.h5py',
        'seqs_Session_08_02_2021_15_51.h5py',
        'seqs_Session_01_03_2021_14_26.h5py',
        'seqs_Session_05_03_2021_11_35.h5py', 
        'seqs_Session_23_03_2021_13_59.h5py', 
        'seqs_Session_06_11_2020_12_16.h5py', 
        'seqs_Session_04_02_2021_11_58.h5py', 
        'seqs_Session_22_02_2021_15_57.h5py', 
        'seqs_Session_05_03_2021_10_53.h5py', 
        'seqs_Session_11_03_2021_12_18.h5py', 
        'seqs_Session_31_03_2021_14_20.h5py'
    ]

    post_process = PostProcessing()
    post_process.set_SRF_angle(-45)
    post_process.set_phi_thresh(45)
    N_files = len(wingkin_files)
    print('N_files: '+str(N_files))
    for i in range(N_files):
        print('file name:')
        print(wingkin_files[i])
        post_process.load_seq_file(wingkin_data_loc,wingkin_files[i])
        post_process.load_data()
        post_process.filter_data()
        post_process.fit_wbs()
        post_process.close_seq_file()
        print('done')
