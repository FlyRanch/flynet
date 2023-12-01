from __future__ import print_function
import sys
import vtk
from PyQt5 import QtGui
from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidget, QTableWidgetItem, QVBoxLayout, QFileDialog
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
#import pickle
import os
import os.path
import pathlib
import math
import copy
import time
import h5py

import pySciCam
from pySciCam.pySciCam import ImageSequence
import glob
import matplotlib.image as mpimg
import re
from scipy.io import loadmat

from .PostProcessing import PostProcessing
from .session_select import Ui_Session_Dialog
from .fly_net_ui import Ui_MainWindow
from .drosophila import model as drosophila 
from . import FlyNet

class CheckableDirModel(QtWidgets.QDirModel):
    def __init__(self, parent=None):
        QtGui.QDirModel.__init__(self, None)
        self.checks = {}

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.CheckStateRole:
            return QtGui.QDirModel.data(self, index, role)
        else:
            if index.column() == 0:
                return self.checkState(index)

    def flags(self, index):
        return QtGui.QDirModel.flags(self, index) | QtCore.Qt.ItemIsUserCheckable

    def checkState(self, index):
        if index in self.checks:
            return self.checks[index]
        else:
            return QtCore.Qt.Unchecked

    def setData(self, index, value, role):
        if (role == QtCore.Qt.CheckStateRole and index.column() == 0):
            self.checks[index] = value
            self.emit(QtCore.SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
            return True 

        return QtGui.QDirModel.setData(self, index, value, role)

# QDialog clasess:

class SelectFolderWindow(QtWidgets.QDialog, Ui_Session_Dialog):

    def __init__(self, directory, parent=None):
        super(SelectFolderWindow,self).__init__(parent)
        self.setupUi(self)
        self.folder_name = None
        self.folder_path = None
        self.file_model = QFileSystemModel()
        self.directory = directory
        self.file_model.setRootPath(directory)
        self.folder_tree.setModel(self.file_model)
        self.folder_tree.setRootIndex(self.file_model.index(self.directory));
        self.folder_tree.clicked.connect(self.set_session_folder)

    def update_file_model(self,new_dir):
        self.directory = new_dir
        self.file_model.setRootPath(new_dir)

    def set_session_folder(self, index):
        indexItem = self.file_model.index(index.row(), 0, index.parent())
        self.folder_name = self.file_model.fileName(indexItem)
        self.folder_path = self.file_model.filePath(indexItem)
        self.selected_session.setText(self.folder_path)

class FlyNetViewer(QtWidgets.QMainWindow, Ui_MainWindow, QObject):

    def __init__(self, parent=None):
        super(FlyNetViewer,self).__init__(parent)
        self.setupUi(self)

        self.ds = 0.040
        self.cam_mag = 0.5
        self.frame_size = [[256,256],[256,256],[256,256]]
        self.frame_range = [0,16375]

        self.scale_set = False
        self.state_L = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        self.state_R = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        self.scale_L = 1.0
        self.scale_R = 1.0

        self.annotation_active = False

        self.load_data_gui()
        self.inactive_frame_gui()

        self.tabWidget.setTabEnabled(1,False)
        self.tabWidget.setTabEnabled(2,False)

    #-----------------------------------------------------------------------
    #
    #    Data loader:
    #
    #-----------------------------------------------------------------------

    def load_data_gui(self):

        #self.base_dir = '/home/flynet/Documents/Flyami_movies'
        #calib_fldr = '/home/flynet/Documents/FlyNet4/data'
        #self.select_calib_window = SelectFolderWindow(calib_fldr)
        #self.mdl_dir = '/home/flynet/Documents/FlyNet4/models'
        #self.network_dir = '/home/flynet/Documents/FlyNet4/networks'

        self.home_dir = pathlib.Path().home()
        self.flynet_dir  = self.home_dir   / 'flynet'
        self.movies_dir  = self.flynet_dir / 'movies'
        self.save_dir    = self.flynet_dir / 'save'
        self.calib_dir   = self.flynet_dir / 'calibrations'
        self.network_dir = self.flynet_dir / 'network'
        self.weights_dir = self.flynet_dir / 'weights' 
        self.mdl_dir     = self.flynet_dir / 'models'

        self.select_session_window = SelectFolderWindow(str(self.movies_dir))
        self.select_session_window.setWindowTitle("Select session folder")

        self.select_calib_window = SelectFolderWindow(str(self.calib_dir))
        self.select_calib_window.setWindowTitle("Select calibration file")

        self.select_movie_window = SelectFolderWindow(str(self.movies_dir))
        self.select_movie_window.setWindowTitle("Select movie folder")

        self.select_save_window = SelectFolderWindow(str(self.save_dir))
        self.select_save_window.setWindowTitle("Select save folder")

        self.select_model_window = SelectFolderWindow(str(self.mdl_dir))
        self.select_model_window.setWindowTitle("Select model folder")

        #self.trig_modes = ['...','start','center','end']
        self.trig_modes = ['...','start','center','end']
        self.network_options = ['...','FlyNet']

        # Parameters
        self.session_path = None
        self.session_folder = None
        self.bckg_path = None
        self.bckg_folder = None
        self.bckg_frames = []
        self.bckg_img_format = None
        self.calib_path = None
        self.calib_file = None
        self.N_cam = None
        self.cam_folders = []
        self.frame_name = None
        self.frame_img_format = None
        self.N_mov = None
        self.mov_folder_list = []
        self.trigger_mode = None
        self.start_frame = None
        self.trigger_frame = None
        self.end_frame = None
        self.N_frames_list = []
        # Additional parameters
        # Start GUI
        self.select_seq_btn.clicked.connect(self.select_session_callback)
        self.seq_folder_disp.setText('...')
        #self.select_bckg_btn.clicked.connect(self.select_bckg_callback)
        #self.bckg_folder_disp.setText('...')
        self.select_calib_btn.clicked.connect(self.select_calib_callback)
        self.calib_folder_disp.setText('...')
        self.load_seq_btn.clicked.connect(self.load_seq_callback)
        self.set_table_widget(self.movie_table)
        self.set_trigger_mode_box(self.trigger_mode_combo)
        self.select_save_fldr_btn.clicked.connect(self.select_save_fldr_callback)
        self.img_viewer_1.set_ses_calc_progress(self.crop_seq_calc_progress)



    def select_session_callback(self):
        self.select_session_window.exec_()
        # in case h5py file is selected move on to results selection
        check_file_name = str(self.select_session_window.folder_name)
        if '.h5py' in check_file_name:
            results_file = check_file_name
            ses_folder = check_file_name.replace('seqs_','').replace('.h5py','')
            self.session_folder = ses_folder
            ses_path = str(self.select_session_window.folder_path)
            self.session_path = ses_path.replace(results_file,'')
            print(results_file)
            print(self.session_folder)
            print(self.session_path)
            self.seq_folder_disp.setText(self.session_folder)
            # get keys:
            os.chdir(self.session_path)
            data = h5py.File(results_file,'r')
            key_list = list(data.keys())
            frame_keys = [i for i in key_list if 'frame' in i]
            data.close()
            # iterate over frames and find how many movies and cameras are present:
            self.N_mov = 0
            self.N_cam = 0
            self.start_frame = 0
            self.end_frame = 0
            for f_k in frame_keys:
                f_k_split = f_k.split('_')
                if int(f_k_split[2])>self.N_mov:
                    self.N_mov = int(f_k_split[2])
                if int(f_k_split[8])>self.N_cam:
                    self.N_cam = int(f_k_split[8])
                if int(f_k_split[4])>self.end_frame:
                    self.end_frame = int(f_k_split[4])
            print('N_movies: '+str(self.N_mov))
            print('N_cam: '+str(self.N_cam))
            print('start frame: ' + str(self.start_frame))
            print('end frame: ' + str(self.end_frame))
            self.mov_folder_list = []
            for i in range(self.N_mov):
                mov_folder = 'H001S'+str(i+1).zfill(4)
                self.mov_folder_list.append(mov_folder)
            self.mov_folder_list.sort()
            print(self.mov_folder_list)
            self.cam_folders = []
            for j in range(self.N_cam):
                cam_folder = 'C'+str(j+1).zfill(3)
                self.cam_folders.append(cam_folder)
            self.cam_folders.sort()
            print(self.cam_folders)
            mov_loaded = True
        else:
            self.session_path = str(self.select_session_window.folder_path)
            print(self.session_path)
            self.session_folder = str(self.select_session_window.folder_name)
            self.seq_folder_disp.setText(self.session_folder)
            self.select_calib_window.update_file_model(self.session_path)
        self.img_loader()
        self.update_table_widget()

    def select_calib_callback(self):
        self.select_calib_window.exec_()
        self.calib_path = self.select_calib_window.folder_path
        if self.calib_path is None:
            return
        print(self.calib_path)
        for root, dirs, files in os.walk(self.calib_path):
            print(root, dirs, files)
            if len(files)>0:
                for file in files:
                    file_name = os.path.splitext(file)[0]
                    file_ext = os.path.splitext(file)[1]
                    if file_ext == '.mat':
                        if file_name == 'DLT_coeff':
                            self.calib_file = file_name
                    elif file_ext == '.txt':
                        if file_name == 'cam_calib':
                            self.calib_file = file_name
        print(self.calib_file)
        if self.calib_file == 'DLT_coeff':
            self.calib_folder_disp.setText(self.calib_file)
            print('load calibration')
            self.LoadCalibrationMatrix('.mat')
            print('calculate projection matrices')
            print('load camera calibration')
            self.img_viewer_1.load_camera_calibration(self.c_params,0)
        elif self.calib_file == 'cam_calib':
            self.calib_folder_disp.setText(self.calib_file)
            print('load calibration')
            self.LoadCalibrationMatrix('.txt')
            print('calculate projection matrices')
            print('load camera calibration')
            self.img_viewer_1.load_camera_calibration(self.c_params,1)
        else:
            print('error: could not find calibration file')
            self.calib_folder_disp.setText('error!')

    def select_save_fldr_callback(self):
        self.select_save_window.exec_()
        self.save_fldr_path = self.select_save_window.folder_path
        print(self.save_fldr_path)
        self.save_fldr_disp.setText(self.save_fldr_path)
        self.img_viewer_1.set_save_fldr(self.save_fldr_path)

    def img_loader(self):
        mov_loaded = False
        try:
            if not self.mov_folder_list:
                # discover file structure:
                for root, dirs, files in os.walk(self.session_path):
                    if len(dirs)>0:
                        for folder in dirs:
                            if 'H001' in folder:
                                if 'C001' in folder:
                                    mov_ext = folder[4:]
                                    self.mov_folder_list.append(mov_ext)
                                    if not self.cam_folders:
                                        self.cam_folders.append('C001')
                                    else:
                                        if 'C001' in self.cam_folders:
                                            pass
                                        else:
                                            self.cam_folders.append('C001')
                                else:
                                    cam_ext = folder[0:4]
                                    if 'C00' in cam_ext:
                                        if cam_ext in self.cam_folders:
                                            pass
                                        else:
                                            self.cam_folders.append(cam_ext)
                self.cam_folders.sort()
                print(self.cam_folders)
                self.N_cam = len(self.cam_folders)
                print('N cam: ' + str(self.N_cam))
                self.mov_folder_list.sort()
                print(self.mov_folder_list)
                self.N_mov = len(self.mov_folder_list)
                print('N mov: ' + str(self.N_mov))
                self.img_viewer_1.set_mov_folders(self.mov_folder_list)
                self.img_viewer_1.set_cam_folders(self.cam_folders)
                self.img_viewer_2.set_N_cam(self.N_cam)
                self.img_viewer_2.set_N_mov(self.N_mov)
                self.img_viewer_2.set_mov_folders(self.mov_folder_list)
                self.img_viewer_2.set_cam_folders(self.cam_folders)
                # iterate over movies and get number of frames and format:
                frame_list = []
                for root, dirs, files in os.walk(self.session_path + '/' + self.cam_folders[0] + self.mov_folder_list[0]):
                    if len(files)>0:
                        for file in files:
                            file_name = str(os.path.splitext(file)[0])
                            file_ext = str(os.path.splitext(file)[1])
                            if file_ext == '.tif':
                                frame_nr = int(''.join(filter(str.isdigit, file_name[-6:])))
                                if frame_nr >= 0:
                                    frame_list.append(frame_nr)
                                if len(frame_list)==1:
                                    self.frame_name = file_name.replace(str(frame_nr),'')
                                    self.frame_img_format = '.tif'
                            elif file_ext == '.bmp':
                                frame_nr = int(''.join(filter(str.isdigit, file_name)))
                                if frame_nr >= 0:
                                    frame_list.append(frame_nr)
                                if len(frame_list)==1:
                                    self.frame_name = file_name.replace(str(frame_nr),'')
                                    self.frame_img_format = '.bmp'
                            elif file_ext == '.mraw':
                                os.chdir(root)
                                raw_type = 'photron_mraw_mono_8bit'
                                try:
                                    data = ImageSequence(file,rawtype=raw_type,height=self.frame_size[0][0],width=self.frame_size[0][1],frames=(0,1))
                                    if data.height*data.width>0:
                                        self.frame_name = file_name
                                        self.frame_img_format = '.mraw'
                                except:
                                    print('error: could not load mraw: check if specified frame height & width is correct.')
                frame_list.sort()
                self.img_viewer_1.set_img_format(self.frame_img_format)
                #self.img_viewer_2.set_img_format(self.frame_img_format)
                print('frame name: ' + self.frame_name)
                print('img format: ' + self.frame_img_format)
                if self.frame_img_format=='.mraw':
                    self.start_frame = self.frame_range[0]
                    print('start frame: ' + str(self.start_frame))
                    self.end_frame = self.frame_range[1]
                    print('end frame: ' + str(self.end_frame))
                    self.img_viewer_1.set_start_frame(self.start_frame,self.end_frame)
                    mov_loaded = True
                else:
                    self.start_frame = frame_list[0]
                    print('start frame: ' + str(self.start_frame))
                    self.end_frame = frame_list[-1]
                    print('end frame: ' + str(self.end_frame))
                    self.img_viewer_1.set_start_frame(self.start_frame,self.end_frame)
                    mov_loaded = True
            else:
                self.img_viewer_1.set_mov_folders(self.mov_folder_list)
                self.img_viewer_1.set_cam_folders(self.cam_folders)
                self.img_viewer_2.set_N_cam(self.N_cam)
                self.img_viewer_2.set_N_mov(self.N_mov)
                self.img_viewer_2.set_mov_folders(self.mov_folder_list)
                self.img_viewer_2.set_cam_folders(self.cam_folders)
                self.img_viewer_1.set_start_frame(self.start_frame,self.end_frame)
                # frame name & img_format
                self.frame_name = 'frame_'
                self.frame_img_format = '.h5py'
                self.img_viewer_1.set_img_format(self.frame_img_format)
                self.img_viewer_1.set_start_frame(self.start_frame,self.end_frame)
                self.img_viewer_1.set_flight_seq_disp(self.load_file_disp)
                self.img_viewer_1.set_session_folder(self.session_path,self.session_folder)
                self.load_crop_seq_btn.clicked.connect(self.load_flight_seqs)
                mov_loaded = True
        except:
            self.N_cam = 0
            self.cam_folders = []
            self.N_mov = 0
            self.mov_folder_list = []
            self.start_frame = 0
            self.end_frame = 0
            print('TIF loader could not load movies')
        return mov_loaded

    def set_trigger_mode_box(self,combo_box_in):
        self.trigger_mode_box = combo_box_in
        self.trigger_mode_box.addItem(self.trig_modes[0])
        self.trigger_mode_box.addItem(self.trig_modes[1])
        self.trigger_mode_box.addItem(self.trig_modes[2])
        self.trigger_mode_box.addItem(self.trig_modes[3])
        self.trigger_mode_box.currentIndexChanged.connect(self.select_trigger_callback)

    def calc_trigger_wb(self):
        try:
            if self.trigger_mode == self.trig_modes[2]:
                self.trigger_frame = int(math.floor((self.end_frame-self.start_frame)/2.0)+1)
            elif self.trigger_mode == self.trig_modes[1]:
                self.trigger_frame = self.start_frame
            elif self.trigger_mode == self.trig_modes[3]:
                self.trigger_frame = self.end_frame
        except:
            print('error: could not calculate trigger frame')
            self.trigger_mode = self.trig_modes[0]
            self.trigger_mode_combo.setCurrentIndex(0)

    def select_trigger_callback(self,ind):
        self.trigger_mode = self.trig_modes[ind]
        self.calc_trigger_wb()

    def set_table_widget(self,table_in):
        self.par_table = table_in
        self.par_table.setRowCount(17)
        self.par_table.setColumnCount(7)
        self.par_table.setItem(0,0,QTableWidgetItem('Movie folders:'))
        self.par_table.setItem(0,1,QTableWidgetItem('Camera 1'))
        self.par_table.setItem(0,2,QTableWidgetItem('Camera 2'))
        self.par_table.setItem(0,3,QTableWidgetItem('Camera 3'))
        self.par_table.setItem(0,4,QTableWidgetItem('Camera 4'))
        self.par_table.setItem(0,5,QTableWidgetItem('Camera 5'))
        self.par_table.setItem(0,6,QTableWidgetItem('Camera 6'))
        self.par_table.resizeColumnsToContents()

    def update_table_widget(self):
        if self.N_mov > 0:
            for i in range(self.N_mov):
                self.par_table.setItem(i+1,0,QTableWidgetItem(self.mov_folder_list[i]))
                for j in range(self.N_cam):
                    self.par_table.setItem(i+1,j+1,QTableWidgetItem(self.cam_folders[j]))
        self.par_table.resizeColumnsToContents()

    def load_seq_callback(self):
        self.active_frame_gui()

    #-----------------------------------------------------------------------
    #
    #    Background subtraction, calibration, masking and cropping
    #
    #-----------------------------------------------------------------------

    def inactive_frame_gui(self):
        # movie spin
        self.movie_spin_1.setMinimum(0)
        self.movie_spin_1.setMaximum(0)
        self.movie_spin_1.setValue(0)
        # frame spin
        self.frame_spin_1.setMinimum(0)
        self.frame_spin_1.setMaximum(0)
        self.frame_spin_1.setValue(0)
        # camera view spin 2

    def active_frame_gui(self):
        # set sequence folder
        self.img_viewer_1.set_session_folder(self.session_path,self.session_folder)
        self.img_viewer_1.set_N_cam(self.N_cam)
        self.img_viewer_1.set_trigger_mode(self.trigger_mode,self.trigger_frame)
        self.img_viewer_1.load_bckg_frames(self.bckg_path,self.bckg_frames,self.bckg_img_format)
        self.img_viewer_1.load_camera_calibration(self.c_params,self.c_type)
        # movie spin
        self.movie_spin_1.setMinimum(1)
        self.movie_spin_1.setMaximum(self.N_mov)
        self.movie_spin_1.setValue(1)
        self.img_viewer_1.set_movie_nr(1)
        self.movie_spin_1.valueChanged.connect(self.img_viewer_1.set_movie_nr)
        # frame spin
        self.frame_spin_1.setMinimum(self.start_frame)
        self.frame_spin_1.setMaximum(self.end_frame)
        self.frame_spin_1.setValue(self.start_frame)
        self.img_viewer_1.add_frame(self.start_frame,self.end_frame)
        self.frame_spin_1.valueChanged.connect(self.img_viewer_1.update_frame)
        # frame slider
        self.frame_slider.setMinimum(self.start_frame)
        self.frame_slider.setMaximum(self.end_frame)
        self.frame_slider.setValue(self.start_frame)
        self.frame_slider.valueChanged.connect(self.img_viewer_1.update_frame)
        # copy frame spin and slider objects into img_viewer_1
        self.img_viewer_1.set_frame_spin(self.frame_spin_1)
        self.img_viewer_1.set_frame_slider(self.frame_slider)
        # Free flight & tethered flight toggles:
        self.img_viewer_1.set_Nu_crop_spin(self.Nu_crop_spin)
        self.img_viewer_1.set_Nv_crop_spin(self.Nv_crop_spin)
        self.img_viewer_1.set_manual_mode_toggle(self.man_select_btn)
        self.img_viewer_1.set_automatic_mode_toggle(self.auto_select_btn)
        self.img_viewer_1.set_crop_x_spin(self.crop_x_spin)
        self.img_viewer_1.set_crop_y_spin(self.crop_y_spin)
        self.img_viewer_1.set_crop_z_spin(self.crop_z_spin)
        # Body thresh spin:
        self.body_thresh_spin.setMinimum(0)
        self.body_thresh_spin.setMaximum(255)
        self.body_thresh_spin.setValue(220)
        self.img_viewer_1.set_body_thresh(220)
        self.body_thresh_spin.valueChanged.connect(self.img_viewer_1.set_body_thresh)
        # Min sequence length spin
        self.min_len_spin.setMinimum(1)
        self.min_len_spin.setMaximum(10000)
        self.min_len_spin.setValue(40)
        self.img_viewer_1.set_min_seq_len(40)
        self.min_len_spin.valueChanged.connect(self.img_viewer_1.set_min_seq_len)
        # ROI mask setting:
        self.img_viewer_1.set_cam_mask_spin(self.cam_mask_spin)
        self.img_viewer_1.set_add_mask_btn(self.add_mask_btn)
        self.img_viewer_1.set_remove_masks_btn(self.remove_masks_btn)
        # Min COM distance:
        self.max_dist_COM_spin.setMinimum(0)
        self.max_dist_COM_spin.setMaximum(1000)
        self.max_dist_COM_spin.setValue(50)
        self.img_viewer_1.set_max_COM_dist(50)
        self.max_dist_COM_spin.valueChanged.connect(self.img_viewer_1.set_max_COM_dist)
        # Calculate flight sequences:
        self.calc_crop_seq_btn.clicked.connect(self.img_viewer_1.calculate_flight_seqs)
        # Progress bar
        self.img_viewer_1.set_ses_calc_progress(self.crop_seq_calc_progress)
        # Sequence display:
        self.img_viewer_1.set_flight_seq_disp(self.load_file_disp)
        # Load flight sequences:
        self.load_crop_seq_btn.clicked.connect(self.load_flight_seqs)

    def set_crop_window(self):
        self.img_viewer_1.set_img_cntr()
        self.img_centers = self.img_viewer_1.uv_centers
        self.crop_window_size = self.img_viewer_1.crop_window
        self.thorax_center = self.img_viewer_1.thorax_center
        if self.annotation_active:
            self.img_viewer_2.load_crop_center(self.img_centers,self.crop_window_size,self.frame_size)

    def LoadCalibrationMatrix(self,calib_ext):
        os.chdir(self.calib_path)
        self.c_type = 0
        if calib_ext=='.mat':
            data = loadmat(self.calib_file + '.mat')
            self.c_params = np.zeros((11,3))
            self.c_params[:,0] = np.squeeze(data['DLT'][0][0][0])
            self.c_params[:,1] = np.squeeze(data['DLT'][0][1][0])
            self.c_params[:,2] = np.squeeze(data['DLT'][0][2][0])
            self.c_type = 0
        elif calib_ext=='.txt':
            self.c_params = np.loadtxt(self.calib_file + '.txt', delimiter='\t')
            self.c_type = 1
        print('calibration data')
        print(self.c_params)

    def load_flight_seqs(self):
        print('load_flight_seqs')
        self.img_viewer_1.load_flight_seqs()
        self.tracking_gui()
        self.load_model_callback(select_window=False)

    def tracking_gui(self):
        self.tabWidget.setTabEnabled(1,True)
        # min seq length:
        try:
            self.img_viewer_2.set_min_seq_length(self.img_viewer_1.min_seq_len)
        except:
            self.img_viewer_2.set_min_seq_length(40)
        # body thresh:
        try:
            self.img_viewer_2.set_body_thresh(self.img_viewer_1.body_thresh)
        except:
            self.img_viewer_2.set_body_thresh(220)
        # img & seg view:
        self.img_viewer_2.set_img_seg_view_btns(self.img_view_btn,self.seg_view_btn)
        # Seq nr spin
        self.img_viewer_2.set_seq_spin(self.seq_nr_spin)
        # Frame spin 2
        self.img_viewer_2.set_frame_spin(self.frame_spin_2)
        # Frame slider 2
        self.img_viewer_2.set_frame_slider(self.frame_slider_2)
        # set start & end frame
        try:
            trigger_mode  = self.img_viewer_1.trigger_mode
            trigger_frame = self.img_viewer_1.trigger_frame
            frame_start   = self.img_viewer_1.start_frame
            frame_end       = self.img_viewer_1.end_frame
            self.img_viewer_2.set_trigger_mode(trigger_mode,trigger_frame,frame_start,frame_end)
        except:
            trigger_mode  = self.trigger_mode
            trigger_frame = self.trigger_frame
            frame_start   = self.start_frame
            frame_end       = self.end_frame
            self.img_viewer_2.set_trigger_mode(trigger_mode,trigger_frame,frame_start,frame_end)
        # Setup image viewer
        self.img_viewer_2.set_seq_file(self.img_viewer_1.save_fldr,self.img_viewer_1.seq_file_name)
        # Movie spin 2
        self.movie_spin_2.setMinimum(1)
        self.movie_spin_2.setMaximum(self.N_mov)
        self.movie_spin_2.setValue(1)
        self.img_viewer_2.set_movie_nr(1)
        self.movie_spin_2.valueChanged.connect(self.img_viewer_2.set_movie_nr)
        # Select model button
        self.select_mdl_btn.clicked.connect(self.load_model_callback)
        # Batch size spin
        self.img_viewer_2.set_batch_size_spin(self.batch_size_spin)
        # Analysis progress
        self.analysis_progress.setValue(0)
        self.outlier_fix_progress.setValue(0)
        self.train_net_progress.setValue(0)
        # Load tracked data button
        self.load_tracked_btn.clicked.connect(self.load_data_tracked_callback)
        self.load_tracked_btn.setEnabled(False)

    def load_model_callback(self, select_window=True):
        if select_window:
            self.select_model_window.exec_()
            model_file_loc = str(self.select_model_window.folder_path)
            model_file_name = str(self.select_model_window.folder_name)
            self.model_name = model_file_name
            self.model_disp.setText(self.model_name)
            sys.path.append(model_file_loc)
            import model
            self.mdl = model.Model()
        else:
            self.mdl = drosophila.Model()

        #print('window size')
        #self.window_size = self.img_viewer_1.window_size
        #print(self.window_size)
        crop_window_size = []
        for n in range(self.N_cam):
            crop_window_size.append(self.img_viewer_1.crop_window)
        #self.mdl.load_camera_calibration(self.c_params,self.N_cam,crop_window_size)
        self.c_type = self.img_viewer_1.c_type
        self.mdl.load_camera_calibration(self.c_params,self.c_type,self.N_cam,crop_window_size)
        self.mdl.set_v_list(self.img_viewer_2.v_list)
        self.mdl.set_model_view_toggle(self.mdl_view_btn)
        self.mdl.set_single_view_toggle(self.single_view_btn)
        self.mdl.set_off_view_toggle(self.off_view_btn)
        self.mdl.set_component_combo(self.mdl_combo)
        self.mdl.set_qx_spin(self.qx_spin)
        self.mdl.set_qy_spin(self.qy_spin)
        self.mdl.set_qz_spin(self.qz_spin)
        self.mdl.set_tx_spin(self.tx_spin)
        self.mdl.set_ty_spin(self.ty_spin)
        self.mdl.set_tz_spin(self.tz_spin)
        self.mdl.set_xi_spin(self.xi_spin)
        self.mdl.set_scale_spin(self.scale_spin)
        #self.mdl.set_scale_table(self.scale_table)
        #self.mdl.set_state_table(self.state_table)
        self.mdl.set_seq_file(self.img_viewer_2.seq_file,self.session_folder)
        self.mdl.set_label_btn(self.add_lbl_btn)
        mdl_scale = [1.0,1.0,1.0,1.0,1.0]
        self.mdl.set_model_scale(mdl_scale)
        self.mdl.set_model_start_state()
        self.mdl.set_N_particles_spin(self.N_particles_spin)
        self.mdl.set_N_iter_spin(self.N_iter_spin)
        #self.mdl.set_cost_spin(self.cost_spin)
        self.mdl.set_pso_search_btn(self.pso_search_btn)
        # set model in img_viewer_2
        self.img_viewer_2.set_model(self.mdl)
        self.mdl.set_seq_list(self.img_viewer_2.seq_keys)
        # Set network combo:
        self.select_weights_window = SelectFolderWindow(str(self.weights_dir))
        self.select_weights_window.setWindowTitle("Select network weights file")
        for net_option in self.network_options:
            self.network_combo.addItem(net_option)
        self.network_combo.currentIndexChanged.connect(self.select_network)
        # Load weights btn:
        self.select_weights_btn.clicked.connect(self.load_weights_callback)
        # Analyze sequence window:
        self.mdl.set_analyze_btn(self.analyze_btn)
        #self.analysis_progress
        self.analysis_progress.setValue(0)
        self.mdl.set_progress_bar(self.analysis_progress)
        # Set load_tracked_btn
        self.mdl.set_load_tracked_data_btn(self.load_tracked_btn)

    def select_network(self,net_ind):
        self.network_index = 0
        if net_ind == 1:
            print('selected FlyNet')
            self.net = FlyNet.Network()
            self.network_index = 1
        #elif net_ind == 2:
        #    print('selected FlyNet2')
        #    os.chdir(self.network_dir+'/FlyNet2')
        #    sys.path.append(os.getcwd())
        #    from FlyNet2 import Network
        #    self.net = Network()
        #    self.network_index = 2
        #elif net_ind==3:
        #    print('selected FlyNet3')
        #    os.chdir(self.network_dir+'/FlyNet3')
        #    sys.path.append(os.getcwd())
        #    from FlyNet3 import Network
        #    self.net = Network()
        #    self.network_index = 3
        #else:
        #    print('no network selected')

    def load_weights_callback(self):
        self.select_weights_window.exec_()
        weight_file_loc = str(self.select_weights_window.folder_path)
        weight_file_name = str(self.select_weights_window.folder_name)
        weight_file_dir = weight_file_loc.replace(weight_file_name,'')
        print(weight_file_dir)
        self.weights_file = weight_file_name
        self.weights_disp.setText(self.weights_file)
        print(self.weights_file)
        if self.network_index == 1:
            self.net.set_weights_file(weight_file_dir,self.weights_file)
        elif self.network_index == 2:
            self.net.set_weights_file(weight_file_dir,self.weights_file)
        elif self.network_index == 3:
            self.net.set_weights_file(weight_file_dir,self.weights_file)
        self.net.set_learning_rate(1.0e-5,1.0e-7)
        self.input_shape = self.net.input_shape
        self.net.load_network()
        self.mdl.set_prediction_network(self.net)
        print('predictor network set')
        self.mdl.set_predictor_button(self.predict_btn)

    def load_data_tracked_callback(self):
        self.filter_gui()
    
    def filter_gui(self):
        self.tabWidget.setTabEnabled(2,True)
        self.filter = PostProcessing()
        self.filter.set_seq_file(self.img_viewer_2.seq_file,self.session_folder)
        self.filter.load_data()
        self.filter.set_component_combo(self.comp_combo_3,self.mdl.mdl_components)
        self.filter.set_mov_spin(self.movie_spin_3)
        self.filter.set_seq_spin(self.seq_spin_3)
        self.filter.set_frame_spin(self.frame_spin_3)
        self.filter.set_frame_slider(self.frame_slider_3)
        self.filter.set_q_plot(self.q_plot)
        self.filter.set_q_plot_zoom(self.q_plot_zoom)
        self.filter.set_t_plot(self.t_plot)
        self.filter.set_t_plot_zoom(self.t_plot_zoom)
        self.filter.set_x_plot(self.x_plot)
        self.filter.set_x_plot_zoom(self.x_plot_zoom)
        self.filter.set_cost_plot(self.cost_plot)
        self.filter.set_cost_plot_zoom(self.cost_plot_zoom)
        self.filter.set_kalman_combo(self.comp_combo_4)
        self.filter.set_n_deriv_kalman_spin(self.n_deriv_kalman_spin)
        self.filter.set_fps_spin(self.fps_spin)
        self.filter.set_q_kalman_spin(self.quat_kalman_spin)
        self.filter.set_t_kalman_spin(self.trans_kalman_spin)
        self.filter.set_x_kalman_spin(self.xi_kalman_spin)
        self.filter.set_kalman_progress_bar(self.kalman_progress)
        self.filter.set_kalman_filter_btn(self.kalman_filter_btn)
        self.filter.set_raw_trace_toggle(self.raw_trace_btn)
        self.filter.set_raw_pred_trace_toggle(self.raw_pred_trace_btn)
        self.filter.set_raw_filt_trace_toggle(self.raw_filt_trace_btn)
        self.filter.set_filt_trace_toggle(self.filt_trace_btn)
        self.filter.set_stroke_trace_toggle(self.stroke_trace_btn)
        self.filter.set_n_deriv_view_spin(self.deriv_spin)
        self.filter.set_N_phi_spin(self.N_phi_spin)
        self.filter.set_N_theta_spin(self.N_theta_spin)
        self.filter.set_N_eta_spin(self.N_eta_spin)
        self.filter.set_N_xi_spin(self.N_xi_spin)
        self.filter.set_fit_wbs_btn(self.fit_wbs_btn)
        self.filter.set_strk_angle_spin(self.strk_angle_spin)
        self.filter.set_phi_thresh_spin(self.phi_thresh_spin)
        self.filter.set_Legendre_progress(self.Legendre_progress)
        self.filter.set_batch_size(self.img_viewer_2.batch_size)
        self.filter.raw_filt_trace_toggled()
        self.filter.set_model(self.mdl)
        self.filter.set_img_viewer(self.img_viewer_2)
        self.filter.set_find_outliers_btn(self.find_outliers_btn)
        self.filter.set_fix_frames_table(self.fix_frames_table)
        self.filter.set_add_outlier_frame_btn(self.add_frame_3_btn)
        self.filter.set_outlier_fix_progress(self.outlier_fix_progress)
        self.filter.set_outlier_frame_table(self.outlier_frame_table)
        self.filter.set_auto_fix_btn(self.auto_fix_btn)
        self.filter.set_update_outlier_btn(self.update_outlier_btn)
        self.filter.set_outlier_spin(self.outlier_spin)
        self.filter.set_mov_nr(1)

    #self.outlier_fix_progress
    #self.train_net_progress
    #self.train_net_btn
    #self.auto_fix_btn
    #self.update_outlier_btn
    #self.outlier_spin
    #self.outlier_frame_table


# -------------------------------------------------------------------------------------------------

def appMain():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = FlyNetViewer()
    mainWindow.show()
    app.exec_()

# -------------------------------------------------------------------------------------------------
#if __name__ == '__main__':
#
#    appMain()
