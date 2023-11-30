#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import sys
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidget, QTableWidgetItem, QVBoxLayout, QFileDialog
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import matplotlib as mpl
import matplotlib.pyplot as plt

from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl import NURBS
from geomdl import tessellate
import numpy as np
import numpy.matlib
import os
import copy
import time
import json
import h5py
import math
import cv2
from scipy.linalg import orth

import pathlib
import importlib.resources
import flynet_optimizer

class Model():

    def __init__(self):
        #self.mdl_dir = '/home/flythreads/Documents/FlyNet4/models/drosophila'
        self.mdl_dir = importlib.resources.files('flynet.drosophila')
        self.label_dir = '/home/flythreads/Documents/FlyNet4/networks/labels'
        self.label_file = 'labels.h5'
        #self.label_file = 'valid_labels.h5'
        self.load_model()
        self.scale = np.array([1.0,1.0,1.0,1.0,1.0])
        self.N_state = 37
        self.state = np.zeros(self.N_state)
        self.COM = np.array([0.0,0.0,0.0,1.0])
        self.comp_ind = -1
        self.transform_M = []
        self.curves = []
        self.mdl_components = ['...','head','thorax','abdomen','wing left','wing right']
        self.full_view = True
        self.buffer_size = 100
        self.state_buffer = np.zeros((37,self.buffer_size))
        self.state_buffer.fill(np.NaN)
        # Set MDL_PSO
        #self.opt = MDL_PSO_lib.MDL_PSO()
        self.opt = flynet_optimizer.MDL_PSO()

    def SetRendering(self,ren,renWin,iren):
        self.ren = ren
        self.renWin = renWin
        self.iren = iren
        # body
        for actor in self.body_actors:
            self.ren.AddActor(actor)
        for actor in self.mem_L_actors:
            self.ren.AddActor(actor)
        for actor in self.vein_L_actors:
            self.ren.AddActor(actor)
        for actor in self.mem_R_actors:
            self.ren.AddActor(actor)
        for actor in self.vein_R_actors:
            self.ren.AddActor(actor)
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

    def load_camera_calibration(self,c_params,c_type,N_cam_in,window_size_in):
        print ("loading camera calibration")
        self.N_cam = N_cam_in
        self.window_size = window_size_in
        self.c_type = c_type
        self.c_params = c_params
        self.world2cam = []
        self.w2c_scaling = []
        self.uv_shift = []
        self.pix_size = 0.020 # 20 micrometers
        self.cam_pos = []
        self.focal_pts = []
        self.view_up = []
        self.window_center = []
        w_dist = 200.0 # 200 milimeters
        self.clip_range = (w_dist/2.0,-w_dist/2.0)
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
                camera_position = np.dot(np.transpose(R),np.array([0.0,0.0,-w_dist]))
                self.cam_pos.append(camera_position)
                self.focal_pts.append(np.array([0.0,0.0,0.0]))
                self.view_up.append(np.dot(np.transpose(R),np.array([0.0,-1.0,0.0])))
                self.window_center.append([0.0,0.0])

    def project_2_uv(self,xyz_pts,cam_nr):
        xyz_now = np.copy(xyz_pts)
        com_pts = np.array([[np.squeeze(self.COM[0])],[np.squeeze(self.COM[1])],[np.squeeze(self.COM[2])],[1.0]])
        xyz_now[:3,:] += np.matlib.repmat(com_pts[:3],1,xyz_pts.shape[1])
        if self.c_type==0:
            uv_pts = np.divide(np.dot(self.world2cam[cam_nr],xyz_now),np.dot(self.w2c_scaling[cam_nr],xyz_now))
            zero_pts = com_pts
            uv_pts_0 = np.divide(np.dot(self.world2cam[cam_nr],zero_pts),np.dot(self.w2c_scaling[cam_nr],zero_pts))
            uv_out = uv_pts
            try:
                uv_out[0,:] = (self.window_size[cam_nr][0]/2.0)+(uv_pts[0,:]-uv_pts_0[0,:])
                uv_out[1,:] = (self.window_size[cam_nr][1]/2.0)-(uv_pts[1,:]-uv_pts_0[1,:])
            except:
                uv_out[0] = (self.window_size[cam_nr][0]/2.0)+(uv_pts[0]-uv_pts_0[0])
                uv_out[1] = (self.window_size[cam_nr][1]/2.0)-(uv_pts[1]-uv_pts_0[1])
        elif self.c_type==1:
            uv_pts = np.dot(self.world2cam[cam_nr],xyz_now)
            zero_pts = com_pts
            uv_pts_0 = np.dot(self.world2cam[cam_nr],zero_pts)
            uv_out = uv_pts
            try:
                uv_out[0,:] = (self.window_size[cam_nr][0]/2.0)+(uv_pts[0,:]-uv_pts_0[0,:])
                uv_out[1,:] = (self.window_size[cam_nr][1]/2.0)-(uv_pts[1,:]-uv_pts_0[1,:])
            except:
                uv_out[0] = (self.window_size[cam_nr][0]/2.0)+(uv_pts[0]-uv_pts_0[0])
                uv_out[1] = (self.window_size[cam_nr][1]/2.0)-(uv_pts[1]-uv_pts_0[1])
        return uv_out

    def load_model(self):
        self.body_stl_files = ['head.stl','thorax.stl','abdomen.stl']
        self.body_clr = (0.01,0.01,0.01)
        self.body_model = []
        self.body_actors = []
        self.body_points = []
        self.body_vertices = []

        for file in self.body_stl_files:
            poly = self.load_stl(file)
            self.body_model.append(poly)
            Mapper = vtk.vtkPolyDataMapper()
            Mapper.SetInputData(poly)
            Actor = vtk.vtkActor()
            Actor.GetProperty().SetColor(self.body_clr[0],self.body_clr[1],self.body_clr[2])
            Actor.SetMapper(Mapper)
            self.body_actors.append(Actor)
            pts = self.extract_points(poly)
            pts_array = np.ones((4,pts.shape[0]))
            pts_array[:3,:] = np.transpose(pts)
            self.body_points.append(pts_array)
            verts = self.extract_vertices(poly)
            self.body_vertices.append(verts)
        self.mem_L_stl_files = ['membrane_0_L.stl','membrane_1_L.stl','membrane_2_L.stl','membrane_3_L.stl']
        self.mem_clr = (0.01,0.01,0.01)
        self.mem_opacity = 0.3
        self.mem_L_model = []
        self.mem_L_actors = []
        self.mem_L_points = []
        self.mem_L_vertices = []
        for file in self.mem_L_stl_files:
            poly = self.load_stl(file)
            self.mem_L_model.append(poly)
            Mapper = vtk.vtkPolyDataMapper()
            Mapper.SetInputData(poly)
            Actor = vtk.vtkActor()
            Actor.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
            Actor.GetProperty().SetOpacity(self.mem_opacity)
            Actor.ForceTranslucentOn()
            Actor.SetMapper(Mapper)
            self.mem_L_actors.append(Actor)
            pts = self.extract_points(poly)
            pts_array = np.ones((4,pts.shape[0]))
            pts_array[:3,:] = np.transpose(pts)
            self.mem_L_points.append(pts_array)
            verts = self.extract_vertices(poly)
            self.mem_L_vertices.append(verts)
        self.vein_L_stl_files = ['vein_0_L.stl','vein_1_L.stl','vein_2_L.stl','vein_3_L.stl','vein_4_L.stl',
            'vein_5_L.stl','vein_C1_L.stl','vein_C2_L.stl','vein_C3_L.stl','vein_A_L.stl','vein_P_L.stl']
        self.vein_clr = (0.01,0.01,0.01)
        self.vein_L_model = []
        self.vein_L_points = []
        self.vein_L_actors = []
        self.vein_L_vertices = []
        for file in self.vein_L_stl_files:
            poly = self.load_stl(file)
            self.vein_L_model.append(poly)
            Mapper = vtk.vtkPolyDataMapper()
            Mapper.SetInputData(poly)
            Actor = vtk.vtkActor()
            Actor.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            Actor.SetMapper(Mapper)
            self.vein_L_actors.append(Actor)
            pts = self.extract_points(poly)
            pts_array = np.ones((4,pts.shape[0]))
            pts_array[:3,:] = np.transpose(pts)
            self.vein_L_points.append(pts_array)
            verts = self.extract_vertices(poly)
            self.vein_L_vertices.append(verts)
        self.mem_R_stl_files = ['membrane_0_R.stl','membrane_1_R.stl','membrane_2_R.stl','membrane_3_R.stl']
        self.mem_R_model = []
        self.mem_R_actors = []
        self.mem_R_points = []
        self.mem_R_vertices = []
        for file in self.mem_R_stl_files:
            poly = self.load_stl(file)
            self.mem_R_model.append(poly)
            Mapper = vtk.vtkPolyDataMapper()
            Mapper.SetInputData(poly)
            Actor = vtk.vtkActor()
            Actor.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
            Actor.GetProperty().SetOpacity(self.mem_opacity)
            Actor.ForceTranslucentOn()
            Actor.SetMapper(Mapper)
            self.mem_R_actors.append(Actor)
            pts = self.extract_points(poly)
            pts_array = np.ones((4,pts.shape[0]))
            pts_array[:3,:] = np.transpose(pts)
            self.mem_R_points.append(pts_array)
            verts = self.extract_vertices(poly)
            self.mem_R_vertices.append(verts)
        self.vein_R_stl_files = ['vein_0_R.stl','vein_1_R.stl','vein_2_R.stl','vein_3_R.stl','vein_4_R.stl',
            'vein_5_R.stl','vein_C1_R.stl','vein_C2_R.stl','vein_C3_R.stl','vein_A_R.stl','vein_P_R.stl']
        self.vein_R_model = []
        self.vein_R_actors = []
        self.vein_R_points = []
        self.vein_R_vertices = []
        for file in self.vein_R_stl_files:
            poly = self.load_stl(file)
            self.vein_R_model.append(poly)
            Mapper = vtk.vtkPolyDataMapper()
            Mapper.SetInputData(poly)
            Actor = vtk.vtkActor()
            Actor.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            Actor.SetMapper(Mapper)
            self.vein_R_actors.append(Actor)
            pts = self.extract_points(poly)
            pts_array = np.ones((4,pts.shape[0]))
            pts_array[:3,:] = np.transpose(pts)
            self.vein_R_points.append(pts_array)
            verts = self.extract_vertices(poly)
            self.vein_R_vertices.append(verts)
        print('drosophila model loaded')

    def load_stl(self,file_name):
        #os.chdir(self.mdl_dir)
        reader = vtk.vtkSTLReader()
        #reader.SetFileName(file_name)
        file_path = pathlib.Path(self.mdl_dir, file_name)
        print(f'file_path {file_path}')
        reader.SetFileName(str(file_path))
        reader.Update()
        poly_out = vtk.vtkPolyData()
        poly_out.DeepCopy(reader.GetOutput())
        return poly_out

    def extract_points(self,polydata_in):
        points = polydata_in.GetPoints()
        array = points.GetData()
        pts_out = vtk_to_numpy(array)
        return pts_out

    def extract_vertices(self,polydata):
        cells = polydata.GetPolys()
        ids = []
        idList = vtk.vtkIdList()
        cells.InitTraversal()
        while cells.GetNextCell(idList):
            for i in range(0, idList.GetNumberOfIds()):
                pId = idList.GetId(i)
                if i == 0:
                    pId_first = pId
                ids.append(pId)
            ids.append(pId_first)
        ids = np.array(ids)
        return ids

    def set_model_scale(self,scale_in):
        self.scale = scale_in
        self.body_pts_scaled = []
        self.body_pts_scaled.append(np.multiply(self.body_points[0],np.array([[scale_in[0]],[scale_in[0]],[scale_in[0]],[1.0]])))
        self.body_pts_scaled.append(np.multiply(self.body_points[1],np.array([[scale_in[1]],[scale_in[1]],[scale_in[1]],[1.0]])))
        self.body_pts_scaled.append(np.multiply(self.body_points[2],np.array([[scale_in[2]],[scale_in[2]],[scale_in[2]],[1.0]])))
        self.mem_L_pts_scaled = []
        self.mem_L_pts_scaled.append(np.multiply(self.mem_L_points[0],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.mem_L_pts_scaled.append(np.multiply(self.mem_L_points[1],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.mem_L_pts_scaled.append(np.multiply(self.mem_L_points[2],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.mem_L_pts_scaled.append(np.multiply(self.mem_L_points[3],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled = []
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[0],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[1],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[2],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[3],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[4],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[5],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[6],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[7],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[8],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[9],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.vein_L_pts_scaled.append(np.multiply(self.vein_L_points[10],np.array([[scale_in[3]],[scale_in[3]],[scale_in[3]],[1.0]])))
        self.mem_R_pts_scaled = []
        self.mem_R_pts_scaled.append(np.multiply(self.mem_R_points[0],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.mem_R_pts_scaled.append(np.multiply(self.mem_R_points[1],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.mem_R_pts_scaled.append(np.multiply(self.mem_R_points[2],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.mem_R_pts_scaled.append(np.multiply(self.mem_R_points[3],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled = []
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[0],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[1],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[2],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[3],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[4],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[5],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[6],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[7],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[8],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[9],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))
        self.vein_R_pts_scaled.append(np.multiply(self.vein_R_points[10],np.array([[scale_in[4]],[scale_in[4]],[scale_in[4]],[1.0]])))

    def set_model_start_state(self):
        #self.set_start_state()
        self.start_state = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        self.set_model_state(self.start_state)

    def set_model_state(self,state_in):
        self.state = state_in
        self.compute_transformation_mats(state_in)
        self.display_curves()
        #self.update_state_table()

    def set_COM(self,com_in):
        self.COM = com_in

    def ModelPlot(self,frame_ind):
        self.ren.SetUseDepthPeeling(True)
        self.renWin.SetOffScreenRendering(1)
        img_list = []
        for i in range(self.N_cam):
            self.renWin.SetSize(self.window_size[0]*4,self.window_size[1]*4)
            camera = self.ren.GetActiveCamera()
            camera.SetParallelProjection(True)
            camera.SetParallelScale(self.pix_size*self.window_size[0])
            camera.SetClippingRange(self.clip_range[0], self.clip_range[1])
            camera.SetPosition(self.cam_pos[i][0], self.cam_pos[i][1], self.cam_pos[i][2])
            camera.SetFocalPoint(self.focal_pts[i][0], self.focal_pts[i][1], self.focal_pts[i][2])
            camera.SetViewUp(self.view_up[i][0],self.view_up[i][1],self.view_up[i][2])
            camera.OrthogonalizeViewUp()
            camera.SetWindowCenter(0,0)
            self.ren.ResetCameraClippingRange()
            self.renWin.Render()
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.renWin)
            w2i.Update()
            overlay_img = w2i.GetOutput()
            n_rows, n_cols, _ = overlay_img.GetDimensions()
            overlay_sc = overlay_img.GetPointData().GetScalars()
            np_img = vtk_to_numpy(overlay_sc)
            np_img = cv2.flip(np_img.reshape(n_rows,n_cols,-1),0)
            gauss_img = cv2.GaussianBlur(np_img,(7,7),cv2.BORDER_DEFAULT)
            img_gray = cv2.cvtColor(gauss_img,cv2.COLOR_RGB2GRAY)
            downsized_img = cv2.resize(img_gray,(self.window_size[0],self.window_size[1]), interpolation = cv2.INTER_AREA)
            noise = cv2.randu(np.zeros((self.window_size[0],self.window_size[1]),dtype=np.uint8),(0),(30))
            img_out = cv2.add(downsized_img,noise)
            img_list.append(img_out)
        #os.chdir('/home/flythreads/Documents/FlyNet4/ArtificialDatagenerator/test_images')
        #combined_img = np.concatenate((img_list[0],img_list[1],img_list[2]),axis=1)
        #cv2.imwrite('frame_' + str(frame_ind) + '.bmp',combined_img)
        #time.sleep(0.001)
        return img_list

    def update_actors(self,scale_in,state_in,color_body,color_vein,color_mem,opacity_mem):
        # Compute transformation matrices:
        self.compute_transformation_mats(state_in)
        # head model
        self.body_actors[0].SetScale(scale_in[0],scale_in[0],scale_in[0])
        self.body_actors[0].GetProperty().SetColor(color_body[0],color_body[1],color_body[2])
        self.body_actors[0].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[0]))
        self.body_actors[0].Modified()
        # thorax model
        self.body_actors[1].SetScale(scale_in[1],scale_in[1],scale_in[1])
        self.body_actors[1].GetProperty().SetColor(color_body[0],color_body[1],color_body[2])
        self.body_actors[1].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[1]))
        self.body_actors[1].Modified()
        # abdomen model
        self.body_actors[2].SetScale(scale_in[2],scale_in[2],scale_in[2])
        self.body_actors[2].GetProperty().SetColor(color_body[0],color_body[1],color_body[2])
        self.body_actors[2].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[2]))
        self.body_actors[2].Modified()
        # wing L model
        # membrane
        self.mem_L_actors[0].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.mem_L_actors[0].GetProperty().SetColor(color_mem[0],color_mem[1],color_mem[2])
        self.mem_L_actors[0].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[3]))
        self.mem_L_actors[0].GetProperty().SetOpacity(opacity_mem)
        self.mem_L_actors[0].ForceTranslucentOn()
        self.mem_L_actors[0].Modified()
        self.mem_L_actors[1].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.mem_L_actors[1].GetProperty().SetColor(color_mem[0],color_mem[1],color_mem[2])
        self.mem_L_actors[1].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[4]))
        self.mem_L_actors[1].GetProperty().SetOpacity(opacity_mem)
        self.mem_L_actors[1].ForceTranslucentOn()
        self.mem_L_actors[1].Modified()
        self.mem_L_actors[2].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.mem_L_actors[2].GetProperty().SetColor(color_mem[0],color_mem[1],color_mem[2])
        self.mem_L_actors[2].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[5]))
        self.mem_L_actors[2].GetProperty().SetOpacity(opacity_mem)
        self.mem_L_actors[2].ForceTranslucentOn()
        self.mem_L_actors[2].Modified()
        self.mem_L_actors[3].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.mem_L_actors[3].GetProperty().SetColor(color_mem[0],color_mem[1],color_mem[2])
        self.mem_L_actors[3].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[6]))
        self.mem_L_actors[3].GetProperty().SetOpacity(opacity_mem)
        self.mem_L_actors[3].ForceTranslucentOn()
        self.mem_L_actors[3].Modified()
        # veins
        self.vein_L_actors[0].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[0].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[0].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[3]))
        self.vein_L_actors[0].Modified()
        self.vein_L_actors[1].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[1].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[1].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[3]))
        self.vein_L_actors[1].Modified()
        self.vein_L_actors[2].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[2].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[2].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[3]))
        self.vein_L_actors[2].Modified()
        self.vein_L_actors[3].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[3].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[3].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[4]))
        self.vein_L_actors[3].Modified()
        self.vein_L_actors[4].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[4].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[4].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[5]))
        self.vein_L_actors[4].Modified()
        self.vein_L_actors[5].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[5].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[5].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[6]))
        self.vein_L_actors[5].Modified()
        self.vein_L_actors[6].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[6].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[6].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[4]))
        self.vein_L_actors[6].Modified()
        self.vein_L_actors[7].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[7].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[7].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[5]))
        self.vein_L_actors[7].Modified()
        self.vein_L_actors[8].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[8].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[8].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[6]))
        self.vein_L_actors[8].Modified()
        self.vein_L_actors[9].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[9].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[9].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[4]))
        self.vein_L_actors[9].Modified()
        self.vein_L_actors[10].SetScale(scale_in[3],scale_in[3],scale_in[3])
        self.vein_L_actors[10].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_L_actors[10].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[4]))
        self.vein_L_actors[10].Modified()
        # wing R model
        # membrane
        self.mem_R_actors[0].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.mem_R_actors[0].GetProperty().SetColor(color_mem[0],color_mem[1],color_mem[2])
        self.mem_R_actors[0].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[7]))
        self.mem_R_actors[0].GetProperty().SetOpacity(opacity_mem)
        self.mem_R_actors[0].ForceTranslucentOn()
        self.mem_R_actors[0].Modified()
        self.mem_R_actors[1].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.mem_R_actors[1].GetProperty().SetColor(color_mem[0],color_mem[1],color_mem[2])
        self.mem_R_actors[1].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[8]))
        self.mem_R_actors[1].GetProperty().SetOpacity(opacity_mem)
        self.mem_R_actors[1].ForceTranslucentOn()
        self.mem_R_actors[1].Modified()
        self.mem_R_actors[2].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.mem_R_actors[2].GetProperty().SetColor(color_mem[0],color_mem[1],color_mem[2])
        self.mem_R_actors[2].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[9]))
        self.mem_R_actors[2].GetProperty().SetOpacity(opacity_mem)
        self.mem_R_actors[2].ForceTranslucentOn()
        self.mem_R_actors[2].Modified()
        self.mem_R_actors[3].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.mem_R_actors[3].GetProperty().SetColor(color_mem[0],color_mem[1],color_mem[2])
        self.mem_R_actors[3].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[10]))
        self.mem_R_actors[3].GetProperty().SetOpacity(opacity_mem)
        self.mem_R_actors[3].ForceTranslucentOn()
        self.mem_R_actors[3].Modified()

        # veins
        self.vein_R_actors[0].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[0].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[0].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[7]))
        self.vein_R_actors[0].Modified()
        self.vein_R_actors[1].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[1].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[1].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[7]))
        self.vein_R_actors[1].Modified()
        self.vein_R_actors[2].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[2].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[2].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[7]))
        self.vein_R_actors[2].Modified()
        self.vein_R_actors[3].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[3].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[3].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[8]))
        self.vein_R_actors[3].Modified()
        self.vein_R_actors[4].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[4].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[4].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[9]))
        self.vein_R_actors[4].Modified()
        self.vein_R_actors[5].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[5].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[5].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[10]))
        self.vein_R_actors[5].Modified()
        self.vein_R_actors[6].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[6].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[6].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[8]))
        self.vein_R_actors[6].Modified()
        self.vein_R_actors[7].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[7].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[7].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[9]))
        self.vein_R_actors[7].Modified()
        self.vein_R_actors[8].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[8].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[8].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[10]))
        self.vein_R_actors[8].Modified()
        self.vein_R_actors[9].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[9].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[9].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[8]))
        self.vein_R_actors[9].Modified()
        self.vein_R_actors[10].SetScale(scale_in[4],scale_in[4],scale_in[4])
        self.vein_R_actors[10].GetProperty().SetColor(color_vein[0],color_vein[1],color_vein[2])
        self.vein_R_actors[10].SetUserMatrix(self.convert_2_vtkMat(self.transform_M[8]))
        self.vein_R_actors[10].Modified()
        #
        #self.iren.Initialize()
        #self.renWin.Render()
        #self.iren.Start()

    def compute_transformation_mats(self,s_in):
        self.transform_M = []
        M_head = self.quat_mat(s_in[0:7])
        self.transform_M.append(M_head)
        M_thorax = self.quat_mat(s_in[7:14])
        self.transform_M.append(M_thorax)
        M_abdomen = self.quat_mat(s_in[14:21])
        self.transform_M.append(M_abdomen)
        M_L0 = self.quat_mat(s_in[21:28])
        self.transform_M.append(M_L0)
        b_L = s_in[28]/3.0
        M_L1 = self.M_axis_1_L(M_L0,b_L)
        self.transform_M.append(M_L1)
        M_L2 = self.M_axis_2_L(M_L1,b_L)
        self.transform_M.append(M_L2)
        M_L3 = self.M_axis_3_L(M_L2,b_L)
        self.transform_M.append(M_L3)
        M_R0 = self.quat_mat(s_in[29:36])
        self.transform_M.append(M_R0)
        b_R = s_in[36]/3.0
        M_R1 = self.M_axis_1_R(M_R0,b_R)
        self.transform_M.append(M_R1)
        M_R2 = self.M_axis_2_R(M_R1,b_R)
        self.transform_M.append(M_R2)
        M_R3 = self.M_axis_3_R(M_R2,b_R)
        self.transform_M.append(M_R3)

    def M_axis_1_L(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = 0.0
        q2 = 1.0*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        M_out[3,3] = 1.0
        return M_out

    def M_axis_2_L(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = -0.05959*np.sin(b_angle/2.0)
        q2 = 0.99822*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        M_out[3,3] = 1.0
        TA = np.dot(R_out,np.array([-0.0867,0.0145,0.0]))
        TB = np.dot(R_in,np.array([-0.0867,0.0145,0.0]))
        M_out[0:3,3] -= TA-TB
        return M_out

    def M_axis_3_L(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = -0.36186*np.sin(b_angle/2.0)
        q2 = 0.93223*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        M_out[3,3] = 1.0
        TA = np.dot(R_out,np.array([-0.0867,0.0145,0.0]))
        TB = np.dot(R_in,np.array([-0.0867,0.0145,0.0]))
        M_out[0:3,3] -= TA-TB
        return M_out

    def M_axis_1_R(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = 0.0
        q2 = 1.0*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        M_out[3,3] = 1.0
        return M_out

    def M_axis_2_R(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = 0.05959*np.sin(b_angle/2.0)
        q2 = 0.99822*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        TA = np.dot(R_out,np.array([-0.0867,-0.0145,0.0]))
        TB = np.dot(R_in,np.array([-0.0867,-0.0145,0.0]))
        M_out[0:3,3] -= TA-TB
        M_out[3,3] = 1.0
        return M_out

    def M_axis_3_R(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = 0.36186*np.sin(b_angle/2.0)
        q2 = 0.93223*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        TA = np.dot(R_out,np.array([-0.0867,-0.0145,0.0]))
        TB = np.dot(R_in,np.array([-0.0867,-0.0145,0.0]))
        M_out[0:3,3] -= TA-TB
        M_out[3,3] = 1.0
        return M_out

    def set_start_state(self):
        self.start_state = np.zeros(37)
        # thorax
        q_x = np.array([np.cos(0.0),np.sin(0.0),0.0,0.0])
        q_y = np.array([np.cos((-45.0/180.0)*np.pi/2.0),0.0,np.sin((-45.0/180.0)*np.pi/2.0),0.0])
        q_z = np.array([np.cos(0.0),np.sin(0.0),0.0,0.0])
        q_f = self.quat_multiply(q_z,self.quat_multiply(q_y,q_x))
        s_thorax = np.array([q_f[0],q_f[1],q_f[2],q_f[3],0.0,0.0,0.0])
        M_thorax = self.quat_mat(s_thorax)
        # head
        q_x = np.array([np.cos(0.0),np.sin(0.0),0.0,0.0])
        q_y = np.array([np.cos((30.0/180.0)*np.pi/2.0),0.0,np.sin((30.0/180.0)*np.pi/2.0),0.0])
        q_z = np.array([np.cos(0.0),0.0,0.0,np.sin(0.0)])
        q_f = self.quat_multiply(q_z,self.quat_multiply(q_y,q_x))
        s_head = np.array([q_f[0],q_f[1],q_f[2],q_f[3],0.0,0.0,0.0])
        M_head = self.quat_mat(s_head)
        M_j_head = np.array([[1.0,0.0,0.0,0.7*self.scale[1]],
            [0.0, 1.0, 0.0, 0.0*self.scale[1]],
            [0.0, 0.0, 1.0, 0.0*self.scale[1]],
            [0.0, 0.0, 0.0, 1.0]])
        M_now = self.aff_mat_multiply(self.aff_mat_multiply(M_thorax,M_j_head),M_head)
        q_now = self.quat_multiply(s_thorax[0:4],s_head[0:4])
        s_head[0:4] = q_now
        s_head[4] = M_now[0,3]
        s_head[5] = M_now[1,3]
        s_head[6] = M_now[2,3]
        # abdomen
        q_x = np.array([np.cos(0.0),np.sin(0.0),0.0,0.0])
        q_y = np.array([np.cos((-30.0/180.0)*np.pi/2.0),0.0,np.sin((-30.0/180.0)*np.pi/2.0),0.0])
        q_z = np.array([np.cos(0.0),0.0,0.0,np.sin(0.0)])
        q_f = self.quat_multiply(q_z,self.quat_multiply(q_y,q_x))
        s_abdomen = np.array([q_f[0],q_f[1],q_f[2],q_f[3],0.0,0.0,0.0])
        M_abdomen = self.quat_mat(s_abdomen)
        M_j_abdomen = np.array([[1.0,0.0,0.0,-0.09*self.scale[1]],
            [0.0, 1.0, 0.0, 0.0*self.scale[1]],
            [0.0, 0.0, 1.0, -0.09*self.scale[1]],
            [0.0, 0.0, 0.0, 1.0]])
        M_now = self.aff_mat_multiply(self.aff_mat_multiply(M_thorax,M_j_abdomen),M_abdomen)
        q_now = self.quat_multiply(s_thorax[0:4],s_abdomen[0:4])
        s_abdomen[0:4] = q_now
        s_abdomen[4] = M_now[0,3]
        s_abdomen[5] = M_now[1,3]
        s_abdomen[6] = M_now[2,3]
        # wing L
        s_wing_L = np.array([np.cos((-45.0/180.0)*np.pi/2.0),0.0,np.sin((-45.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0,0.0])
        M_wing_L = self.quat_mat(s_wing_L[:7])
        M_j_wing_L = np.array([[1.0,0.0,0.0,0.0*self.scale[1]],
            [0.0, 1.0, 0.0, 0.6*self.scale[1]],
            [0.0, 0.0, 1.0, 0.0*self.scale[1]],
            [0.0, 0.0, 0.0, 1.0]])
        M_now = self.aff_mat_multiply(self.aff_mat_multiply(M_thorax,M_j_wing_L),M_wing_L)
        q_now = self.quat_multiply(s_thorax[0:4],s_wing_L[0:4])
        s_wing_L[0:4] = q_now
        s_wing_L[4] = M_now[0,3]
        s_wing_L[5] = M_now[1,3]
        s_wing_L[6] = M_now[2,3]
        # wing R
        s_wing_R = np.array([np.cos((-45.0/180.0)*np.pi/2.0),0.0,np.sin((-45.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0,0.0])
        M_wing_R = self.quat_mat(s_wing_R[:7])
        M_j_wing_R = np.array([[1.0,0.0,0.0,0.0*self.scale[1]],
            [0.0, 1.0, 0.0, -0.6*self.scale[1]],
            [0.0, 0.0, 1.0, 0.0*self.scale[1]],
            [0.0, 0.0, 0.0, 1.0]])
        M_now = self.aff_mat_multiply(self.aff_mat_multiply(M_thorax,M_j_wing_R),M_wing_R)
        q_now = self.quat_multiply(s_thorax[0:4],s_wing_R[0:4])
        s_wing_R[0:4] = q_now
        s_wing_R[4] = M_now[0,3]
        s_wing_R[5] = M_now[1,3]
        s_wing_R[6] = M_now[2,3]
        # set state
        self.start_state[0:7] = s_head
        self.start_state[7:14] = s_thorax
        self.start_state[14:21] = s_abdomen
        self.start_state[21:29] = s_wing_L
        self.start_state[29:37] = s_wing_R
        self.set_model_state(self.start_state)
        self.qx = [0.0,0.0,0.0,0.0,0.0]
        self.qy = [0.0,0.0,0.0,0.0,0.0]
        self.qz = [0.0,0.0,0.0,0.0,0.0]
        self.tx = [self.state[4],self.state[11],self.state[18],self.state[25],self.state[33]]
        self.ty = [self.state[5],self.state[12],self.state[19],self.state[26],self.state[34]]
        self.tz = [self.state[6],self.state[13],self.state[20],self.state[27],self.state[35]]
        self.xi = [0,0,0,self.state[28],self.state[36]]

    def update_state(self):
        # thorax
        q_x = np.array([np.cos(self.qx[1]*np.pi/2.0),np.sin(self.qx[1]*np.pi/2.0),0.0,0.0])
        q_y = np.array([np.cos(self.qy[1]*np.pi/2.0),0.0,np.sin(self.qy[1]*np.pi/2.0),0.0])
        q_z = np.array([np.cos(self.qz[1]*np.pi/2.0),0.0,0.0,np.sin(self.qz[1]*np.pi/2.0)])
        q_f = self.quat_multiply(q_z,self.quat_multiply(q_y,q_x))
        s_thorax = np.array([q_f[0],q_f[1],q_f[2],q_f[3],self.tx[1],self.ty[1],self.tz[1]])
        # head
        q_x = np.array([np.cos(self.qx[0]*np.pi/2.0),np.sin(self.qx[0]*np.pi/2.0),0.0,0.0])
        q_y = np.array([np.cos(self.qy[0]*np.pi/2.0),0.0,np.sin(self.qy[0]*np.pi/2.0),0.0])
        q_z = np.array([np.cos(self.qz[0]*np.pi/2.0),0.0,0.0,np.sin(self.qz[0]*np.pi/2.0)])
        q_f = self.quat_multiply(q_z,self.quat_multiply(q_y,q_x))
        s_head = np.array([q_f[0],q_f[1],q_f[2],q_f[3],self.tx[0],self.ty[0],self.tz[0]])
        # abdomen
        q_x = np.array([np.cos(self.qx[2]*np.pi/2.0),np.sin(self.qx[2]*np.pi/2.0),0.0,0.0])
        q_y = np.array([np.cos(self.qy[2]*np.pi/2.0),0.0,np.sin(self.qy[2]*np.pi/2.0),0.0])
        q_z = np.array([np.cos(self.qz[2]*np.pi/2.0),0.0,0.0,np.sin(self.qz[2]*np.pi/2.0)])
        q_f = self.quat_multiply(q_z,self.quat_multiply(q_y,q_x))
        s_abdomen = np.array([q_f[0],q_f[1],q_f[2],q_f[3],self.tx[2],self.ty[2],self.tz[2]])
        # wing L
        q_x = np.array([np.cos(self.qx[3]*np.pi/2.0),np.sin(self.qx[3]*np.pi/2.0),0.0,0.0])
        q_y = np.array([np.cos(self.qy[3]*np.pi/2.0),0.0,np.sin(self.qy[3]*np.pi/2.0),0.0])
        q_z = np.array([np.cos(self.qz[3]*np.pi/2.0),0.0,0.0,np.sin(self.qz[3]*np.pi/2.0)])
        q_f = self.quat_multiply(q_z,self.quat_multiply(q_y,q_x))
        s_wing_L = np.array([q_f[0],q_f[1],q_f[2],q_f[3],self.tx[3],self.ty[3],self.tz[3],self.xi[3]])
        # wing R
        q_x = np.array([np.cos(self.qx[4]*np.pi/2.0),np.sin(self.qx[4]*np.pi/2.0),0.0,0.0])
        q_y = np.array([np.cos(self.qy[4]*np.pi/2.0),0.0,np.sin(self.qy[4]*np.pi/2.0),0.0])
        q_z = np.array([np.cos(self.qz[4]*np.pi/2.0),0.0,0.0,np.sin(self.qz[4]*np.pi/2.0)])
        q_f = self.quat_multiply(q_z,self.quat_multiply(q_y,q_x))
        s_wing_R = np.array([q_f[0],q_f[1],q_f[2],q_f[3],self.tx[4],self.ty[4],self.tz[4],self.xi[4]])
        # set state
        self.state[0:7] = s_head
        self.state[7:14] = s_thorax
        self.state[14:21] = s_abdomen
        self.state[21:29] = s_wing_L
        self.state[29:37] = s_wing_R
        # display
        self.set_model_state(self.state)

    def aff_mat_multiply(self,M_A, M_B):
        R_A = M_A[0:3,0:3]
        R_B = M_B[0:3,0:3]
        T_A = M_A[0:3,3]
        T_B = M_B[0:3,3]
        M_C = np.eye(4)
        M_C[0:3,0:3] = np.dot(R_A,R_B)
        T_C = np.dot(R_A,T_B)+T_A
        M_C[0:3,3] = T_C
        return M_C

    def quat_multiply(self,q_A,q_B):
        QA = np.squeeze(np.array([[q_A[0],-q_A[1],-q_A[2],-q_A[3]],
            [q_A[1],q_A[0],-q_A[3],q_A[2]],
            [q_A[2],q_A[3],q_A[0],-q_A[1]],
            [q_A[3],-q_A[2],q_A[1],q_A[0]]]))
        q_C = np.dot(QA,q_B)
        q_C /= math.sqrt(pow(q_C[0],2)+pow(q_C[1],2)+pow(q_C[2],2)+pow(q_C[3],2))
        return q_C

    def quat_mat(self,s_in):
        q0 = np.squeeze(s_in[0])
        q1 = np.squeeze(s_in[1])
        q2 = np.squeeze(s_in[2])
        q3 = np.squeeze(s_in[3])
        tx = np.squeeze(s_in[4])
        ty = np.squeeze(s_in[5])
        tz = np.squeeze(s_in[6])
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        if q_norm > 0.01:
            q0 /= q_norm
            q1 /= q_norm
            q2 /= q_norm
            q3 /= q_norm
        else:
            q0 = 1.0
            q1 = 0.0
            q2 = 0.0
            q3 = 0.0
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, tx],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1, ty],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        return M

    def convert_2_vtkMat(self,M):
        M_vtk = vtk.vtkMatrix4x4()
        M_vtk.SetElement(0,0,M[0,0])
        M_vtk.SetElement(0,1,M[0,1])
        M_vtk.SetElement(0,2,M[0,2])
        M_vtk.SetElement(0,3,M[0,3])
        M_vtk.SetElement(1,0,M[1,0])
        M_vtk.SetElement(1,1,M[1,1])
        M_vtk.SetElement(1,2,M[1,2])
        M_vtk.SetElement(1,3,M[1,3])
        M_vtk.SetElement(2,0,M[2,0])
        M_vtk.SetElement(2,1,M[2,1])
        M_vtk.SetElement(2,2,M[2,2])
        M_vtk.SetElement(2,3,M[2,3])
        M_vtk.SetElement(3,0,M[3,0])
        M_vtk.SetElement(3,1,M[3,1])
        M_vtk.SetElement(3,2,M[3,2])
        M_vtk.SetElement(3,3,M[3,3])
        return M_vtk

    def set_v_list(self,v_list_in):
        self.v_list = v_list_in

    def display_curves(self):
        self.remove_curves()
        head_pts = np.dot(self.transform_M[0],self.body_pts_scaled[0])
        thorax_pts = np.dot(self.transform_M[1],self.body_pts_scaled[1])
        abdomen_pts = np.dot(self.transform_M[2],self.body_pts_scaled[2])
        vein_0_L_pts = np.dot(self.transform_M[3],self.vein_L_pts_scaled[0])
        vein_1_L_pts = np.dot(self.transform_M[3],self.vein_L_pts_scaled[1])
        vein_2_L_pts = np.dot(self.transform_M[3],self.vein_L_pts_scaled[2])
        vein_3_L_pts = np.dot(self.transform_M[3],self.vein_L_pts_scaled[3])
        vein_4_L_pts = np.dot(self.transform_M[5],self.vein_L_pts_scaled[4])
        vein_5_L_pts = np.dot(self.transform_M[5],self.vein_L_pts_scaled[5])
        vein_C1_L_pts = np.dot(self.transform_M[4],self.vein_L_pts_scaled[6])
        vein_C2_L_pts = np.dot(self.transform_M[5],self.vein_L_pts_scaled[7])
        vein_C3_L_pts = np.dot(self.transform_M[6],self.vein_L_pts_scaled[8])
        vein_A_L_pts = np.dot(self.transform_M[4],self.vein_L_pts_scaled[9])
        vein_P_L_pts = np.dot(self.transform_M[5],self.vein_L_pts_scaled[10])
        vein_0_R_pts = np.dot(self.transform_M[7],self.vein_R_pts_scaled[0])
        vein_1_R_pts = np.dot(self.transform_M[7],self.vein_R_pts_scaled[1])
        vein_2_R_pts = np.dot(self.transform_M[7],self.vein_R_pts_scaled[2])
        vein_3_R_pts = np.dot(self.transform_M[7],self.vein_R_pts_scaled[3])
        vein_4_R_pts = np.dot(self.transform_M[9],self.vein_R_pts_scaled[4])
        vein_5_R_pts = np.dot(self.transform_M[9],self.vein_R_pts_scaled[5])
        vein_C1_R_pts = np.dot(self.transform_M[8],self.vein_R_pts_scaled[6])
        vein_C2_R_pts = np.dot(self.transform_M[9],self.vein_R_pts_scaled[7])
        vein_C3_R_pts = np.dot(self.transform_M[10],self.vein_R_pts_scaled[8])
        vein_A_R_pts = np.dot(self.transform_M[8],self.vein_R_pts_scaled[9])
        vein_P_R_pts = np.dot(self.transform_M[9],self.vein_R_pts_scaled[10])
        select_clr = [255,0,0]
        off_clr = [255,255,0]
        if self.comp_ind == 0:
            if self.full_view:
                self.compute_curves(self.body_vertices[0],head_pts,off_clr)
            else:
                self.compute_curves(self.body_vertices[0],head_pts,select_clr)
        else:
            if self.full_view:
                self.compute_curves(self.body_vertices[0],head_pts,off_clr)
        if self.comp_ind == 1:
            if self.full_view:
                self.compute_curves(self.body_vertices[1],thorax_pts,off_clr)
            else:
                self.compute_curves(self.body_vertices[1],thorax_pts,select_clr)
        else:
            if self.full_view:
                self.compute_curves(self.body_vertices[1],thorax_pts,off_clr)
        if self.comp_ind == 2:
            if self.full_view:
                self.compute_curves(self.body_vertices[2],abdomen_pts,off_clr)
            else:
                self.compute_curves(self.body_vertices[2],abdomen_pts,select_clr)
        else:
            if self.full_view:
                self.compute_curves(self.body_vertices[2],abdomen_pts,off_clr)
        if self.comp_ind == 3:
            if self.full_view:
                self.compute_curves(self.vein_L_vertices[0],vein_0_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[1],vein_1_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[2],vein_2_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[3],vein_3_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[4],vein_4_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[5],vein_5_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[6],vein_C1_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[7],vein_C2_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[8],vein_C3_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[9],vein_A_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[10],vein_P_L_pts,off_clr)
            else:
                self.compute_curves(self.vein_L_vertices[0],vein_0_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[1],vein_1_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[2],vein_2_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[3],vein_3_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[4],vein_4_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[5],vein_5_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[6],vein_C1_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[7],vein_C2_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[8],vein_C3_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[9],vein_A_L_pts,select_clr)
                self.compute_curves(self.vein_L_vertices[10],vein_P_L_pts,select_clr)
        else:
            if self.full_view:
                self.compute_curves(self.vein_L_vertices[0],vein_0_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[1],vein_1_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[2],vein_2_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[3],vein_3_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[4],vein_4_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[5],vein_5_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[6],vein_C1_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[7],vein_C2_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[8],vein_C3_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[9],vein_A_L_pts,off_clr)
                self.compute_curves(self.vein_L_vertices[10],vein_P_L_pts,off_clr)
        if self.comp_ind == 4:
            if self.full_view:
                self.compute_curves(self.vein_R_vertices[0],vein_0_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[1],vein_1_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[2],vein_2_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[3],vein_3_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[4],vein_4_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[5],vein_5_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[6],vein_C1_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[7],vein_C2_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[8],vein_C3_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[9],vein_A_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[10],vein_P_R_pts,off_clr)
            else:
                self.compute_curves(self.vein_R_vertices[0],vein_0_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[1],vein_1_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[2],vein_2_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[3],vein_3_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[4],vein_4_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[5],vein_5_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[6],vein_C1_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[7],vein_C2_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[8],vein_C3_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[9],vein_A_R_pts,select_clr)
                self.compute_curves(self.vein_R_vertices[10],vein_P_R_pts,select_clr)
        else:
            if self.full_view:
                self.compute_curves(self.vein_R_vertices[0],vein_0_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[1],vein_1_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[2],vein_2_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[3],vein_3_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[4],vein_4_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[5],vein_5_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[6],vein_C1_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[7],vein_C2_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[8],vein_C3_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[9],vein_A_R_pts,off_clr)
                self.compute_curves(self.vein_R_vertices[10],vein_P_R_pts,off_clr)

    def compute_curves(self,verts_in,pts_in,color_in):
        # Project to camera views:
        for n in range(self.N_cam):
            uv_pts = self.project_2_uv(pts_in,n)
            u_pts = np.transpose(uv_pts[0,verts_in])
            v_pts = np.transpose(uv_pts[1,verts_in])
            curve = pg.PlotCurveItem(x=u_pts,y=v_pts,pen=color_in,connect='pairs')
            self.curves[n].append(curve)
            self.v_list[n].addItem(curve)

    def remove_curves(self):
        if len(self.curves)>0:
            for n in range(self.N_cam):
                N_curves = len(self.curves[n])
                for i in range(N_curves):
                    self.v_list[n].removeItem(self.curves[n][i])
        self.curves = []
        for n in range(self.N_cam):
            self.curves.append([])

    def set_model_view_toggle(self,toggle_in):
        self.full_view_toggle = toggle_in
        self.full_view_toggle.setChecked(True)
        self.full_view = True
        self.full_view_toggle.toggled.connect(self.full_view_toggled)

    def full_view_toggled(self):
        if self.full_view_toggle.isChecked() == True:
            self.full_view = True
            self.display_curves()

    def set_single_view_toggle(self,toggle_in):
        self.single_view_toggle = toggle_in
        self.single_view_toggle.setChecked(False)
        self.single_view_toggle.toggled.connect(self.single_view_toggled)

    def single_view_toggled(self):
        if self.single_view_toggle.isChecked() == True:
            self.full_view = False
            self.display_curves()

    def set_off_view_toggle(self,toggle_in):
        self.off_view_toggle = toggle_in
        self.off_view_toggle.setChecked(False)
        self.off_view_toggle.toggled.connect(self.off_view_toggled)

    def off_view_toggled(self):
        if self.off_view_toggle.isChecked() == True:
            self.full_view = False
            self.remove_curves()

    def set_component_combo(self,combo_in):
        self.component_combo = combo_in
        for comp in self.mdl_components:
            self.component_combo.addItem(comp)
        self.component_combo.currentIndexChanged.connect(self.component_selected)

    def component_selected(self,ind):
        self.comp_ind = ind-1
        if self.comp_ind >= 0 and self.comp_ind < 3:
            self.qx_spin.setValue(self.qx[self.comp_ind])
            self.qy_spin.setValue(self.qy[self.comp_ind])
            self.qz_spin.setValue(self.qz[self.comp_ind])
            self.tx_spin.setValue(self.tx[self.comp_ind])
            self.ty_spin.setValue(self.ty[self.comp_ind])
            self.tz_spin.setValue(self.tz[self.comp_ind])
            self.scale_spin.setValue(self.scale[self.comp_ind])
            self.xi_spin.setMinimum(0.0)
            self.xi_spin.setMaximum(0.0)
            self.xi_spin.setValue(0.0)
        if self.comp_ind > 2:
            self.qx_spin.setValue(self.qx[self.comp_ind])
            self.qy_spin.setValue(self.qy[self.comp_ind])
            self.qz_spin.setValue(self.qz[self.comp_ind])
            self.tx_spin.setValue(self.tx[self.comp_ind])
            self.ty_spin.setValue(self.ty[self.comp_ind])
            self.tz_spin.setValue(self.tz[self.comp_ind])
            self.scale_spin.setValue(self.scale[self.comp_ind])
            self.xi_spin.setMinimum(-2.0)
            self.xi_spin.setMaximum(2.0)
            self.xi_spin.setValue(self.xi[self.comp_ind])

    def set_qx_spin(self,spin_in):
        self.qx_spin = spin_in
        self.qx_spin.setMinimum(-2.0)
        self.qx_spin.setMaximum(2.0)
        self.qx_spin.setDecimals(3)
        self.qx_spin.setSingleStep(0.001)
        self.qx_spin.setValue(0.0)
        self.qx = [0.0,0.0,0.0,0.0,0.0]
        self.set_qx(0.0)
        self.qx_spin.valueChanged.connect(self.set_qx)

    def set_qx(self,q_in):
        if self.comp_ind >= 0:
            delta_q = np.pi/2.0*(self.qx[self.comp_ind]-q_in)
            self.qx[self.comp_ind] = q_in
            if self.comp_ind == 0:
                q_prev = self.state[0:4]
                q_x = np.array([np.cos(delta_q),np.sin(delta_q),0.0,0.0])
                q_now = self.quat_multiply(q_prev,q_x)
                self.state[0:4] = q_now
            elif self.comp_ind == 1:
                q_prev = self.state[7:11]
                q_x = np.array([np.cos(delta_q),np.sin(delta_q),0.0,0.0])
                q_now = self.quat_multiply(q_prev,q_x)
                self.state[7:11] = q_now
            elif self.comp_ind == 2:
                q_prev = self.state[14:18]
                q_x = np.array([np.cos(delta_q),np.sin(delta_q),0.0,0.0])
                q_now = self.quat_multiply(q_prev,q_x)
                self.state[14:18] = q_now
            elif self.comp_ind == 3:
                q_prev = self.state[21:25]
                q_x = np.array([np.cos(delta_q),np.sin(delta_q),0.0,0.0])
                q_now = self.quat_multiply(q_prev,q_x)
                self.state[21:25] = q_now
            elif self.comp_ind == 4:
                q_prev = self.state[29:33]
                q_x = np.array([np.cos(delta_q),np.sin(delta_q),0.0,0.0])
                q_now = self.quat_multiply(q_prev,q_x)
                self.state[29:33] = q_now
            self.set_model_state(self.state)

    def set_qy_spin(self,spin_in):
        self.qy_spin = spin_in
        self.qy_spin.setMinimum(-2.0)
        self.qy_spin.setMaximum(2.0)
        self.qy_spin.setDecimals(3)
        self.qy_spin.setSingleStep(0.001)
        self.qy_spin.setValue(0.0)
        self.qy = [0.0,0.0,0.0,0.0,0.0]
        self.set_qy(0.0)
        self.qy_spin.valueChanged.connect(self.set_qy)

    def set_qy(self,q_in):
        if self.comp_ind >= 0:
            delta_q = np.pi/2.0*(self.qy[self.comp_ind]-q_in)
            self.qy[self.comp_ind] = q_in
            if self.comp_ind == 0:
                q_prev = self.state[0:4]
                q_y = np.array([np.cos(delta_q),0.0,np.sin(delta_q),0.0])
                q_now = self.quat_multiply(q_prev,q_y)
                self.state[0:4] = q_now
            elif self.comp_ind == 1:
                q_prev = self.state[7:11]
                q_y = np.array([np.cos(delta_q),0.0,np.sin(delta_q),0.0])
                q_now = self.quat_multiply(q_prev,q_y)
                self.state[7:11] = q_now
            elif self.comp_ind == 2:
                q_prev = self.state[14:18]
                q_y = np.array([np.cos(delta_q),0.0,np.sin(delta_q),0.0])
                q_now = self.quat_multiply(q_prev,q_y)
                self.state[14:18] = q_now
            elif self.comp_ind == 3:
                q_prev = self.state[21:25]
                q_y = np.array([np.cos(delta_q),0.0,np.sin(delta_q),0.0])
                q_now = self.quat_multiply(q_prev,q_y)
                self.state[21:25] = q_now
            elif self.comp_ind == 4:
                q_prev = self.state[29:33]
                q_y = np.array([np.cos(delta_q),0.0,np.sin(delta_q),0.0])
                q_now = self.quat_multiply(q_prev,q_y)
                self.state[29:33] = q_now
            self.set_model_state(self.state)

    def set_qz_spin(self,spin_in):
        self.qz_spin = spin_in
        self.qz_spin.setMinimum(-2.0)
        self.qz_spin.setMaximum(2.0)
        self.qz_spin.setDecimals(3)
        self.qz_spin.setSingleStep(0.001)
        self.qz_spin.setValue(0.0)
        self.qz = [0.0,0.0,0.0,0.0,0.0]
        self.set_qz(0.0)
        self.qz_spin.valueChanged.connect(self.set_qz)

    def set_qz(self,q_in):
        if self.comp_ind >= 0:
            delta_q = np.pi/2.0*(self.qz[self.comp_ind]-q_in)
            self.qz[self.comp_ind] = q_in
            if self.comp_ind == 0:
                q_prev = self.state[0:4]
                q_z = np.array([np.cos(delta_q),0.0,0.0,np.sin(delta_q)])
                q_now = self.quat_multiply(q_prev,q_z)
                self.state[0:4] = q_now
            elif self.comp_ind == 1:
                q_prev = self.state[7:11]
                q_z = np.array([np.cos(delta_q),0.0,0.0,np.sin(delta_q)])
                q_now = self.quat_multiply(q_prev,q_z)
                self.state[7:11] = q_now
            elif self.comp_ind == 2:
                q_prev = self.state[14:18]
                q_z = np.array([np.cos(delta_q),0.0,0.0,np.sin(delta_q)])
                q_now = self.quat_multiply(q_prev,q_z)
                self.state[14:18] = q_now
            elif self.comp_ind == 3:
                q_prev = self.state[21:25]
                q_z = np.array([np.cos(delta_q),0.0,0.0,np.sin(delta_q)])
                q_now = self.quat_multiply(q_prev,q_z)
                self.state[21:25] = q_now
            elif self.comp_ind == 4:
                q_prev = self.state[29:33]
                q_z = np.array([np.cos(delta_q),0.0,0.0,np.sin(delta_q)])
                q_now = self.quat_multiply(q_prev,q_z)
                self.state[29:33] = q_now
            self.set_model_state(self.state)

    def set_tx_spin(self,spin_in):
        self.tx_spin = spin_in
        self.tx_spin.setMinimum(-10.0)
        self.tx_spin.setMaximum(10.0)
        self.tx_spin.setDecimals(2)
        self.tx_spin.setSingleStep(0.01)
        self.tx_spin.setValue(0.0)
        self.tx = [0.0,0.0,0.0,0.0,0.0]
        self.set_tx(0.0)
        self.tx_spin.valueChanged.connect(self.set_tx)

    def set_tx(self,val_in):
        if self.comp_ind >= 0:
            self.tx[self.comp_ind] = val_in
            if self.comp_ind == 0:
                self.state[4] = val_in
            elif self.comp_ind == 1:
                self.state[11] = val_in
            elif self.comp_ind == 2:
                self.state[18] = val_in
            elif self.comp_ind == 3:
                self.state[25] = val_in
            elif self.comp_ind == 4:
                self.state[33] = val_in
            self.set_model_state(self.state)

    def set_ty_spin(self,spin_in):
        self.ty_spin = spin_in
        self.ty_spin.setMinimum(-10.0)
        self.ty_spin.setMaximum(10.0)
        self.ty_spin.setDecimals(2)
        self.ty_spin.setSingleStep(0.01)
        self.ty_spin.setValue(0.0)
        self.ty = [0.0,0.0,0.0,0.0,0.0]
        self.set_ty(0.0)
        self.ty_spin.valueChanged.connect(self.set_ty)

    def set_ty(self,val_in):
        if self.comp_ind >= 0:
            self.ty[self.comp_ind] = val_in
            if self.comp_ind == 0:
                self.state[5] = val_in
            elif self.comp_ind == 1:
                self.state[12] = val_in
            elif self.comp_ind == 2:
                self.state[19] = val_in
            elif self.comp_ind == 3:
                self.state[26] = val_in
            elif self.comp_ind == 4:
                self.state[34] = val_in
            self.set_model_state(self.state)

    def set_tz_spin(self,spin_in):
        self.tz_spin = spin_in
        self.tz_spin.setMinimum(-10.0)
        self.tz_spin.setMaximum(10.0)
        self.tz_spin.setDecimals(2)
        self.tz_spin.setSingleStep(0.01)
        self.tz_spin.setValue(0.0)
        self.tz = [0.0,0.0,0.0,0.0,0.0]
        self.set_tz(0.0)
        self.tz_spin.valueChanged.connect(self.set_tz)

    def set_tz(self,val_in):
        if self.comp_ind >= 0:
            self.tz[self.comp_ind] = val_in
            if self.comp_ind == 0:
                self.state[6] = val_in
            elif self.comp_ind == 1:
                self.state[13] = val_in
            elif self.comp_ind == 2:
                self.state[20] = val_in
            elif self.comp_ind == 3:
                self.state[27] = val_in
            elif self.comp_ind == 4:
                self.state[35] = val_in
            self.set_model_state(self.state)

    def set_xi_spin(self,spin_in):
        self.xi_spin = spin_in
        self.xi_spin.setMinimum(-2.0)
        self.xi_spin.setMaximum(2.0)
        self.xi_spin.setDecimals(2)
        self.xi_spin.setSingleStep(0.01)
        self.xi_spin.setValue(0.0)
        self.xi = [0.0,0.0,0.0,0.0,0.0]
        self.set_xi(0.0)
        self.xi_spin.valueChanged.connect(self.set_xi)

    def set_xi(self,val_in):
        if self.comp_ind > 2:
            self.xi[self.comp_ind] = val_in
            if self.comp_ind == 3:
                self.state[28] = val_in
            elif self.comp_ind == 4:
                self.state[36] = val_in
            self.set_model_state(self.state)

    def set_scale_spin(self,spin_in):
        self.scale_spin = spin_in
        self.scale_spin.setMinimum(0.1)
        self.scale_spin.setMaximum(3.0)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setSingleStep(0.001)
        self.scale_spin.setValue(1.0)
        self.scale = [1.0,1.0,1.0,1.0,1.0]
        self.scale_spin.valueChanged.connect(self.set_scale)

    def set_scale(self,scale_in):
        if self.comp_ind >= 0:
            self.scale[self.comp_ind] = scale_in
            self.set_model_scale(self.scale)
            self.set_model_state(self.state)
            #self.update_scale_table()

    def set_scale_table(self,table_in):
        self.scale_table = table_in
        self.scale_table.setRowCount(5)
        self.scale_table.setColumnCount(2)
        self.scale_table.setItem(0,0,QTableWidgetItem('head scale:'))
        self.scale_table.setItem(1,0,QTableWidgetItem('thorax scale:'))
        self.scale_table.setItem(2,0,QTableWidgetItem('abdomen scale:'))
        self.scale_table.setItem(3,0,QTableWidgetItem('wing left scale:'))
        self.scale_table.setItem(4,0,QTableWidgetItem('wing right scale:'))
        self.scale_table.setItem(0,1,QTableWidgetItem(str(self.scale[0])))
        self.scale_table.setItem(1,1,QTableWidgetItem(str(self.scale[1])))
        self.scale_table.setItem(2,1,QTableWidgetItem(str(self.scale[2])))
        self.scale_table.setItem(3,1,QTableWidgetItem(str(self.scale[3])))
        self.scale_table.setItem(4,1,QTableWidgetItem(str(self.scale[4])))

    def update_scale_table(self):
        self.scale_table.setItem(0,1,QTableWidgetItem(str(self.scale[0])))
        self.scale_table.setItem(1,1,QTableWidgetItem(str(self.scale[1])))
        self.scale_table.setItem(2,1,QTableWidgetItem(str(self.scale[2])))
        self.scale_table.setItem(3,1,QTableWidgetItem(str(self.scale[3])))
        self.scale_table.setItem(4,1,QTableWidgetItem(str(self.scale[4])))
        self.scale_table.resizeColumnsToContents()

    def set_state_table(self,table_in):
        self.state_table = table_in
        self.state_table.setRowCount(37)
        self.state_table.setColumnCount(2)
        self.state_table.setItem(0,0,QTableWidgetItem('head q0:'))
        self.state_table.setItem(1,0,QTableWidgetItem('head q1:'))
        self.state_table.setItem(2,0,QTableWidgetItem('head q2:'))
        self.state_table.setItem(3,0,QTableWidgetItem('head q3:'))
        self.state_table.setItem(4,0,QTableWidgetItem('head tx:'))
        self.state_table.setItem(5,0,QTableWidgetItem('head ty:'))
        self.state_table.setItem(6,0,QTableWidgetItem('head tz:'))
        self.state_table.setItem(7,0,QTableWidgetItem('thorax q0:'))
        self.state_table.setItem(8,0,QTableWidgetItem('thorax q1:'))
        self.state_table.setItem(9,0,QTableWidgetItem('thorax q2:'))
        self.state_table.setItem(10,0,QTableWidgetItem('thorax q3:'))
        self.state_table.setItem(11,0,QTableWidgetItem('thorax tx:'))
        self.state_table.setItem(12,0,QTableWidgetItem('thorax ty:'))
        self.state_table.setItem(13,0,QTableWidgetItem('thorax tz:'))
        self.state_table.setItem(14,0,QTableWidgetItem('abdomen q0:'))
        self.state_table.setItem(15,0,QTableWidgetItem('abdomen q1:'))
        self.state_table.setItem(16,0,QTableWidgetItem('abdomen q2:'))
        self.state_table.setItem(17,0,QTableWidgetItem('abdomen q3:'))
        self.state_table.setItem(18,0,QTableWidgetItem('abdomen tx:'))
        self.state_table.setItem(19,0,QTableWidgetItem('abdomen ty:'))
        self.state_table.setItem(20,0,QTableWidgetItem('abdomen tz:'))
        self.state_table.setItem(21,0,QTableWidgetItem('wing L q0:'))
        self.state_table.setItem(22,0,QTableWidgetItem('wing L q1:'))
        self.state_table.setItem(23,0,QTableWidgetItem('wing L q2:'))
        self.state_table.setItem(24,0,QTableWidgetItem('wing L q3:'))
        self.state_table.setItem(25,0,QTableWidgetItem('wing L tx:'))
        self.state_table.setItem(26,0,QTableWidgetItem('wing L ty:'))
        self.state_table.setItem(27,0,QTableWidgetItem('wing L tz:'))
        self.state_table.setItem(28,0,QTableWidgetItem('wing L xi:'))
        self.state_table.setItem(29,0,QTableWidgetItem('wing R q0:'))
        self.state_table.setItem(30,0,QTableWidgetItem('wing R q1:'))
        self.state_table.setItem(31,0,QTableWidgetItem('wing R q2:'))
        self.state_table.setItem(32,0,QTableWidgetItem('wing R q3:'))
        self.state_table.setItem(33,0,QTableWidgetItem('wing R tx:'))
        self.state_table.setItem(34,0,QTableWidgetItem('wing R ty:'))
        self.state_table.setItem(35,0,QTableWidgetItem('wing R tz:'))
        self.state_table.setItem(36,0,QTableWidgetItem('wing R xi:'))
        self.state_table.setItem(0,1,QTableWidgetItem(str(self.state[0])))
        self.state_table.setItem(1,1,QTableWidgetItem(str(self.state[1])))
        self.state_table.setItem(2,1,QTableWidgetItem(str(self.state[2])))
        self.state_table.setItem(3,1,QTableWidgetItem(str(self.state[3])))
        self.state_table.setItem(4,1,QTableWidgetItem(str(self.state[4])))
        self.state_table.setItem(5,1,QTableWidgetItem(str(self.state[5])))
        self.state_table.setItem(6,1,QTableWidgetItem(str(self.state[6])))
        self.state_table.setItem(7,1,QTableWidgetItem(str(self.state[7])))
        self.state_table.setItem(8,1,QTableWidgetItem(str(self.state[8])))
        self.state_table.setItem(9,1,QTableWidgetItem(str(self.state[9])))
        self.state_table.setItem(10,1,QTableWidgetItem(str(self.state[10])))
        self.state_table.setItem(11,1,QTableWidgetItem(str(self.state[11])))
        self.state_table.setItem(12,1,QTableWidgetItem(str(self.state[12])))
        self.state_table.setItem(13,1,QTableWidgetItem(str(self.state[13])))
        self.state_table.setItem(14,1,QTableWidgetItem(str(self.state[14])))
        self.state_table.setItem(15,1,QTableWidgetItem(str(self.state[15])))
        self.state_table.setItem(16,1,QTableWidgetItem(str(self.state[16])))
        self.state_table.setItem(17,1,QTableWidgetItem(str(self.state[17])))
        self.state_table.setItem(18,1,QTableWidgetItem(str(self.state[18])))
        self.state_table.setItem(19,1,QTableWidgetItem(str(self.state[19])))
        self.state_table.setItem(20,1,QTableWidgetItem(str(self.state[20])))
        self.state_table.setItem(21,1,QTableWidgetItem(str(self.state[21])))
        self.state_table.setItem(22,1,QTableWidgetItem(str(self.state[22])))
        self.state_table.setItem(23,1,QTableWidgetItem(str(self.state[23])))
        self.state_table.setItem(24,1,QTableWidgetItem(str(self.state[24])))
        self.state_table.setItem(25,1,QTableWidgetItem(str(self.state[25])))
        self.state_table.setItem(26,1,QTableWidgetItem(str(self.state[26])))
        self.state_table.setItem(27,1,QTableWidgetItem(str(self.state[27])))
        self.state_table.setItem(28,1,QTableWidgetItem(str(self.state[28])))
        self.state_table.setItem(29,1,QTableWidgetItem(str(self.state[29])))
        self.state_table.setItem(30,1,QTableWidgetItem(str(self.state[30])))
        self.state_table.setItem(31,1,QTableWidgetItem(str(self.state[31])))
        self.state_table.setItem(32,1,QTableWidgetItem(str(self.state[32])))
        self.state_table.setItem(33,1,QTableWidgetItem(str(self.state[33])))
        self.state_table.setItem(34,1,QTableWidgetItem(str(self.state[34])))
        self.state_table.setItem(35,1,QTableWidgetItem(str(self.state[35])))
        self.state_table.setItem(36,1,QTableWidgetItem(str(self.state[36])))

    def update_state_table(self):
        self.state_table.setItem(0,0,QTableWidgetItem('head q0:'))
        self.state_table.setItem(1,0,QTableWidgetItem('head q1:'))
        self.state_table.setItem(2,0,QTableWidgetItem('head q2:'))
        self.state_table.setItem(3,0,QTableWidgetItem('head q3:'))
        self.state_table.setItem(4,0,QTableWidgetItem('head tx:'))
        self.state_table.setItem(5,0,QTableWidgetItem('head ty:'))
        self.state_table.setItem(6,0,QTableWidgetItem('head tz:'))
        self.state_table.setItem(7,0,QTableWidgetItem('thorax q0:'))
        self.state_table.setItem(8,0,QTableWidgetItem('thorax q1:'))
        self.state_table.setItem(9,0,QTableWidgetItem('thorax q2:'))
        self.state_table.setItem(10,0,QTableWidgetItem('thorax q3:'))
        self.state_table.setItem(11,0,QTableWidgetItem('thorax tx:'))
        self.state_table.setItem(12,0,QTableWidgetItem('thorax ty:'))
        self.state_table.setItem(13,0,QTableWidgetItem('thorax tz:'))
        self.state_table.setItem(14,0,QTableWidgetItem('abdomen q0:'))
        self.state_table.setItem(15,0,QTableWidgetItem('abdomen q1:'))
        self.state_table.setItem(16,0,QTableWidgetItem('abdomen q2:'))
        self.state_table.setItem(17,0,QTableWidgetItem('abdomen q3:'))
        self.state_table.setItem(18,0,QTableWidgetItem('abdomen tx:'))
        self.state_table.setItem(19,0,QTableWidgetItem('abdomen ty:'))
        self.state_table.setItem(20,0,QTableWidgetItem('abdomen tz:'))
        self.state_table.setItem(21,0,QTableWidgetItem('wing L q0:'))
        self.state_table.setItem(22,0,QTableWidgetItem('wing L q1:'))
        self.state_table.setItem(23,0,QTableWidgetItem('wing L q2:'))
        self.state_table.setItem(24,0,QTableWidgetItem('wing L q3:'))
        self.state_table.setItem(25,0,QTableWidgetItem('wing L tx:'))
        self.state_table.setItem(26,0,QTableWidgetItem('wing L ty:'))
        self.state_table.setItem(27,0,QTableWidgetItem('wing L tz:'))
        self.state_table.setItem(28,0,QTableWidgetItem('wing L xi:'))
        self.state_table.setItem(29,0,QTableWidgetItem('wing R q0:'))
        self.state_table.setItem(30,0,QTableWidgetItem('wing R q1:'))
        self.state_table.setItem(31,0,QTableWidgetItem('wing R q2:'))
        self.state_table.setItem(32,0,QTableWidgetItem('wing R q3:'))
        self.state_table.setItem(33,0,QTableWidgetItem('wing R tx:'))
        self.state_table.setItem(34,0,QTableWidgetItem('wing R ty:'))
        self.state_table.setItem(35,0,QTableWidgetItem('wing R tz:'))
        self.state_table.setItem(36,0,QTableWidgetItem('wing R xi:'))
        self.state_table.setItem(0,1,QTableWidgetItem(str(self.state[0])))
        self.state_table.setItem(1,1,QTableWidgetItem(str(self.state[1])))
        self.state_table.setItem(2,1,QTableWidgetItem(str(self.state[2])))
        self.state_table.setItem(3,1,QTableWidgetItem(str(self.state[3])))
        self.state_table.setItem(4,1,QTableWidgetItem(str(self.state[4])))
        self.state_table.setItem(5,1,QTableWidgetItem(str(self.state[5])))
        self.state_table.setItem(6,1,QTableWidgetItem(str(self.state[6])))
        self.state_table.setItem(7,1,QTableWidgetItem(str(self.state[7])))
        self.state_table.setItem(8,1,QTableWidgetItem(str(self.state[8])))
        self.state_table.setItem(9,1,QTableWidgetItem(str(self.state[9])))
        self.state_table.setItem(10,1,QTableWidgetItem(str(self.state[10])))
        self.state_table.setItem(11,1,QTableWidgetItem(str(self.state[11])))
        self.state_table.setItem(12,1,QTableWidgetItem(str(self.state[12])))
        self.state_table.setItem(13,1,QTableWidgetItem(str(self.state[13])))
        self.state_table.setItem(14,1,QTableWidgetItem(str(self.state[14])))
        self.state_table.setItem(15,1,QTableWidgetItem(str(self.state[15])))
        self.state_table.setItem(16,1,QTableWidgetItem(str(self.state[16])))
        self.state_table.setItem(17,1,QTableWidgetItem(str(self.state[17])))
        self.state_table.setItem(18,1,QTableWidgetItem(str(self.state[18])))
        self.state_table.setItem(19,1,QTableWidgetItem(str(self.state[19])))
        self.state_table.setItem(20,1,QTableWidgetItem(str(self.state[20])))
        self.state_table.setItem(21,1,QTableWidgetItem(str(self.state[21])))
        self.state_table.setItem(22,1,QTableWidgetItem(str(self.state[22])))
        self.state_table.setItem(23,1,QTableWidgetItem(str(self.state[23])))
        self.state_table.setItem(24,1,QTableWidgetItem(str(self.state[24])))
        self.state_table.setItem(25,1,QTableWidgetItem(str(self.state[25])))
        self.state_table.setItem(26,1,QTableWidgetItem(str(self.state[26])))
        self.state_table.setItem(27,1,QTableWidgetItem(str(self.state[27])))
        self.state_table.setItem(28,1,QTableWidgetItem(str(self.state[28])))
        self.state_table.setItem(29,1,QTableWidgetItem(str(self.state[29])))
        self.state_table.setItem(30,1,QTableWidgetItem(str(self.state[30])))
        self.state_table.setItem(31,1,QTableWidgetItem(str(self.state[31])))
        self.state_table.setItem(32,1,QTableWidgetItem(str(self.state[32])))
        self.state_table.setItem(33,1,QTableWidgetItem(str(self.state[33])))
        self.state_table.setItem(34,1,QTableWidgetItem(str(self.state[34])))
        self.state_table.setItem(35,1,QTableWidgetItem(str(self.state[35])))
        self.state_table.setItem(36,1,QTableWidgetItem(str(self.state[36])))
        self.state_table.resizeColumnsToContents()

    def set_seq_file(self,seq_file_in,seq_name_in):
        self.seq_name = seq_name_in
        self.seq_file = seq_file_in    

    def set_label_btn(self,btn_in):
        self.add_lbl_btn = btn_in
        self.add_lbl_btn.clicked.connect(self.save_lbl)

    def set_frame(self,frames_in,mov_nr,seq_nr,frame_nr):
        self.mov_nr = mov_nr+1
        self.seq_nr = seq_nr+1
        self.frame_nr = frame_nr+1
        self.frames = np.copy(frames_in)

    def set_batch_index(self,batch_index_in,frame_nr_in):
        self.frame_nr = frame_nr_in+1
        self.batch_index = batch_index_in
        self.frames = self.frame_batch[:,:,:,self.batch_index]
        self.COM = self.COM_batch[:,self.batch_index]
        body_mask_list = []
        wing_mask_list = []
        b_mask_list = []
        w_mask_list = []
        N_batch = 1
        for i in range(N_batch):
            for n in range(self.N_cam):
                b_mask_list.append(np.transpose(self.body_mask_batch[:,:,n,self.batch_index]))
                w_mask_list.append(np.transpose(self.wing_mask_batch[:,:,n,self.batch_index]))
            body_mask_list.append(np.stack(b_mask_list,axis=2)*1.0)
            wing_mask_list.append(np.stack(w_mask_list,axis=2)*1.0)
        self.body_mask = np.stack(body_mask_list,axis=3)
        self.wing_mask = np.stack(wing_mask_list,axis=3)
        #self.body_mask = np.expand_dims(self.body_mask_batch[:,:,:,self.batch_index],axis=3)
        #self.wing_mask = np.expand_dims(self.wing_mask_batch[:,:,:,self.batch_index],axis=3)

    def set_batch(self,com_batch_in,frame_batch_in,body_mask_batch_in,wing_mask_batch_in,mov_nr_in,seq_nr_in,f_start_in,f_end_in,batch_ind):
        self.mov_nr = mov_nr_in
        self.seq_nr = seq_nr_in
        self.COM_batch = com_batch_in
        self.frame_batch = frame_batch_in
        self.body_mask_batch = body_mask_batch_in
        self.wing_mask_batch = wing_mask_batch_in
        self.f_start = f_start_in
        self.f_end = f_end_in
        self.batch_index = batch_ind

    def save_lbl(self):
        scale_now = np.copy(self.scale)
        state_now = np.copy(self.state)
        frames_now = self.frames
        frame_nr_now = self.frame_nr
        seq_nr_now = self.seq_nr
        mov_nr_now = self.mov_nr
        seq_name_now = self.seq_name
        # Open label file:
        os.chdir(self.label_dir)
        try:
            lbl_file = h5py.File(self.label_file,'r+')
            lbl_keys = lbl_file.keys()
            group_key = seq_name_now + '_' + str(mov_nr_now) + '_' + str(seq_nr_now) + '_' + str(frame_nr_now)
            if group_key in lbl_keys:
                grp = lbl_file[group_key]
                frames_data = grp['frames']
                frames_data[...] = frames_now
                state_data = grp['state']
                state_data[...] = state_now
                scale_data = grp['scale']
                scale_data[...] = scale_now
            else:
                grp = lbl_file.create_group(group_key)
                grp.create_dataset('frames',data=frames_now)
                grp.create_dataset('state',data=state_now)
                grp.create_dataset('scale',data=scale_now)
                grp.create_dataset('frame_nr',data=frame_nr_now)
                grp.create_dataset('seq_nr',data=seq_nr_now)
                grp.create_dataset('mov_nr',data=mov_nr_now)
                grp.create_dataset('seq_name',data=seq_name_now)
            lbl_file.close()
            print('saved: ' + group_key)
        except:
            print('error could not open label file')

    def set_masks(self,body_mask_in,wing_mask_in):
        body_mask_list = []
        wing_mask_list = []
        b_mask_list = []
        w_mask_list = []
        N_batch = 1
        for i in range(N_batch):
            for n in range(self.N_cam):
                b_mask_list.append(np.transpose(body_mask_in[n]))
                w_mask_list.append(np.transpose(wing_mask_in[n]))
            body_mask_list.append(np.stack(b_mask_list,axis=2)*1.0)
            wing_mask_list.append(np.stack(w_mask_list,axis=2)*1.0)
        self.body_mask = np.stack(body_mask_list,axis=3)
        self.wing_mask = np.stack(wing_mask_list,axis=3)

    def set_N_particles_spin(self,spin_in):
        self.N_particles_spin = spin_in
        self.N_particles_spin.setMinimum(2)
        self.N_particles_spin.setMaximum(1024)
        self.N_particles_spin.setValue(16)
        self.N_particles = 16
        self.N_particles_spin.valueChanged.connect(self.set_N_particles)

    def set_N_particles(self,N_part_in):
        self.N_particles = N_part_in
        #self.setup_pso(self.batch_size)

    def set_N_iter_spin(self,spin_in):
        self.N_iter_spin = spin_in
        self.N_iter_spin.setMinimum(10)
        self.N_iter_spin.setMaximum(10000)
        self.N_iter_spin.setValue(300)
        self.N_iter = 300
        self.N_iter_spin.valueChanged.connect(self.set_N_iter)

    def set_N_iter(self,N_iter_in):
        self.N_iter = N_iter_in
        #self.setup_pso(self.batch_size)

    def set_cost_spin(self,cost_spin_in):
        self.cost_spin = cost_spin_in
        self.cost_spin.setMinimum(0.00001)
        self.cost_spin.setMaximum(1.0)
        self.cost_spin.setDecimals(5)
        self.cost_spin.setSingleStep(0.00001)
        self.cost_spin.setValue(0.00001)
        self.cost_thresh = 0.00001
        self.cost_spin.valueChanged.connect(self.set_cost_thresh)

    def set_cost_thresh(self,cost_in):
        self.cost_thresh = cost_in
        #self.setup_pso(self.batch_size)

    def set_pso_search_btn(self,btn_in):
        self.setup_pso(1)
        self.pso_search_btn = btn_in
        self.pso_search_btn.clicked.connect(self.pso_fit)

    def setup_pso(self,batch_size_in):
        #self.opt.set_PSO_param(0.3,0.3,0.3,0.15,37,self.N_particles,self.N_iter)
        self.opt.set_PSO_param(0.5,0.3,0.3,0.15,37,self.N_particles,self.N_iter)
        model_file = 'model_parameters.json'
        self.opt.load_model_json(self.mdl_dir,model_file)
        wndw_size = np.asarray(self.window_size,dtype=np.float64)
        self.opt.set_calibration(self.c_type,self.c_params,wndw_size,self.N_cam)
        scale_array = np.asarray(self.scale)
        self.opt.set_scale(scale_array,5)
        self.opt.set_batch_size(batch_size_in)
        self.opt.setup_threads(self.mdl_dir)

    def pso_fit(self):
        scale_array = np.asarray(self.scale)
        self.opt.set_scale(scale_array,5)
        #self.opt.set_batch_size(1)
        self.setup_pso(1)
        com_arr = np.zeros((4,1))
        com_arr[:,0] = self.COM
        self.load_binary_images(self.body_mask,self.wing_mask,1,1)
        self.opt.set_COM(com_arr)
        # setup inital positions particles:
        init_start = np.zeros((self.N_state,1))
        try:
            state_pred = self.net.predict_single_frame(self.frames)
            init_start[:,0] = np.squeeze(state_pred)
        except:
            init_start[:,0] = np.squeeze(self.state)
        self.opt.set_particle_start(init_start)
        opt_state = self.opt.PSO_fit()
        print(np.squeeze(opt_state[:,0]))
        self.set_model_state(np.squeeze(opt_state[:37,0]))
        self.save_to_buffer()
        mdl_img = self.opt.return_model_img()
        #print(mdl_img)
        #img_list = []
        #for n in range(self.N_cam):
        #    img_list.append(mdl_img[:,:,n])
        #img_stack = np.uint8(np.hstack(img_list))
        #os.chdir('/home/flythreads/Documents/FlyNet4/trash')
        #cv2.imwrite('mdl_img.jpg',img_stack)
        #img_list = []
        #for n in range(self.N_cam):
        #    img_list.append(self.wing_mask[:,:,n])
        #img_stack = np.uint8(np.hstack(img_list))
        #cv2.imwrite('wing_img.jpg',img_stack) 
        #self.set_model_state(np.squeeze(opt_state[:37,0]))

    def add_graphs(self):
        self.graph_list = []
        for n in range(self.N_cam):
            self.graph_list.append(Graph(i))
            self.v_list[i].addItem(self.graph_list[i])
            self.wing_txt = ['L_r','L_t','R_r','R_t','B_c','B_h','B_JL','B_JR','B_a']
            self.wing_sym = ["o","o","o","o","o","o","o","o","o"]
            self.wing_clr = ['r','r','b','b','g','g','g','g','g']
            uv_pos = np.concatenate((np.transpose(self.wing_L_uv[i]),np.transpose(self.wing_R_uv[i]),np.transpose(self.body_uv[i])),axis=0)
            print(uv_pos)
            self.graph_list[i].setData(pos=uv_pos, size=2, symbol=self.wing_sym, pxMode=False, text=self.wing_txt, textcolor=self.wing_clr)
        self.add_wing_contours()

    def add_wing_contours(self):
        self.remove_wing_contours()
        # Update state
        self.state_calc.set_state(self.state_L,self.state_R)
        # Retrieve 3d coordinates left and right wings:
        wing_L_cnts = self.state_calc.wing_contour_L()
        wing_R_cnts = self.state_calc.wing_contour_R()
        # obtain 2D projections:
        cnts_L_uv = []
        for cnt in wing_L_cnts:
            cnts_L_uv.append(self.contours2uv(cnt))
        cnts_R_uv = []
        for cnt in wing_R_cnts:
            cnts_R_uv.append(self.contours2uv(cnt))
        # Add contour plots to the image items:
        self.contours_L = []
        for i,cnt_pts in enumerate(cnts_L_uv):
            for n in range(self.N_cam):
                curve_pts = np.transpose(cnt_pts[n][0:2,:])
                curve = pg.PlotCurveItem(x=curve_pts[:,0],y=curve_pts[:,1],pen=[255,0,0])
                self.contours_L.append(curve)
                self.v_list[n].addItem(self.contours_L[i*self.N_cam+n])
        self.contours_R = []
        for i,cnt_pts in enumerate(cnts_R_uv):
            for n in range(self.N_cam):
                curve_pts = np.transpose(cnt_pts[n][0:2,:])
                curve = pg.PlotCurveItem(x=curve_pts[:,0],y=curve_pts[:,1],pen=[0,0,255])
                self.contours_R.append(curve)
                self.v_list[n].addItem(self.contours_R[i*self.N_cam+n])
        self.contours_B = []
        for i in range(len(self.body_contour_outline)):
            for n in range(self.N_cam):
                curve_pts = np.transpose(self.body_uv[n][0:2,self.body_contour_outline[i]])
                curve = pg.PlotCurveItem(x=curve_pts[:,0],y=curve_pts[:,1],pen=[0,255,0])
                self.contours_B.append(curve)
                self.v_list[n].addItem(self.contours_B[i*self.N_cam+n])

    def remove_wing_contours(self):
        N_L = len(self.contours_L)
        if N_L>0:
            for i in range(N_L):
                self.v_list[i%self.N_cam].removeItem(self.contours_L[i])
            self.contours_L = []
        N_R = len(self.contours_R)
        if N_R>0:
            for i in range(N_R):
                self.v_list[i%self.N_cam].removeItem(self.contours_R[i])
            self.contours_R = []
        N_B = len(self.contours_B)
        if N_B>0:
            for i in range(N_B):
                self.v_list[i%self.N_cam].removeItem(self.contours_B[i])
            self.contours_B = []

    def set_prediction_network(self,network_in):
        self.net = network_in

    def set_predictor_button(self,btn_in):
        self.predict_btn = btn_in
        self.predict_btn.clicked.connect(self.predict_frame)

    def predict_frame(self):
        state_pred = self.net.predict_single_frame(self.frames)
        self.set_model_state(state_pred)

    def save_to_buffer(self):
        np.roll(self.state_buffer,1,axis=1)
        self.state_buffer[:,0] = self.state

    def set_batch_size(self,batch_size_in):
        self.batch_size = batch_size_in

    def set_analyze_btn(self,btn_in):
        self.analyze_btn = btn_in
        self.analyze_btn.clicked.connect(self.analyze_sequences)

    def set_seq_list(self,seq_keys_list_in):
        self.seq_keys_list = seq_keys_list_in

    def set_batch_loader(self,batch_loader_in):
        self.batch_loader = batch_loader_in

    def analyze_sequences(self):
        key_list = list(self.seq_file.keys())
        #self.setup_pso()
        find_batch_fit = any('batch_fit_' in x for x in key_list)
        if find_batch_fit:
            print('Fitting has already been performed')
            self.load_tracked_btn.setEnabled(True)
            self.progress_bar.setValue(100)
        else:
            print('batch_size: '+str(self.batch_size))
            self.setup_pso(self.batch_size)
            # Get number of movies
            N_mov = len(self.seq_keys_list)
            print('Number of movies: '+str(N_mov))
            # Get total number of frames:
            N_frames_total = 0
            N_seqs_total = 0
            for i in range(N_mov):
                N_seq = len(self.seq_keys_list[i])
                N_seqs_total += N_seq
                for j in range(N_seq):
                    N_frames_total += len(self.seq_keys_list[i][j])
            print('Total number of sequences: '+str(N_seqs_total))
            print('Total number of frames: '+str(N_frames_total))
            self.progress_bar.setValue(0)
            # Analyze sequences:
            print('Starting analysis:')
            f_cntr = 0
            for i in range(N_mov):
                N_seq = len(self.seq_keys_list[i])
                for j in range(N_seq):
                    N_f = len(self.seq_keys_list[i][j])
                    self.seq_keys_list[i][j].sort()
                    batch_nr = 0
                    for k in range(N_f):
                        # Check if center frame:
                        if k%self.batch_size==0:
                            f_start = self.seq_keys_list[i][j][k]
                            if f_start<N_f:
                                batch_nr+=1
                                f_end = self.seq_keys_list[i][j][k]+self.batch_size
                                if f_end>=N_f:
                                    f_end = N_f
                                n_batch = f_end-f_start
                                f_cntr += n_batch
                                img_batch,com_batch,body_mask_btch,wing_mask_btch = self.batch_loader(i,j,f_start,f_end)
                                print('Analyzing batch: '+str(batch_nr)+', mov: '+str(i+1)+', seq: '+str(j+1)+', frame start: '+str(f_start)+', frame end: '+str(f_end))
                                self.pso_fit_batch(img_batch,com_batch,body_mask_btch,wing_mask_btch,i,j,f_start,f_end,batch_nr)
                            progress_percentage = ((f_cntr*1.0)/(N_frames_total*1.0-1.0))*100.0
                            self.progress_bar.setValue(progress_percentage)
            # ae addition
            self.progress_bar.setValue(100)
            self.load_tracked_btn.setEnabled(True)

    def pso_fit_batch(self,img_batch,com_batch,body_mask_btch,wing_mask_btch,mov_nr_in,seq_nr_in,f_start,f_end,b_ind):
        t_start = time.time()
        n_batch = f_end-f_start
        # make batch prediction:
        state_batch = np.transpose(np.matlib.repmat(np.squeeze(self.state),n_batch,1))
        init_start = self.net.predict_batch(img_batch,state_batch)
        self.load_binary_images(body_mask_btch,wing_mask_btch,n_batch,0)
        self.opt.set_COM(com_batch)
        self.opt.set_particle_start(init_start)
        opt_state = self.opt.PSO_fit()
        # Save frame to file:
        #try:
        key_list = list(self.seq_file.keys())
        group_key = 'batch_fit_m_'+str(mov_nr_in+1)+'_s_'+str(seq_nr_in+1)+'_b_'+str(b_ind)
        if group_key in key_list:
            grp = self.seq_file[group_key]
            temp_data = grp['opt_state']
            temp_data[...] = opt_state
            temp_data = grp['pred_state']
            temp_data[...] = init_start
            #temp_data = grp['img_batch']
            #temp_data[...] = img_batch
            temp_data = grp['com_batch']
            temp_data[...] = com_batch
            temp_data = grp['mov_nr']
            temp_data[...] = mov_nr_in
            temp_data = grp['seq_nr']
            temp_data[...] = seq_nr_in
            temp_data = grp['batch_nr']
            temp_data[...] = b_ind
            temp_data = grp['f_start']
            temp_data[...] = f_start
            temp_data = grp['f_end']
            temp_data[...] = f_end
        else:
            grp = self.seq_file.create_group(group_key)
            grp.create_dataset('opt_state',data=opt_state)
            grp.create_dataset('pred_state',data=init_start)
            #grp.create_dataset('img_batch',data=img_batch,compression='lzf')
            grp.create_dataset('com_batch',data=com_batch)
            grp.create_dataset('mov_nr',data=mov_nr_in)
            grp.create_dataset('seq_nr',data=seq_nr_in)
            grp.create_dataset('batch_nr',data=b_ind)
            grp.create_dataset('f_start',data=f_start)
            grp.create_dataset('f_end',data=f_end)
        #except:
        #    print('error: could not save '+group_key)
        #print(opt_state[40:,:])
        print('elapsed time: '+str(time.time()-t_start))

    def load_binary_images(self,body_mask_btch,wing_mask_btch,n_batch,transpose_on):
        self.opt.set_batch_size(n_batch)
        for i in range(n_batch):
            for n in range(self.N_cam):
                if transpose_on ==0:
                    body_img = body_mask_btch[:,:,n,i]
                    wing_img = wing_mask_btch[:,:,n,i]
                else:
                    body_img = np.transpose(body_mask_btch[:,:,n,i])
                    wing_img = np.transpose(wing_mask_btch[:,:,n,i])
                body_indx = np.argwhere(body_img)*1.0
                wing_indx = np.argwhere(wing_img)*1.0
                self.opt.set_bin_imgs(body_indx,wing_indx,i,n,body_indx.shape[0],wing_indx.shape[0])

    def fit_outlier(self,mov_nr_in,seq_nr_in,frame_nr_in,setup_fit):
        # Assumed that current frame is set to outlier frame
        if setup_fit:
            # setup pso:
            self.opt.set_PSO_param(0.5,0.3,0.3,0.15,37,32,500) # large nr of particles, many iterations
            model_file = 'model_parameters.json'
            self.opt.load_model_json(self.mdl_dir,model_file)
            wndw_size = np.asarray(self.window_size,dtype=np.float64)
            self.opt.set_calibration(self.c_type,self.c_params,wndw_size,self.N_cam)
            scale_array = np.asarray(self.scale)
            self.opt.set_scale(scale_array,5)
            self.opt.set_batch_size(1)
            self.opt.setup_threads(self.mdl_dir)
        f_start = frame_nr_in-5
        f_end     = frame_nr_in+5
        N_f = len(self.seq_keys_list[mov_nr_in][seq_nr_in])
        if f_start <0:
            f_start = 1
            f_end     = 12
        elif f_end>N_f:
            f_start = N_f-11
            f_end     = N_f
        img_batch,com_batch,body_mask_btch,wing_mask_btch = self.batch_loader(mov_nr_in,seq_nr_in,f_start,f_end)
        com_now = np.zeros((4,1))
        com_now[:,0] = com_batch[:,5]
        self.body_mask = np.expand_dims(body_mask_btch[:,:,:,5],axis=3)
        self.wing_mask = np.expand_dims(wing_mask_btch[:,:,:,5],axis=3)
        self.load_binary_images(self.body_mask,self.wing_mask,1,1)
        self.opt.set_COM(com_now)
        init_start = np.zeros((self.N_state,1))
        state_pred = self.net.predict_single_frame(img_batch[:,:,:,5])
        init_start[:,0] = np.squeeze(state_pred)
        self.opt.set_particle_start(init_start)
        opt_state = self.opt.PSO_fit()
        return opt_state

    def set_progress_bar(self,bar_in):
        self.progress_bar = bar_in
        self.progress_bar.setValue(0)

    def set_load_tracked_data_btn(self,btn_in):
        self.load_tracked_btn = btn_in
