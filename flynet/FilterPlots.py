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

class FilterPlots(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None):
        #pg.setConfigOption('background', 'w')
        #pg.GraphicsWindow.__init__(self)
        pg.GraphicsLayoutWidget.__init__(self)
        self.setParent(parent)
        self.labels = ['y-axis','x-axis']
        self.curves = []
        self.N_curves = 8
        self.curves_legends = ['1','2','3','4','5','6','7','8']
        self.curves_pens = [(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255),(255,255,255)]
        self.N_wb_lines = 10
        self.init_plots()

    def init_plots(self):
        # curves:
        self.p1 = self.addPlot(labels={'left':self.labels[0], 'bottom':self.labels[1]})
        self.p1.addLegend()
        for i,curve in enumerate(self.curves):
            self.curves.append([np.linspace(0.0,1.0,num=10,endpoint=True),np.zeros(10)])
            curve_i = self.p1.plot(curve[0],curve[1],pen=self.curves_pens[i],name=self.curves_legends[i])
            self.curves.append(curve_i)
        # time line:
        self.inf_line = pg.InfiniteLine(angle=90, movable=True,pen=(0,3))
        self.p1.addItem(self.inf_line)
        # wb lines:
        self.wb_lines = []
        for i in range(self.N_wb_lines):
            wb_line = pg.InfiniteLine(angle=90, movable=True,pen=(255,255,0))
            wb_line.setValue(-1)
            self.wb_lines.append(wb_line)
            self.p1.addItem(self.wb_lines[i])

    def remove_plots(self):
        for curve in self.curves:
            self.p1.removeItem(curve)
        self.curves = []

    def set_curves(self,labels_in,curves_in,curves_pen_in,curves_legends_in):
        self.remove_plots()
        self.labels = labels_in
        self.p1.setLabel(axis='left',text=self.labels[0])
        self.p1.setLabel(axis='bottom',text=self.labels[1])
        self.curves_pens = curves_pen_in
        self.curves_legends = curves_legends_in
        for i,curve in enumerate(curves_in):
            curve_i = self.p1.plot(curve[0],curve[1],pen=self.curves_pens[i],name=self.curves_legends[i])
            self.curves.append(curve_i)

    def remove_inf_lines(self):
        for line in self.wb_lines:
            self.p1.removeItem(line)
        self.wb_lines = []

    def set_wb_lines(self,t_locs):
        self.remove_inf_lines()
        N_locs = t_locs.shape[0]
        for i in range(N_locs+1):
            line_i = pg.InfiniteLine(angle=90, movable=True,pen=(255,255,0))
            if i < N_locs:
                line_i.setValue(t_locs[i,0])
            elif i == N_locs:
                line_i.setValue(t_locs[i-1,1])
            self.wb_lines.append(line_i)
            self.p1.addItem(self.wb_lines[i])

    def set_time_line(self,t_pos):
        self.inf_line.setValue(t_pos)

    def set_x_range(self,range_in):
        self.p1.setXRange(range_in[0],range_in[1],padding=0)

    def set_y_range(self,range_in):
        self.p1.setYRange(range_in[0],range_in[1],padding=0)

    '''
    def set_title(self,title_in):
        self.setWindowTitle(title_in)

    def set_labels(self,x_label_in,y_label_in):
        self.x_label = x_label_in
        self.y_label = y_label_in



    def setup_plot(self,x_label,y_label):
        self.p1 = self.addPlot(labels =  {'left':self.y_label, 'bottom':self.x_label})
        self.p1.addLegend()
        self.i1_curve = self.p1.plot(np.linspace(0.0,1.0,num=10,endpoint=True),np.zeros(10),pen=(255,0,0),name='i1')
        self.i2_curve = self.p1.plot(np.linspace(0.0,1.0,num=10,endpoint=True),np.zeros(10),pen=(0,0,255),name='i2')
        # time line
        self.inf_line = pg.InfiniteLine(angle=90, movable=True,pen=(0,3))
        self.p1.addItem(self.inf_line)

    def set_data(self,t_vec,i1_data,i2_data):
        self.t_vec = t_vec
        self.i1_data = i1_data
        self.i2_data = i2_data
        self.i1_curve.setData(self.t_vec,self.i1_data)
        self.i2_curve.setData(self.t_vec,self.i2_data)

    def update_time_line(self,t_pos):
        self.inf_line.setValue(t_pos)

    def set_trigger_lines(self,t_trigger):
        N_trigger = t_trigger.shape[0]
        for i in range(N_trigger):
            trig_line = pg.InfiniteLine(angle=90, movable=False)
            trig_line.setValue(t_trigger[i])
            self.p1.addItem(trig_line)
    '''
