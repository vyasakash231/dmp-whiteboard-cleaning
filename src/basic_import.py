#! /usr/bin/python3
import os
import time
import sys
from abc import ABC, abstractmethod
from math import *
import threading
import copy
import json
import glob
import queue
import argparse
from threading import Lock, Thread
import cv2 as cv
from cv2 import aruco
import pyrealsense2 as rs
import numpy as np
import numpy.linalg as LA
import quaternion
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
np.set_printoptions(suppress=True)

import matplotlib
matplotlib.use('TkAgg')  # Must be before importing pyplot
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import rospy
import tf, tf2_ros
from tf.transformations import *
from tf2_msgs.msg import TFMessage

sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../common/imp")))

import DR_init
DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "a0509"

from DSR_ROBOT import *
from DR_common import *

from dsr_msgs.msg import *
from dsr_msgs.srv import *
from geometry_msgs.msg import Pose, PoseStamped, Twist, Quaternion, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

import scipy
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.linalg import expm

