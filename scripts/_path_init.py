""" Add visualDet3D to PYTHONPATH
"""
import sys
import os

visualDet3D_path = os.path.dirname(sys.path[0])  #two folders upwards
sys.path.insert(0, visualDet3D_path)