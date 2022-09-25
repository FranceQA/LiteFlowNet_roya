import argparse
import glob
from turtle import color
import numpy as np
from flowiz import read_flow, convert_from_flow
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2 as cv
from numpy.linalg import inv
import cupy

def visualizer(flo_file,segm_file):
  #Variables
  TAG_FLOAT = 202021.25
  THRESHOLD = 0.1
  START_POINT = np.array([0,0])
  TOLERANCE = 0
  SPORE_MASK = 2
  current_point = START_POINT
  line_pairs = []

  #Features
  flow = read_flow(flo_file)
  seg_im = Image.open(segm_file)

  #CROP
  #seg_im = np.array(seg_im)
  #seg_im = seg_im[0:256,0:256,:]
  #flow = flow[0:256,0:256,:]
  #RESIZE
  seg_im = np.array(seg_im)
  seg_im = cv.resize(seg_im, (100,100))
  flow = cv.resize(flow, (100,100))

  #Ref image
  seg_im = np.array(seg_im)
  plt.figure()
  plt.imshow(seg_im)
  plt.show()

  spore_mask = seg_im == SPORE_MASK
  color_flow = convert_from_flow(flow)
  vector_x, vector_y = flow[round(current_point[0]), round(current_point[1])]
  
  print("Flo:\t", vector_x, vector_y)

  next_point_y = current_point[0] + vector_y
  next_point_x = current_point[1] + vector_x
  next_point = np.array([next_point_y, next_point_x])

  if np.sum(np.sqrt(np.power(next_point - current_point, 2))) > TOLERANCE:
      line_pairs.append([current_point, next_point])

      print(
          current_point,
          next_point,
          flow[round(current_point[0]), round(current_point[1])]
      )

      current_point = next_point

  for first_point, second_point in line_pairs:
      first_y, first_x = first_point
      second_y, second_x = second_point
      first_y, first_x = int(first_y), int(first_x)
      second_y, second_x = int(second_y), int(second_x)
      plt.plot([first_x, second_x], [first_y, second_y])
  plt.imshow(color_flow)

  plt.show()

def sub_visualizer(flow,seg_im):
  #Variables
  TAG_FLOAT = 202021.25
  THRESHOLD = 0.1
  START_POINT = np.array([0,0])
  TOLERANCE = 0
  SPORE_MASK = 2
  current_point = START_POINT
  line_pairs = []

  #Ref image
  seg_im = np.array(seg_im)
  plt.figure()
  plt.imshow(seg_im)
  plt.show()

  spore_mask = seg_im == SPORE_MASK
  color_flow = convert_from_flow(flow)
  vector_x, vector_y = flow[round(current_point[0]), round(current_point[1])]
  
  print("Flo:\t", vector_x, vector_y)

  next_point_y = current_point[0] + vector_y
  next_point_x = current_point[1] + vector_x
  next_point = np.array([next_point_y, next_point_x])

  if np.sum(np.sqrt(np.power(next_point - current_point, 2))) > TOLERANCE:
      line_pairs.append([current_point, next_point])

      print(
          current_point,
          next_point,
          flow[round(current_point[0]), round(current_point[1])]
      )

      current_point = next_point

  for first_point, second_point in line_pairs:
      first_y, first_x = first_point
      second_y, second_x = second_point
      first_y, first_x = int(first_y), int(first_x)
      second_y, second_x = int(second_y), int(second_x)
      plt.plot([first_x, second_x], [first_y, second_y])
  plt.imshow(color_flow)

  plt.show()

def newflo(flow):
  flowa,flowb = cupy.dsplit(flow, 2)
  dimensions = flow.shape[0]
  pts = cupy.linspace(0,dimensions-1, dimensions)
  xv, yv = cupy.meshgrid(pts, pts)
  X = cupy.concatenate((xv.reshape(-1,1),yv.reshape(-1,1)),axis=1)  #Crea el par de coordenadas que utilizaremos
  fa = flowa.reshape(-1,1)                                        #Salida en cada par de coordanadas
  fb = flowb.reshape(-1,1)  
  xi = cupy.concatenate((cupy.ones((dimensions**2,1)),X),axis=1)
  ya = yb = []
  tau=0.9
  for i in range(dimensions**2):
    w = cupy.exp(-cupy.sum((xi-xi[i])**2,axis=1))/(2*tau**2)
    aux = cupy.dot(xi.transpose(),cupy.diag(w))
    thetaa = cupy.dot(cupy.linalg.inv(cupy.dot(aux,xi)),cupy.dot(aux,fa))
    thetab = cupy.dot(cupy.linalg.inv(cupy.dot(aux,xi)),cupy.dot(aux,fb))
    ya = ya + [cupy.dot(xi[i],thetaa)]
    yb = yb + [cupy.dot(xi[i],thetab)]
  return ya,yb

path = '/content/'
flo_files = glob.glob(path + "/*.flo")
segm_files = glob.glob(path + "/*.png")

flow = read_flow(flo_files[0])
flow = cv.resize(flow, (100,100))
flowa,flowb = np.dsplit(flow, 2)
flowr = cupy.array(flow)
ya,yb = newflo(flowr)

k=[]
for i in ya:
  k.append(cupy.ndarray.get(i))

l=[]
for j in yb:
  l.append(cupy.ndarray.get(j))



flowf = np.concatenate((k,l),axis=1)
flowf = flowf.reshape(100,100,2)

seg_im = Image.open(segm_files[0])
seg_im = np.array(seg_im)
seg_im = cv.resize(seg_im, (100,100))

visualizer(flo_files[0],segm_files[0])
sub_visualizer(flowf,seg_im)
