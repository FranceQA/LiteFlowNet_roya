from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
import math
from flowiz import read_flow, convert_from_flow


def new_flow(flo,flo_limit):
  xs =[]
  ys =[]
  za =[]
  zb =[]

  cx=0
  for i in range(len(flo)):
    cy=0
    for j in range(len(flo[i])):
      [x,y] = flo[i][j]
      if (x>flo_limit) or (y>flo_limit):          #Rango mínimo de flujo en cualquier dirección / hipotenusa puede ser alternativa
        xs.append(cx)             #Coordenadas
        ys.append(cy)
        za.append(x)
        zb.append(y)
      cy = cy+1
    cx = cx+1

  Y = np.linspace(0, 719, 720)
  X = np.linspace(0, 1279, 1280)
  X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
  interp = CloughTocher2DInterpolator(list(zip(ys, xs)), za)
  Za = interp(X, Y)
  interp = CloughTocher2DInterpolator(list(zip(ys, xs)), zb)
  Zb = interp(X, Y)

  aux = np.stack((Za,Zb), axis=-1)

  for i in range(len(aux)):
    for j in range(len(aux[i])):
      for k in range(len(aux[i][j])):
        if math.isnan(aux[i][j][k]):
          aux[i][j][k] = 0.0

  return aux

def load_data(path):
    flo_paths = glob(os.path.join(path, '*.flo'))
    img0_paths = [x.replace('flow.flo', 'img1.tif') for x in flo_paths]
    img1_paths = [x.replace('flow.flo', 'img2.tif') for x in flo_paths]
    return flo_paths, img0_paths, img1_paths

def load_data_roya(path):
  videos = glob(os.path.join(path,'group_1/*'))

  flo, img0, img1 = [], [], []
  for video in videos:
    img_paths = glob(os.path.join(video,'*.png')); img_paths.sort()
    img0_paths = img_paths[0:-1]
    img1_paths = img_paths[1:]
    flo_paths = [x.replace('.png', '.flo') for x in img0_paths]
    flo_paths = [x.replace('group_1', 'flow') for x in flo_paths]

    flo = flo + flo_paths
    img0 = img0 + img0_paths
    img1 = img1 + img0_paths

  return flo, img0, img1

class MyDataset(Dataset):
    def __init__(self, path,  transform=None):
        self.flo_paths, self.img0_paths, self.img1_paths = load_data(path)
        self.transform = transform

    def __getitem__(self, i):
        img1 = cv.imread(self.img0_paths[i])
        img2 = cv.imread(self.img1_paths[i])
        flo = readFlowFile(self.flo_paths[i])
        if self.transform is not None:
            img1,img2,flo = self.transform(img1,img2,flo)
        return img1, img2, flo

    def __len__(self):
        return len(self.img0_paths)

class MyDataset_roya(Dataset):
    def __init__(self, path,  transform=None):
        self.flo_paths, self.img0_paths, self.img1_paths = load_data_roya(path)
        self.transform = transform

    def __getitem__(self, i):
        img1 = cv.imread(self.img0_paths[i])
        img2 = cv.imread(self.img1_paths[i])
        flo = read_flow(self.flo_paths[i])
        img1 = cv.resize(img1,(256,256))
        img2 = cv.resize(img2,(256,256))
        flo = cv.resize(flo,(256,256))
        if self.transform is not None:
            img1,img2,flo = self.transform(img1,img2,flo)
        return img1, img2, flo

    def __len__(self):
        return len(self.img0_paths)
