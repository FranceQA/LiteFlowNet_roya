from torchsummary import summary
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from glob import glob
import os
from lite_flownet import liteflownet 
import torch
from torch.autograd import Variable
from utils.multiscaleloss import realEPE,RMSE
from tqdm import tqdm
from utils.dataloader import MyDataset,MyDataset_roya
from utils.augmentations import Basetransform
from utils.flowlib import *
import torch.nn.functional as F

__all__ = [
    'eval'
]

#======================================================================
#Visualizador

def write_flo(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    flow = flow[0, :, :, :]
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()

#======================================================================

def find_NewFile(path):
    # 获取文件夹中的所有文�?
    lists = glob(os.path.join(path, '*.tar'))
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(x))
    # 把目录和文件名合成一个路�?
    file_new = lists[-1]
    return file_new




def eval(model,ImgLoader):
    total_test_rmse = []
    iterator = iter(ImgLoader)
    step = len(ImgLoader)
    # step = 50
    model.eval()
    print('evaluating... ')
    for i in tqdm(range(step)):
        img1, img2, flo = next(iterator)
        img1 = Variable(torch.FloatTensor(img1.float()))
        img2 = Variable(torch.FloatTensor(img2.float()))
        flo = Variable(torch.FloatTensor(flo.float()))
        imgL, imgR, flowl0 = img1.cuda(), img2.cuda(), flo.cuda()
        output = model((imgL, imgR))
        total_test_rmse += [RMSE(output.detach(), flowl0.detach()).cpu().numpy()]
        #print(f'Salida {output[0].cpu().detach().numpy().shape} y referencia {flowl0[0].cpu().detach().numpy().shape}')
        #print(img1.cpu().detach().numpy().shape)
        if i == 20:
            flo1 = output[0].cpu().detach().numpy().reshape(256,256,2)
            flo2 = flowl0[0].cpu().detach().numpy().reshape(256,256,2)
            img = img1[0].cpu().detach().numpy().reshape(256,256,3)
            #/home/fquesada/Documents/pruebas
            write_flow(flo1,'/home/fquesada/Documents/pruebas/flo_out.flo')
            write_flow(flo2,'/home/fquesada/Documents/pruebas/flo_ref.flo')
            cv.imwrite('/home/fquesada/Documents/pruebas/imgp.png',img)
        #    a = output.detach().cpu()
        #    b = flowl0.detach().cpu()
        #    print(np.array(b))
        #    print(np.array(a).shape)
        #    write_flow(np.array(a),'/home/fquesada/output.flo')
        #    write_flow(np.array(b),'/home/fquesada/routput.flo')
    return np.mean(total_test_rmse)


def eval_with_epe(model,ImgLoader):
    total_test_rmse = []
    total_test_epe = []
    iterator = iter(ImgLoader)
    step = len(ImgLoader)
    # step = 50
    model.eval()
    print('evaluating... ')
    for i in tqdm(range(step)):
        img1, img2, flo = next(iterator)
        img1 = Variable(torch.FloatTensor(img1.float()))
        img2 = Variable(torch.FloatTensor(img2.float()))
        flo = Variable(torch.FloatTensor(flo.float()))
        imgL, imgR, flowl0 = img1.cuda(), img2.cuda(), flo.cuda()
        output = model((imgL, imgR))
        total_test_rmse += [RMSE(output.detach(), flowl0.detach()).cpu().numpy()]
        total_test_epe += [realEPE(output.detach(), flowl0.detach()).cpu().numpy()]
    return np.mean(total_test_rmse), np.mean(total_test_epe)

