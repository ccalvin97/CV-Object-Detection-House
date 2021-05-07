#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from itertools import groupby
from skimage import morphology,measure
from PIL import Image
from scipy import misc
import pdb
import re
import pandas as pd
import cv2
import matplotlib as plt
from matplotlib import pyplot as plt
import argparse



# 从label图得???boundingbox 和图上连通域数量 object_num
def getboundingbox(image, rgbmask):
    # mask.shape = [image.shape[0], image.shape[1], classnum]
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[np.where(np.all(image == rgbmask[1],axis=-1))[:2]] = 1
    # 删掉小于20像素的目???    
    mask_without_small = morphology.remove_small_objects(mask, min_size=20, connectivity=2)
    # 连通域标记
    label_image = measure.label(mask_without_small)
    #统计object个数
    object_num = len(measure.regionprops(label_image))
    boundingbox = list()
    for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
        boundingbox.append(region.bbox)
    return object_num, boundingbox   ##### boundingbox =[x,y,z,q] --> y, x, y', x'

def convert_labels(path, x1, y1, x2, y2):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
    """
    def sorting(l1, l2):
        if l1 > l2:
            lmax, lmin = l1, l2
            return lmax, lmin
        else:
            lmax, lmin = l2, l1
            return lmax, lmin
    size = get_img_shape(path)
    xmax, xmin = sorting(x1, x2)
    ymax, ymin = sorting(y1, y2)
    dw = 1./size[1]
    dh = 1./size[0]
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def get_img_shape(path):
    img = cv2.imread(path)
    try:
        return img.shape
    except AttributeError:
        print('error! ', path)
        return (None, None, None)


        
def main():
    # Get arguments
    parser = argparse.ArgumentParser(description='welcome to the segmentation 2 object detection programme')
    parser.add_argument('-p_in', '--p_in', type=str,
        help='path of input image dir - Semantic Segmentation Label data, PS. label should only have [0,0,0],[1,1,1] colour  & size should be 1024*1024, ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel')
    parser.add_argument('-p_out', '--p_out', type=str,
        help='data out abs address for picture ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')   
    args = parser.parse_args()

    # 因为一张图片里只有一种类别的目标，所以label图标记只有黑白两???    
    rgbmask = np.array([[0,0,0],[1,1,1]],dtype=np.uint8)    
    if os.path.exists(args.p_out) is False:
        os.mkdir(args.p_out)

    for root, _, fnames in sorted(os.walk(args.p_in)):
        for fname in sorted(fnames):
            labelpath = os.path.join(root, fname)
            labelimage = misc.imread(labelpath)
            objectnum, bbox = getboundingbox(labelimage, rgbmask)
            labelfilename = labelpath.split('/')[-1]
            ImageID = labelfilename.split('.')[0].split('.')[0]
            bbox = pd.DataFrame(bbox, columns=['y1', 'x1', 'y2', 'x2'])
            if len(bbox) != 0:
                bbox['x'], bbox['y'], bbox['width'], bbox['height'] = zip(*bbox.apply(lambda row: convert_labels(labelpath, row['x1'], row['y1'], row['x2'], row['y2']), axis=1))
                bbox['0']=bbox['x'].apply(lambda x: '0' if x else '0' )
                bbox = bbox[['0', 'x', 'y', 'width', 'height']]
            bbox.to_csv(args.p_out+'/'+ImageID+'.txt', index=False, header=False, sep =' ')

if __name__ == '__main__':
    main()
    
'''
Plot and Check Annotation Function

def from_yolo_to_cor(box):
    img_h, img_w, _ = [1024,1024,3]
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    box = pd.read_csv(box,sep=" ", header=None)
    x1, y1 = (box[1] + box[3]/2)*img_w, (box[2] + box[4]/2)*img_h
    x2, y2 = (box[1] - box[3]/2)*img_w, (box[2] - box[4]/2)*img_h
    return x1, y1, x2, y2
    
def draw_boxes(img, boxes):
    img = cv2.imread(img)
    for box in boxes:
        x1, y1, x2, y2 = from_yolo_to_cor(box)
        for j in range(len(x1)) :
            cv2.rectangle(img, (int(x1[j]), int(y1[j])), (int(x2[j]), int(y2[j])), (0,255,0), 3)
    plt.imshow(img)

draw_boxes('C:\\Users\\heca0002\\Desktop\\333\\10_0__.tif', ['C:\\Users\\heca0002\\Desktop\\result\\10_0__.txt'])
'''
