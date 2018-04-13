# -*- coding: utf-8 -*-

import cv2 as cv
import os


def img_resize(filepath):
    for parent, dirnames, filenames in os.walk(filepath):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for filename in filenames:  # 输出文件信息
            img_path = os.path.join(parent, filename)  # 输出文件路径信息
            filePath = img_path
            print(img_path)
            img=cv.imread(img_path) # 读取图片
            res=cv.resize(img,(32,32),interpolation=cv.INTER_CUBIC ) #改变图片尺寸
            cv.imwrite(img_path,res) #原路径存放


if __name__ == '__main__':
    filepath = 'S:/INPUT/NoMove/Train-2/64/'
    img_resize(filepath)
