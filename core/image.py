#coding:utf-8

import cv2
import numpy as np
import os

from PIL import Image

class captchaImg:
    def __init__(self, s='', img=0, chrList=None):
        self.code = s
        self.capImg = img
        self.chrList = chrList if chrList else []
    def show(self):
        self.capImg.show()


    def showChrList(self):
        for each in self.chrList:
            each.show()


    def save(self,filePath):
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        self.capImg.save(filePath+self.code+'.png', 'PNG')

    def save_With_ChrList(self, filePath):
        filePath = filePath+self.code+"/"
        if not os.path.exists(filePath):
            os.makedirs(filePath)

        self.capImg.save(filePath+self.code+'.png', 'PNG')
        i = 0
        for each in self.chrList:
          each.save(filePath+str(i)+'.png', 'PNG')
          i += 1

    def save_ChrList(self, filePath):

        if not os.path.exists(filePath):
            os.makedirs(filePath)
        code = self.code
        i = 0

        for each in self.chrList:
            path = filePath+'{}_{}_{}.png'.format(code, i, code[i],)
            each.save(path, 'PNG')
            i += 1
