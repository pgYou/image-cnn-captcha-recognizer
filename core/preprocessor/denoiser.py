#coding:utf-8
import copy
from PIL import Image
import numpy as np
import cv2

from core.image import captchaImg

class deNoicer:
    def toBina_PIL(self, imgC):
        """
        将彩图转化为二值图
        :param imgC:
        :return:
        """
        imgC.capImg = imgC.capImg.convert('1')
        return imgC

    def toGray(self, imgC):
        """将彩图转化成灰度图"""
        imgC.capImg = imgC.capImg.convert('L')
        return imgC

    def toBina(self, imgC, key=0):
        """
        优化版二值化算法，指定key，不指定通过计算得出key
        :param imgC:caotchaImg 类
        :param key: int 类
        :return: caotchaImg 类
        """
        height = 0
        width = 0

        # 判断图片是否为灰度图，如果不是转化为灰度图
        if imgC.capImg.mode!= 'L':
            imgC = self.toGray(imgC)

        #获取图片灰度像素数组
        matrix = np.asarray(imgC.capImg, 'f')
        height = matrix.shape[0]
        width = matrix.shape[1]
        # matrix.flags.writeable = True

        # 如果没有指定阈值
        if key==0:
            sum = 0
            for i in range(height):
                for j in range(width):
                    sum+=matrix[i][j]
            key = sum/(width*height)

        # 二值化
        for i in range(height):
            for j in range(width):
                if matrix[i][j]>key:
                    matrix[i][j] = 255
                else:
                    matrix[i][j] = 0

        imgC.capImg  = Image.fromarray(matrix)

        return self.toBina_PIL(imgC)

    def deNoise_Bina_8Near(self, imgC, N=5, Z=1):
        """
        利用像素八领域，对二值图去噪点
        :param imgC: caotchaImg 类
        :param N: int 降噪等级
        :param Z:降噪次数
        :return:caotchaImg 类
        """
        if imgC.capImg.mode != '1':
            return None

        matrix = np.asarray(imgC.capImg,'f')
        # print(matrix)
        self.__Bina2Bina_255(matrix)
        height = matrix.shape[0]
        width = matrix.shape[1]
        for k in range(Z):
            print("dNoice with "+ str(N))
            matrix2 = copy.deepcopy(matrix)

            for i in range(1, height - 1):
                for j in range(1, width - 1):

                    if self.__isNoisePixel(matrix2, i, j, N):
                        matrix[i][j] = 255

        matrix = self.__clearBlackBorder(matrix)
        imgC.capImg = Image.fromarray(matrix)
        return self.toBina_PIL(imgC)

    def deNoise_Bina_rela(self, imgC, N=4, Z=1, xD=2, yD=2):
        """
        利用像素周围相关性，对二值图去噪点
        :param imgC: caotchaImg 类
        :param D:int 分离差，如果判断是噪点，代替取样距离
        :return:caotchaImg 类
        """
        if imgC.capImg.mode != '1':
            return None
        matrix = np.asarray(imgC.capImg, 'f')
        height = matrix.shape[0]
        width = matrix.shape[1]

        for k in range(Z):
            matrix2 = copy.deepcopy(matrix)
            for i in range(1,height-1):
                for j in range(1,width-1):
                    matrix[i][j]=self.__getPixel(matrix2,i,j,N,xD,yD)

        matrix = self.__clearBlackBorder(matrix)
        imgC.capImg = Image.fromarray(matrix)
        return self.toBina_PIL(imgC)

    def __isNoisePixel(self, imgMatrix, x, y, N=5):
        """
         判断该点是否为噪点像素，如果是噪点则返回True
        :param imgMatrix:
        :param x:
        :param y:
        :param N: 阈值，周围白色像素大于等于N视为噪点
        :return: bool
        """
        if imgMatrix[x][y] == 1:
            return False

        nearDots = 0 # (x,y)点八领域统计
        if 0 != imgMatrix[x-1][y-1]:
            nearDots += 1
        if 0 != imgMatrix[x-1][ y ]:
            nearDots += 1
        if 0 != imgMatrix[x-1][y+1]:
            nearDots += 1
        if 0 != imgMatrix[ x ][y-1]:
            nearDots += 1
        if 0 != imgMatrix[ x ][y+1] :
            nearDots += 1
        if 0 != imgMatrix[x+1][y-1]:
            nearDots += 1
        if 0 != imgMatrix[x+1][ y ] :
            nearDots += 1
        if 0 != imgMatrix[x+1][y+1] :
            nearDots += 1

        if nearDots >= N:
            return True
        else:
            return False

    def __getPixel(self, imgMatrix, x, y, N=4,xD=2,yD=2):
        """
        判断该点是否为噪点，如果判断为噪点则返回上邻点值，否则返回本像素点值
        N:该点与八领域相关性阈值0-8
        D:分离差，如果判断是噪点，代替取样距离
        """
        L = imgMatrix[x][y]
        nearDots = 0  # (x,y)点相关性统计,相关性低为孤立点
        if L == imgMatrix[x-1][y-1]:
            nearDots += 1
        if L == imgMatrix[x-1][ y ]:
            nearDots += 1
        if L == imgMatrix[x-1][y+1]:
            nearDots += 1
        if L == imgMatrix[ x ][y-1]:
            nearDots += 1
        if L == imgMatrix[ x ][y+1] :
            nearDots += 1
        if L == imgMatrix[x+1][y-1]:
            nearDots += 1
        if L == imgMatrix[x+1][ y ] :
            nearDots += 1
        if L == imgMatrix[x+1][y+1] :
            nearDots += 1

        if nearDots < N:
            return 255  if (imgMatrix[x-xD][y-yD]!=0) else 0   #相关性低可能为噪点，返回邻点
        else:
            return 255  if (imgMatrix[x][y]!=0) else 0


    def getNoiseLine(self, img):
        """
        对灰度图或者二值图边缘检测，如果传入二值图自动转化为只有黑白的灰度图
        :param img: PIL.image
        :return: 包含 直线起点和终点的列表
        """
        if img.mode != 'L':
            img = self.__Bina2Bina_255(img)

        edges = cv2.Canny(img,30,75,apertureSize=3)
        umat = cv2.HoughLinesP(cv2.UMat(np.asarray(edges)),1,np.pi/180,70,70)
        line = cv2.UMat.get(umat)

        return line

    def __Bina2Bina_255(self,matrix):
        """
        将0-1 的二值图图转化为0-255的二值图，便于计算
        """

        height = matrix.shape[0]
        width = matrix.shape[1]
        for i in range(0, height):
            for j in range(0, width):
                if matrix[i][j] ==1:
                    matrix [i][j] = 255

        return matrix

    def __clearBlackBorder(self,matrix):
        w = len(matrix)
        h = len(matrix[0])
        for i in range(w):
            matrix[i][h-1]=255
            matrix[i][0] = 255
        for i in range(h):
            matrix[0][i]=255
            matrix[w-1][i] = 255
        return matrix

dn = deNoicer()
