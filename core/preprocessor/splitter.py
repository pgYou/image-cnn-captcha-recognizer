import copy

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
from itertools import groupby

from core.image import captchaImg

class spliter:

    def segmenter(self, imgC, debug = False):
        if type(imgC.capImg) != np.ndarray:
            if imgC.capImg.mode != "RGB":
                imgC.capImg = imgC.capImg.convert("RGB")
            imgC.capImg = np.asarray(imgC.capImg)
            imgC.capImg = cv.cvtColor(imgC.capImg, cv.COLOR_RGB2BGR)

        return self.segmenter_On_Outline(imgC, debug)

    def segmenter_On_Outline(self, imgC, debug=False):

        mat = imgC.capImg

        binary = cv.cvtColor(mat, cv.COLOR_BGR2GRAY)
        binary = np.clip(binary, 0, 255)

        binary = np.asarray(binary, np.uint8)
        # 二值化后保存为浮点型提高精度，不符合findContours函数的参数要求
        binary = self.__clearBlackBorder(binary)
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)#cvStartFindContours_Impl

        rectangles = []
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            if w<159 and w>10 and h>20:

                if len(rectangles) != 0:
                    x_, y_, w_, h_ = rectangles[-1]
                    if x_<x and y_ < y and x_+w_ >x+w and y_+h_>y+h:
                        continue

                rectangles.append([x, y, w, h])

        rectangles.sort(key=lambda x: x[0])

        img2 = copy.copy(imgC.capImg)
        img2 = Image.fromarray(imgC.capImg)

        for c in rectangles:
            x, y, w, h = c
            color = (0, 0, 0)
            cv.rectangle(imgC.capImg, (x, y), (x + w, y + h), color, 1)

        imgC.capImg = Image.fromarray(imgC.capImg)

        i = 0
        for c in rectangles:
            x, y, w, h = c
            imgC.chrList.append(img2.crop((x+1, y+1, x + w, y + h)))
            i += 1
        if debug:

            imgC.capImg.show()


        # 单个字符图片不足四张，查找最大图片分割
        while i < 4 and i>0:
            max_w = 0
            flag = 4
            for j in range(0, i):
                w = imgC.chrList[j].size[0]
                if w > max_w:
                    max_w = w
                    flag = j
            print("第"+str(flag+1)+"/"+str(i)+"张字符图片粘连，进行分割")

            # 利用水滴法分割，如果粘连宽度大于60且只有两个粘连字符，按三个字符粘连在一起处理
            if max_w > 60 and i==2:
                img1, img2, img3= self.drop_fall(imgC.chrList[flag], 3, debug)
                imgC.chrList[flag] = img1
                imgC.chrList.insert(flag + 1, img2)
                imgC.chrList.insert(flag + 2, img3)
                i += 2

            else:
                img1, img2 = self.drop_fall(imgC.chrList[flag], debug=debug)
                imgC.chrList[flag] = img1
                imgC.chrList.insert(flag+1, img2)
                i += 1

        for i in range(len(imgC.chrList)):
            imgC.chrList[i] = imgC.chrList[i].resize((25,40), Image.Resampling.LANCZOS)


        return imgC

    def drop_fall(self, img, mode=2, debug=False):
        """水滴法分割粘连的字符"""
        if img.mode != '1':
            img = img.convert('1')

        width, height = img.size
        # 获取垂直投影直方图
        hist_width = self.vertical(img)

        # 生成直方图并显示
        if debug:
            hist_img = Image.new('RGB', (width, height), "#FFFFFF")
            draw = ImageDraw.Draw(hist_img)

            for x in range(width):
                y = hist_width[x]
                draw.line((x, height-y, x, height), 'black')
            img.show()


        if mode == 2:
            # 获取滴水路径起始点
            start_x = self.get_start_x(hist_width)

            #开始滴水算法
            start_route = []
            for y in range(height):
                start_route.append((0, y))

            end_route = self.get_end_route(img, start_x)
            # 注意这里groupby
            # 按y分组，y相同的点取y大点
            filter_end_route = [max(list(k)) for _, k in groupby(end_route, lambda x: x[1])]

            if debug:
                for i, j in end_route:
                    hist_img.putpixel((i, j), (255, 0, 0))
                hist_img.show()


            img1 = self.do_split(img, start_route, filter_end_route)

            start_route = list(map(lambda x: (x[0] + 1, x[1]), filter_end_route))  # python3中map不返回list需要自己转换
            end_route = []
            for y in range(height):
                end_route.append((width - 1, y))
            img2 = self.do_split(img, start_route, end_route)

            return img1, img2

        elif mode == 3:
            start_x1 = self.get_start_x(hist_width[0:((width*2)//3)].copy(), width//8)
            start_x2 = self.get_start_x(hist_width[width//3:width].copy(), width//8) + start_x1

            start_route = []
            for y in range(height):
                start_route.append((0, y))


            mid_route = self.get_end_route(img, start_x1)
            end_route = self.get_end_route(img, start_x2)

            filter_mid_route = [max(list(k)) for _, k in groupby(mid_route, lambda x: x[1])]
            filter_end_route = [max(list(k)) for _, k in groupby(end_route, lambda x: x[1])]

            if debug:
                for i, j in filter_mid_route:
                    hist_img.putpixel((i, j), (255, 0, 0))
                for i, j in filter_end_route:
                    hist_img.putpixel((i, j), (255, 0, 0))
                hist_img.show()

            img1 = self.do_split(img, start_route, filter_mid_route)
            img2 = self.do_split(img, filter_mid_route, filter_end_route)

            end_route = []
            for y in range(height):
                end_route.append((width - 1, y))
            img3 = self.do_split(img, filter_end_route, end_route)

            return img1, img2, img3

    def get_nearby_pix_value(self, img_pix, x, y, j):
        """获取临近5个点像素数据"""
        if j == 1:
            return 0 if img_pix[x-1, y+1] == 0 else 1
        elif j == 2:
            return 0 if img_pix[x, y+1] == 0 else 1
        elif j == 3:
            return 0 if img_pix[x+1, y+1] == 0 else 1

        elif j == 4:
            return 0 if img_pix[x+1, y] == 0 else 1
        elif j == 5:
            return 0 if img_pix[x-1, y] == 0 else 1
        else:
            return 0

    def vertical(self, img):
        """传入二值化后的图片进行垂直投影"""
        if img.mode != 'L':
            img = img.convert('1')

        pixdata = img.load()
        w, h = img.size
        result = []
        for x in range(w):
            black = 0
            for y in range(h):
                if pixdata[x, y] == 0:
                    black += 1
            result.append(black)
        return result

    def get_start_x(self, hist_width, offest=4):
        """
        根据图片垂直投影的结果来确定起点
        hist_width中间值 前后取3个值 再这范围内取最小值
        """
        mid = len(hist_width) // 2

        temp = hist_width[mid - offest:mid + offest+1]
        return mid - offest + temp.index(min(temp))

    def get_end_route(self, img, start_x):
        """获取滴水路径"""
        if img.mode != '1':
            img = img.convert('1')
        width = img.size[0]
        height = img.size[1]
        left_limit = 0
        right_limit = img.size[0] - 1

        end_route = []
        cur_p = (start_x, 0)
        last_p = cur_p
        end_route.append(cur_p)

        while cur_p[1] < (height - 1) and cur_p[0] < (width - 1):
            sum_n = 0
            max_w = 0
            next_x = cur_p[0]
            next_y = cur_p[1]
            pix_img = img.load()
            #求wi = max_w
            for i in range(1, 6):
                cur_w = self.get_nearby_pix_value(pix_img, cur_p[0], cur_p[1], i) * (6 - i)
                sum_n += cur_w
                if max_w < cur_w:
                    max_w = cur_w

            if sum_n == 0:
                # 如果全黑则看惯性
                max_w = 4

            if sum_n == 15:
                # 全白垂直下落
                max_w = 6
            # print(sum_n, max_w, cur_p)
            if max_w == 1:
                next_x = cur_p[0] - 1
                next_y = cur_p[1]

            elif max_w == 2:
                next_x = cur_p[0]+ 1
                next_y = cur_p[1]
            elif max_w == 3:
                next_x = cur_p[0] + 1
                next_y = cur_p[1] + 1
            elif max_w == 5:
                next_x = cur_p[0] - 1
                next_y = cur_p[1] + 1
            elif max_w == 6:
                next_x = cur_p[0]
                next_y = cur_p[1] + 1

            elif max_w == 4:
                if next_x > last_p[0]:
                    # 向右
                    next_x = cur_p[0] + 1
                    next_y = cur_p[1] + 1
                if next_x < last_p[0]:
                    next_x = cur_p[0]
                    next_y = cur_p[1] + 1
                if sum_n == 0:
                    next_x = cur_p[0]
                    next_y = cur_p[1] + 1
            else:
                raise Exception("get end route error")

            if last_p[0] == next_x and last_p[1] == next_y:
                if next_x < cur_p[0]:
                    max_w = 5
                    next_x = cur_p[0] + 1
                    next_y = cur_p[1] + 1
                else:
                    max_w = 3
                    next_x = cur_p[0] - 1
                    next_y = cur_p[1] + 1
            last_p = cur_p

            if next_x > right_limit:
                next_x = right_limit
                next_y = cur_p[1] + 1
            if next_x < left_limit:
                next_x = left_limit
                next_y = cur_p[1] + 1
            cur_p = (next_x, next_y)
            end_route.append(cur_p)
        return end_route

    def do_split(self, source_image, starts, filter_ends):
        """
        具体实行切割
        : param starts: 每一行的起始点 tuple of list
        : param ends: 每一行的终止点
        """
        left = starts[0][0]
        top = starts[0][1]
        right = filter_ends[0][0]
        bottom = filter_ends[0][1]
        pixdata = source_image.load()
        # 统计画布大小
        len_s = len(starts)
        len_ed = len(filter_ends)
        if len_ed <len_s:
            d = len_s-len_ed
            for i in range(d):
                filter_ends.append((filter_ends[len_ed-1][0],len_ed+i))

        for i in range(len_s):
            left = min(starts[i][0], left)
            top = min(starts[i][1], top)
            right = max(filter_ends[i][0], right)
            bottom = max(filter_ends[i][1], bottom)

        if right > source_image.size[0]-3:
            right = filter_ends[0][0]
            for i in range(1,len_s):
                filter_ends[i]= (filter_ends[0][0], filter_ends[i][1])

        width = right - left + 1
        height = bottom - top + 1
        image = Image.new('1', (width, height), 255)

        for i in range(height):
            start = starts[i]
            end = filter_ends[i]
            for x in range(start[0], end[0] + 1):
                if pixdata[x, start[1]] == 0:
                    image.putpixel((x - left, start[1] - top), 0)
        return image

    def __clearBlackBorder(self, matrix):
        w = len(matrix)
        h = len(matrix[0])
        for i in range(w):
            matrix[i][h - 1] = 1
            matrix[i][0] = 1
        for i in range(h):
            matrix[0][i] = 1
            matrix[w - 1][i] = 1
        return matrix

spt = spliter()
