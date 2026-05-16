#coding:utf-8
from captcha.image import ImageCaptcha
import random
from core.image import captchaImg
class CaptchaGenerator:
    def GenerateRand(self,num=1,path='data/samples/'):
        """
        批量生成随机验证码图片，可以指定数量和保存路径
        :param num:
        :param path:
        :return:
        """
        f=open(path+'list.txt','a')
        image=ImageCaptcha()
        for i in range(num):
            code = self.getRandCode()
            image.write(code,path+code+'.png')
            f.write(code+'\n')
        f.close()
        return 0

    def Generate_train_img(self, num, path = None):
        capImgs = []

        for i in range(num):
            IC = ImageCaptcha()
            img = captchaImg()
            code = self.getRandCode()
            img.capImg = IC.create_captcha_image(code, '#8b8b83', '#ffdead')
            # img.capImg = IC.create_noise_dots(IC.capImg, '#8b8b83',number=0)
            img.code = code
            capImgs.append(img)
            if path != None:
                img.save(path)
        return capImgs


    def Generate_With_Code(self,code,path='data/samples/'):
        """
        生成指定验证码验证图片
        :param num:
        :param path:
        :return:
        """
        f=open(path+'list.txt','a')
        IC=ImageCaptcha()

        IC.write(code,path+code+'.png')
        f.write(code+'\n')
        f.close()
        return 0

    def GenerateCap(self,mode = '1'):
        """
        生成一个随机验证码的图片并返回封装类
        :param mode char 1:生成带点噪声和线条干扰的的验证码图片
                    2:生成只带点噪声的验证码图片
        :return:
        """
        code = self.getRandCode()
        img=captchaImg()
        IC = ImageCaptcha()
        img.code=code
        if mode == '1':
            img.capImg=IC.generate_image(code)
        elif mode == '2':
            img.capImg = IC.create_captcha_image(code,	'#8b8b83','#ffdead')
            img.capImg = IC.create_noise_dots(img.capImg, '#8b8b83')
        elif mode == '3':
            img.capImg = IC.create_captcha_image(code, '#8b8b83', '#ffdead')
            img.capImg = IC.create_noise_curve(img.capImg, '#8b8b83')
        elif mode == '4':
            img.capImg = IC.create_captcha_image(code, '#8b8b83', '#ffdead')
            img.capImg = IC.create_noise_curve(img.capImg, '#8b8b83')
            img.capImg = IC.create_noise_dots(img.capImg, '#8b8b83')
        return img

    def getRandCode(self,mode = '1'):
        """
        生成四位随机验证码，mode=1只包含字母，mode等于2 包含数字字母
        :param mode:
        :return:
        """
        code = ''
        for j in range(4):
            if mode == '1':
                k = random.randint(0, 25)
            else:
                k = random.randint(0,35)

            if k > 25:
                code = code + str(k - 26)
            else:
                code = code + chr(k + ord('a'))
        return code

cg = CaptchaGenerator()
