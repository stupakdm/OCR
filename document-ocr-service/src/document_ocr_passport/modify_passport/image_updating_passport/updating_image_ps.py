from __future__ import print_function

import os

import cv2
#import matplotlib.pyplot as plt
import numpy as np
# from rotate_image import show_img
#from imutils.object_detection import non_max_suppression


class Update:
    '''
    Класс для улучшения качества изображения

    '''
    def __int__(self):
        pass
    # ДОДЕЛАТЬ алгоритм

    @staticmethod
    def find_max_ext(contours, fl=0):
        max_extLeft = 0
        max_extRight = 0
        max_extTop = 0
        max_extBot = 0
        con = 0
        if len(contours) != 0:
            max_extLeft = tuple(contours[0][contours[0][:, :, 0].argmin()][0])
            max_extRight = tuple(contours[0][contours[0][:, :, 0].argmax()][0])
            max_extTop = tuple(contours[0][contours[0][:, :, 1].argmin()][0])
            max_extBot = tuple(contours[0][contours[0][:, :, 1].argmax()][0])
            con = contours[0]
        for cont in contours:
            extLeft = tuple(cont[cont[:, :, 0].argmin()][0])
            extRight = tuple(cont[cont[:, :, 0].argmax()][0])
            extTop = tuple(cont[cont[:, :, 1].argmin()][0])
            extBot = tuple(cont[cont[:, :, 1].argmax()][0])
            if abs(extLeft[0] - extRight[0]) > abs(max_extLeft[0] - max_extRight[0]):
                con = cont
                max_extLeft = extLeft
                max_extRight = extRight
            if abs(extTop[1] - extBot[1]) > abs(max_extTop[1] - max_extBot[1]):
                max_extTop = extTop
                max_extBot = extBot

        if fl == 0:
            return con
        else:
            return (max_extLeft[0] + abs(max_extLeft[0] - max_extRight[0]) // 2,
                    max_extTop[1] + abs(max_extTop[1] - max_extBot[1]) // 2)

    def find_rect(self, image, key):
        # image = Update.black_white(image)
        if key != 1:
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # print(thresh)
            # print(thresh.shape)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(contours)
            W, H = image.shape
            dif = np.array([[0, 0], [H - 1, 0], [0, W - 1], [H - 1, W - 1]])
            # print('new_arr', new_arr)
            # print('new_arr.shape', new_arr.shape)
            # dif = np.reshape(dif, (4,2))
            print('dif shape', dif.shape)
            # print('cont shape', contours[0].shape)
            for i in range(len(contours)):
                # j = 0
                new_arr = []
                shape = contours[i].shape[0]
                # print('cont[0]', contours[i][0])
                # print('ind', np.in1d(contours[i],dif))
                # b = np.setdiff1d(contours[i], dif)
                # print('b', b)
                # new_arr = np.array([[0,0], []])
                for j in range(contours[i].shape[0]):
                    if not np.any(np.all(dif == contours[i][j][0], axis=1)):
                        # print(contours[i][j].shape)
                        new_arr.append(contours[i][j])
                        # new_arr = np.append(new_arr, contours[i][j], axis = 0)

                new_arr = np.reshape(np.array(new_arr), (len(new_arr), 1, 2))
                # dif = np.append(dif, contours[i][j][0], axis=1)

                contours[i] = new_arr.copy()

            contours = tuple(contours)
        else:
            contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        # print(contours.shape)
        con = self.find_max_ext(contours, fl=0)
        # img = image.copy()
        # hull = cv2.convexHull(cont)
        # show_img(hull, 'HULL')
        # sm = cv2.arcLength(hull, True)
        # apd = cv2.approxPolyDP(cont, 0.02 * sm, True)
        # cv2.drawContours(img, [apd], -1, 100, 3)
        # cv2.drawContours(img, [cont], -1, 100, 3)
        # cv2.circle(img, extLeft, 8, (150), -1)
        # cv2.circle(img, extRight, 8, (175), -1)
        # cv2.circle(img, extTop, 8, (200), -1)
        # cv2.circle(img, extBot, 8, (225), -1)
        # show_img(img, 'Cont')
        # c = max(contours, key = cv2.contourArea)

        # hull = cv2.convexHull(con)
        # sm = cv2.arcLength(hull, True)

        # x,y,h,w = cv2.boundingRect(con)
        # cv2.rectangle(img, (x,y), (x+w, y+h), 255, -1)
        img = image.copy()
        img[:, :] = [0]
        rect = cv2.minAreaRect(con)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # print(box)
        w, h = img.shape
        sq = cv2.contourArea(box)
        # hull  =cv2.convexHull(con)
        # sq = cv2.contourArea(hull)
        flag = 0
        # print(w*h/sq)
        if w * h / sq > 10:
            flag = 1
        cv2.drawContours(img, [box], 0, 255, 3)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centre_x, centre_y = self.find_max_ext(contours, fl=1)
        # img = image.copy()
        img[:, :] = [0]
        h, w = img.shape
        w = w // 2
        h = h // 2
        cent_x = w - centre_x
        cent_y = h - centre_y
        box[:, 0] = box[:, 0] + cent_x
        box[:, 1] = box[:, 1] + cent_y
        # cv2.drawContours(img, [box], 0, 255, 3)
        # apd = cv2.approxPolyDP(con, 0.02*sm, True)
        # cv2.drawContours(img, [apd], -1, 255, 3)
        # if flag == 0:
        print(flag)
        cv2.drawContours(img, [box], 0, 255, thickness=3)
        # show_img(img, 'after_all')
        '''x,y,w,h = cv2.boundingRect(c)
        for cont in contours:
            cv2.drawContours(image, cont, -1, color, 5)
            show_img(image, 'contours')
            sm = cv2.arcLength(cont, True)
            apd = cv2.approxPolyDP(cont, 0.001*sm, True)
            if len(apd) == 4:
                cv2.drawContours(image, [apd], -1, (100), 3)
            color +=10'''
        show_img(img, 'contours')

        # if flag == 0:

        return img, flag


    def extract_red_contours(self, image):
        image1 = np.copy(image)
        cols, rows = image.shape[0:2]
        for x in range(cols):
            for y in range(rows):
                col = image[x, y]
                if max(col) - min(col) <= 12:
                    image1[x, y] = [0, 0, 0]
                else:
                    image1[x, y] = [255, 255, 255]

        return image1


    def change_dpi(self, img, dpi=300):
        cv2.imwrite('save_1.jpg', img)
        os.system(f'mogrify -set density {dpi} save_1.jpg')
        img = cv2.imread('save_1.jpg')
        os.system('rm save_1.jpg')
        return img

    def corner_images(self, img):
        operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        operatedImage = np.float32(operatedImage)

        dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

        dest = cv2.dilate(dest, None)

        #print(dest.shape)
        #print(img.shape)
        # print(dest)
        ind_y, ind_x = np.where(dest > 0.01 * dest.max())
        ind_y = np.sort(ind_y)
        ind_x = np.sort(ind_x)
        H, W = img.shape[:2]

        if ind_y[0] > 4:
            ind_y -= 4
        if ind_y[-1] < H - 4:
            ind_y[-1] += 4

        if ind_x[0] > 4:
            ind_x -= 4
        if ind_x[-1] < W - 4:
            ind_x[-1] += 4

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = self.gaussian_blur(img, kernel=(3, 3))
        # print(np.where(dest > 0.01* dest.max()))
        # img[dest > 0.01 * dest.max()] = [0,0,255]
        #show_img(img, 'CornerHarris')
        if (((ind_x[0] - 0)+(W-ind_x[-1])) > abs(ind_x[-1]-ind_x[0])//2) or (((ind_y[0] - 0)+(H-ind_y[-1])) > abs(ind_y[-1]-ind_y[0])//2):
            print('Cropp width and height: ', H, W)
            img = img[ind_y[0]:ind_y[-1], ind_x[0]:ind_x[-1]]
            img = self.gaussian_blur(img, kernel=(3, 3))
            # print(np.where(dest > 0.01* dest.max()))
            # img[dest > 0.01 * dest.max()] = [0,0,255]
            show_img(img, 'CornerHarris')
        return img


    colors = ('GRAY', 'LAB', 'HLS', 'XYZ', 'YCrCb', 'YUV')
    boards = ('CONSTANT', 'REPLICATE', 'REFLECT', 'ISOLATED')
    ex = ('OPEN', 'CLOSE', 'GRADIENT', 'TOPHAT', 'BLACKHAT')
    shapes = ('RECT', 'ELLIPSE', 'CROSS')

    # def drawcontours(self, ):

    def box_update(self, img):
        unsharp = self.unsharp_mask(img)
        gaus = self.gaussian_blur(unsharp, (5, 5))

        img = gaus

        widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        img = cv2.dilate(img, widthKernel, borderType=cv2.BORDER_REFLECT)

        heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        dilate = cv2.dilate(img, heightKernel)
        return dilate


    @staticmethod
    def __choose_board(board):
        return cv2.__dict__['BORDER_' + board]
        # return border_type


    def erosion(self, image, k=(5, 5), board='CONSTANT'):
        kernel = np.ones(k, np.uint8)
        erosion = cv2.erode(image, kernel, borderType=Update.__choose_board(board), iterations=1)
        return erosion


    # Черно-белое фото и обработка всяких контуров
    def black_white(self, image):
        # new_image = cv2.threshold(image, )
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return img

    # Сглаживание и фильтрация

    # Доделать функции с заменой hsv-фильтров (1)
    @staticmethod
    def hsv_change(arr, param):
        if param < 0:
            t = arr[np.where(arr > abs(param))].copy()
            # np.add(t, param, out=t, casting='unsafe')
            t += np.uint8(param)
            arr[np.where(arr <= abs(param))] = 0
            arr[np.where(arr > abs(param))] = t
        elif param > 0:
            t = arr[np.where(arr < 255 - param)].copy()
            # np.add(t, param, out=t, casting='unsafe')
            t += np.uint8(param)
            arr[np.where(arr >= 255 - param)] = 255
            arr[np.where(arr < 255 - param)] = t
        return arr

    # Доделать функции с заменой hsv-фильтров (2)
    def change_hsv(self, image, hue=0, satur=0, value=0):
        '''HSV -
            H - цветовой фон
            S - насыщенность
            V - яроксть

            '''
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        show_img(new_image, 'hsv')
        h, s, v = cv2.split(new_image)

        lim = 255

        # lim = 255 - hue
        # h[h>lim] = 255
        h = self.hsv_change(h, hue)
        s = self.hsv_change(s, satur)
        v = self.hsv_change(v, value)
        """if hue < 0:
            t = h[np.where(h>abs(hue))].copy()
            t[:] += hue
            h[np.where(h<=abs(hue))] = 0
            h[np.where(h>abs(hue))] = t
        else:
            t = h[np.where(h < 255-hue)].copy()
            t[:] += hue
            h[np.where(h >= 255 - hue)] = 255
            h[np.where(h < 255 - hue)] = t


        lim = 255 - satur
        s[s>lim] = 255
        s[s <= lim] += satur"""
        """s[s > lim] = 255
        s[s < abs(satur)] = abs(satur)
        # s[s <= lim] += satur
        if satur < 0:
            s[s <= lim] -= abs(satur)
        else:
            s[s <= lim] += abs(satur)"""

        """lim = 255 - value
        v[v>lim] = 255
        v[v<= lim] += value"""

        final_hsv = cv2.merge((h, s, v))
        show_img(final_hsv, 'HSV_CHANGE')
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img


    # Изменить яркость и констрастность изображения
    def bright_contrast(self, image, bright=0, contrast=1.0):
        ''' bright - [0, 100]
            contrast = [1.0,3.0]'''

        new_image = np.zeros(image.shape, image.dtype)
        # new_image = list(map(self.__change_pixel, new_image, ))
        """for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y,x,c] = np.clip(contrast*image[y,x,c] + bright, 0, 255)"""
        new_image = cv2.convertScaleAbs(image, alpha=contrast, beta=bright)
        # new_image[y, x] = np.clip(contrast * image[y, x] + bright, 0, 255)
        # for c in range(image.shape[2]):
        #    if image[y, x, c] < 80:
        #        new_image[y, x, c] = np.clip(contrast * image[y, x, c] + bright, 0, 255)

        return new_image

    # Изменить размер изображения


    def gaussian_blur(self, image, kernel=(5, 5), sigma=0.0):
        gaus = cv2.GaussianBlur(image, kernel, sigma)
        return gaus



    def unsharp_mask(self, image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        # widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
        try:
            blurred = self.gaussian_blur(image, kernel_size, sigma)

            # blurred = self.median_blur(image)
            # blurred = self.averaging_blur(image)
            sharpened = float(amount + 1) * image - float(amount) * blurred
            sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
            sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
            sharpened = sharpened.round().astype(np.uint8)
            if threshold > 0:
                low_contrast_mask = np.absolute(image - blurred) < threshold
                np.copyto(sharpened, image, where=low_contrast_mask)
            return sharpened
        except:
            return image


def show_img(img, title):
     pass
    # print(img.shape)
    # print(img)
     #plt.imshow(img)
     #plt.title(title)
     #plt.show()
# def
# def calcGray
