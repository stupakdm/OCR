from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Update:

    # ДОДЕЛАТЬ алгоритм
    @staticmethod
    def _check_boards(col, left, right):
        '''fl = 0
        if (col[0] >= left and col[0] <= right):
            fl += 1
        if col[1] >= left and col[1] <= right:
            fl += 1
        if col[2] >= left and col[2] <= right:
            fl += 1
        if fl >= 2:
            return True
        return False'''
        if col >= left and col <= right:
            return True
        return False

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

    def extract_contours_text(self, image):
        image1 = np.copy(image)
        cols, rows = image.shape[0:2]
        for x in range(cols):
            for y in range(rows):
                col = image[x, y]
                if max(col) - min(col) <= 10:
                    # text
                    # if self._check_boards(col, 40, 105):
                    if self._check_boards(col[0], 40, 115) and self._check_boards(col[1], 40, 115) and self._check_boards(col[2],40, 115):
                        '''new_col = np.array([0] * 3)
                        fl = 0
                        for i in range(max(0, x - 1), min(cols, x + 2)):
                            for j in range(max(0, y - 1), min(rows, y + 2)):
                                if i == x and j == y:
                                    continue
                                col1 = image[i, j]
                                if self._check_boards(col1[0], 40, 115) and self._check_boards(col1[1], 40, 115) and self._check_boards(col1[2], 40, 115):
                                    continue
                                else:

                                    new_col += 10
                        image1[x,y] = new_col
                        if max(col) - 125 > 0:
                            image1[x, y] += max(col) - 125'''
                        continue
                    else:
                        new_col = np.array([255] * 3)
                        '''fl = 0
                        c = 0
                        for i in range(max(0, x - 1), min(cols, x + 2)):
                            for j in range(max(0, y - 1), min(rows, y + 2)):
                                c += 1
                                if 201 <= i <= 208 and 418 <= j <= 423:
                                    print(image[i, j])
                                if i == x and j == y:
                                    continue
                                col1 = image[i, j]
                                if self._check_boards(col1[0], 40, 125) and self._check_boards(col1[1], 40, 125) and self._check_boards(col1[2], 40, 125):
                                    fl += 1
                                else:
                                    continue
                        if fl != 0:
                            new_col = np.array([20 * (c - fl) + max(col) - 125])'''
                        image1[x, y] = new_col
                else:
                    continue
                    # if self._check_boards(col, 40, 100):
                    #    image[x][y] = np.array([0, 0, 0])
                    # else:
                    #image1[x][y] = np.array([255, 0, 0])
        return image1

    colors = ('GRAY', 'LAB', 'HLS', 'XYZ', 'YCrCb', 'YUV')
    boards = ('CONSTANT', 'REPLICATE', 'REFLECT', 'ISOLATED')
    ex = ('OPEN', 'CLOSE', 'GRADIENT', 'TOPHAT', 'BLACKHAT')
    shapes = ('RECT', 'ELLIPSE', 'CROSS')

    def getStructure(self, k = (5,5), choice = 'RECT'):
        return cv2.getStructuringElement(cv2.__dict__['MORPH_'+choice], k)

    def morphology(self,  image, k = (5,5), choice = 'OPEN'):
        kernel = np.ones(k, np.uint8)
        opening = cv2.morphologyEx(image, cv2.__dict__['MORPH_'+choice], kernel)
        return opening

    @staticmethod
    def __choose_board(board):
        return cv2.__dict__['BORDER_'+board]
        #return border_type

    def dilation(self, image, k = (5,5), board = 'CONSTANT'):
        kernel = np.ones(k, np.uint8)
        dilate = cv2.dilate(image, kernel, borderType=Update.__choose_board(board), iterations = 1)
        return dilate

    def erosion(self, image, k= (5,5), board = 'CONSTANT'):
        kernel = np.ones(k, np.uint8)
        erosion = cv2.erode(image, kernel, borderType=Update.__choose_board(board), iterations = 1)
        return erosion

    def change_color(self, image, choice = 'GRAY'):
        return cv2.cvtColor(image, cv2.__dict__['COLOR_BGR2'+choice])

    # Черно-белое фото и обработка всяких контуров
    def black_white(self, image):
        # new_image = cv2.threshold(image, )
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return img

    # Сглаживание и фильтрация
    def change_smoothing_filtering(self, image, type='Gaus'):
        blur = 0
        if type == 'bilaterial':
            blur = cv2.bilateralFilter(image, 9, 41, 41)
        elif type == 'Median':
            blur = cv2.medianBlur(image, 7)
        elif type == 'Gaus':
            blur == cv2.GaussianBlur(image, (7, 7), 0)

        return blur

    def change_hsv(self, image, hue=0, satur=0, value=0):
        '''HSV -
            H - цветовой фон
            S - насыщенность
            V - яроксть

            '''
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(new_image)

        lim = 255

        h[h > lim] = 255
        h[h < abs(hue)] = abs(hue)
        print(h[h <= lim])
        if hue < 0:
            h[h <= lim] -= abs(hue)
        else:
            h[h <= lim] += abs(hue)

        s[s > lim] = 255
        s[s < abs(satur)] = abs(satur)
        # s[s <= lim] += satur
        if satur < 0:
            s[s <= lim] -= abs(satur)
        else:
            s[s <= lim] += abs(satur)

        v[v > lim] = 255
        v[v < abs(value)] = abs(value)
        # v[v<=lim] += value
        if value < 0:
            v[v <= lim] -= abs(value)
        else:
            v[v <= lim] += abs(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def change_bright_contrast(self, image, bright=0, contr1=1.0, contr2=1.0):
        b = np.zeros(image.shape, image.dtype)
        c = cv2.addWeighted(image, contr1, b, contr2, bright)
        return c

    # Изменить яркость и констрастность изображения
    def bright_contrast(self, image, bright=0, contrast=1.0):
        ''' bright - [0, 100]
            contrast = [1.0,3.0]'''
        new_image = np.zeros(image.shape, image.dtype)
        # new_image = list(map(self.__change_pixel, new_image, ))
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y, x, c] = np.clip(contrast * image[y, x, c] + bright, 0, 255)

        return new_image

    # Изменить размер изображения
    def resize_img(self, image, rows, cols, interpolation = cv2.INTER_CUBIC):
        print(rows,cols)
        img = cv2.resize(image, (rows, cols), interpolation = interpolation)
        return img

    def fix_pixels(self, image):
        rows, cols, _ = image.shape
        for x in range(rows):
            for y in range(cols):
                if image[x, y, 0] > 150 and image[x, y, 1] > 150 and image[x, y, 2] > 150:
                    image[x, y] = np.array([255] * 3)
                else:
                    image[x, y] = np.array([0] * 3)
        print(image)
        return image

    # Можно добавить threshold
    def img_threshold(self, image, type=cv2.THRESH_BINARY):
        ret, thresh1 = cv2.threshold(image, 100, 255, type)
        # ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        # ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
        # ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
        # ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
        return thresh1

    # Резкость
    def sharp_image(self, image, t=9, n=1):
        kernel = np.array([[-1, -1, -1],
                           [-1, t, -1],
                           [-1, -1, -1]], np.float32)
        kernel = (1 / n) * kernel
        sharpened = cv2.filter2D(image, -1,
                                 kernel)  # applying the sharpening kernel to the input image & displaying it.
        return sharpened

    def bilaterial_filter(self, image, d=0, sigmaColor=10, sigmaSpace=30):
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    # Повышение резкости, но нужно ещё проэкспериментировать
    def true_sharp(self, image, origImg):
        image = self.gaussian_blur(image, kernel=(0, 0), sigma=3.0)
        image = cv2.addWeighted(image, 1.5, origImg, -0.5, 0)
        return image

    def gaussian_blur(self, image, kernel=(3, 3), sigma=0.0):
        gaus = cv2.GaussianBlur(image, kernel, sigma)
        return gaus

    def median_blur(self, image, m=5):
        median = cv2.medianBlur(image, m)
        return median

    # Не особо улучшает качество(может лучше не использовать)
    def averaging_blur(self, image, kernel=(5, 5)):
        average = cv2.blur(image, kernel)
        return average

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
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

    # def
    # def calcGray
