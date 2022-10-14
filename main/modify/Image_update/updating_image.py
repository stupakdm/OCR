from __future__ import print_function

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from rotate_image import show_img
from imutils.object_detection import non_max_suppression


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

    def find_left_angle(self, image, flag):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont = self.find_max_ext(contours, fl=0)
        extLeft = tuple(cont[cont[:, :, 0].argmin()][0])
        extRight = tuple(cont[cont[:, :, 0].argmax()][0])
        extTop = tuple(cont[cont[:, :, 1].argmin()][0])
        extBot = tuple(cont[cont[:, :, 1].argmax()][0])
        """print("extleft", extLeft)
        print('extRight', extRight)
        print("extTop", extLeft)
        print('extBot', extRight)"""
        if flag == 1:
            return abs(extLeft[0] - extRight[0])
        else:
            return abs(extTop[1] - extBot[1])

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

    def draw_contours(self, image, orig_image):
        bl_min = np.array((80), np.uint8)
        bl_max = np.array((255), np.uint8)
        lim_max = 255
        lim_min = 0
        image[image > 120] = 255
        print((image < 40).shape)
        image[image < 40] = 0
        # indx = np.where(np.logical_and(image>40, image<120))[0]
        # image[indx] = image*2
        mask1 = np.logical_and((image >= 40, image <= 120))
        image = image[np.where((image > 40) & (image < 120))]
        idx = np.all([image > 40, image < 120], 0)[0]
        image[idx] = image * 2
        # logic = np.logical_and(image > 40, image<120)
        # print(logic.shape)
        # image[logic] = image*2
        # print(np.logical_and(image > 40, image<120))
        # image[np.logical_and(image > 40, image<120)] = image*2
        # invert = cv2.bitwise_not(image)
        thresh = cv2.inRange(image, bl_min, bl_max)
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # print(contours[-2])
        # contours = contours[-2]
        img = cv2.drawContours(orig_image, contours, contourIdx=-1, color=(0, 0, 0), thickness=cv2.FILLED,
                               hierarchy=hierarchy)
        return img
        # for c in contours:

    def sobel_operator(self, image):
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        src = cv2.GaussianBlur(image, (3, 3), 0)

        gray = self.black_white(src)

        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        return grad

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
                    if self._check_boards(col[0], 40, 115) and self._check_boards(col[1], 40,
                                                                                  115) and self._check_boards(col[2],
                                                                                                              40, 115):
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
                    # image1[x][y] = np.array([255, 0, 0])
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

        print(dest.shape)
        print(img.shape)
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


        if (((ind_x[0] - 0)+(W-ind_x[-1])) > abs(ind_x[-1]-ind_x[0])//2) or (((ind_y[0] - 0)+(H-ind_y[-1])) > abs(ind_y[-1]-ind_y[0])//2):
            print('Cropp width and height: ', H, W)
            img = img[ind_y[0]:ind_y[-1], ind_x[0]:ind_x[-1]]
            img = self.gaussian_blur(img, kernel=(3, 3))
            # print(np.where(dest > 0.01* dest.max()))
            # img[dest > 0.01 * dest.max()] = [0,0,255]
            show_img(img, 'CornerHarris')
        return img

    def try_contrast_CLACHE(self, img):
        clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l2 = clahe.apply(l)

        lab = cv2.merge((l2, a, b))
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img2

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

    def getContours(self, image, orig_img):
        """
        Get contours from image(GRAY, or after HSV or after CANNY or GaussianBLUR)
        """
        res = orig_img.copy()
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        area_max = cv2.contourArea(contours[0])
        con = contours[0]
        for c in contours:
            # x,y,w,h, = cv2.boundingRect(c)

            area = cv2.contourArea(c)
            if area > area_max:
                area_max = area
                con = c
            cv2.drawContours(res, [c], -1, (0, 0, 0), 1)
        cv2.drawContours(res, [con], -1, (255, 0, 0), 1)
        return res

    def getEdges(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(img, (3, 3), 0)
        show_img(blur_img, 'Gausian')

        edges = cv2.Canny(blur_img, 100, 200)
        show_img(edges, 'canny')
        return edges

    def subtract(self, img):
        gr = img.copy()

        for i in range(5):
            kernel = self.getStructure(k=(2 * i + 1, 2 * i + 1), choice='ELLIPSE')

            gr = self.morphology(gr, kernel, 'CLOSE')
            gr = self.morphology(gr, kernel, 'OPEN')

        dif = cv2.subtract(gr, img)
        bw = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        dark = cv2.threshold(gr, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        darkpix = img[np.where(dark > 0)]
        darkpix = cv2.threshold(darkpix, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        bw[np.where(dark > 0)] = darkpix.T
        return bw

    def getStructure(self, k=(5, 5), choice='RECT'):
        return cv2.getStructuringElement(cv2.__dict__['MORPH_' + choice], k)

    def morphology(self, image, k=(5, 5), choice='OPEN'):
        # kernel = np.ones(k, np.uint8)
        kernel = k
        opening = cv2.morphologyEx(image, cv2.__dict__['MORPH_' + choice], kernel)
        return opening

    @staticmethod
    def __choose_board(board):
        return cv2.__dict__['BORDER_' + board]
        # return border_type

    def dilation(self, image, k=(5, 5), board='CONSTANT'):
        kernel = np.ones(k, np.uint8)
        dilate = cv2.dilate(image, kernel, borderType=Update.__choose_board(board), iterations=1)
        return dilate

    def erosion(self, image, k=(5, 5), board='CONSTANT'):
        kernel = np.ones(k, np.uint8)
        erosion = cv2.erode(image, kernel, borderType=Update.__choose_board(board), iterations=1)
        return erosion

    def change_color(self, image, choice='GRAY'):
        return cv2.cvtColor(image, cv2.__dict__['COLOR_BGR2' + choice])

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
            blur = cv2.GaussianBlur(image, (7, 7), 0)

        return blur

    def using_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        show_img(hsv, 'hsv')
        lab = self.change_color(img, choice='LAB')
        show_img(lab, 'lab')
        hls = self.change_color(img, choice='HLS')
        show_img(hls, 'hls')
        xyz = self.change_color(img, choice='XYZ')
        show_img(xyz, 'xyz')
        ycc = self.change_color(img, choice='YCrCb')
        show_img(ycc, 'YCrCb')
        yuv = self.change_color(img, choice='YUV')
        show_img(yuv, 'yuv')

        # FOR LAB
        lower_black = np.array([85, 125, 125])
        upper_black = np.array([130, 135, 135])
        # lower_black = np.array([0,0,0])
        # upper_black = np.array([130, 125, 130])
        return None

    # DNN для нахождения границ с текстом
    def dnn_using(self, img):
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        orig = img.copy()
        # model = cv2.dnn.readNet(model = "frozen_east_text_detection.pb", config='resnet50.config.txt', framework = 'Tensorflow')
        model = cv2.dnn.readNet(model="../frozen_east_text_detection.pb")
        # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        (H, W) = img.shape[0:2]
        newW = 640
        newH = 640
        rW = W / float(newW)
        rH = H / float(newH)
        # rW = W
        # rH = H
        print(img.shape[:2])
        img = cv2.resize(img, (newW, newH))
        (H, W) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (W, H),
                                     (123.68, 116.78, 103.94),
                                     swapRB=True, crop=False)
        print(blob.shape)
        model.setInput(blob)
        (scores, geometry) = model.forward(layerNames)

        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.5:
                    continue
                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score to
                # our respective lists
                for rect in rects:
                    fl = 0
                    if rect[0] <= startX <= rect[2] and rect[1] <= startY <= rect[3]:
                        fl = 1
                    elif rect[0] <= endX <= rect[2] and rect[1] <= endY <= rect[3]:
                        fl = 1
                    elif rect[0] <= startX <= rect[2] and rect[1] <= endY <= rect[3]:
                        fl = 1
                    elif rect[0] <= endX <= rect[2] and rect[1] <= startY <= rect[3]:
                        fl = 1

                    if fl == 1:
                        rect[0] = min(rect[0], startX)
                        rect[1] = min(rect[1], startY)
                        rect[2] = max(rect[2], endX)
                        rect[3] = max(rect[3], endY)

                deltaX = int((endX - startX) * 0.1)
                startX -= deltaX
                endX += deltaX
                deltaY = int((endY - startY) * 0.1)
                startY -= deltaY
                endY += deltaY

                rects.append([startX, startY, endX, endY])
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        new_boxes = []
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            new_boxes.append((startX, startY, endX, endY))
            # draw the bounding box on the image
            # cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # show the output image
        # show_img(orig, 'Text Detection')
        """#net.
        #net.setInput(blob)
        #model = cv2.dnn_TextDetectionModel("textbox.prototxt", "TextBoxes_icdar13.caffemodel"
        #)
        #textSpotter = cv2.text.TextDetectorCNN_create(
        #    "textbox.prototxt", "TextBoxes_icdar13.caffemodel"
        #)
        sr, _ = model.detectTextRectangles(img)
        for rect in sr:
            points = cv2.boxPoints(rect)
            cv2.drawContours(img, points, color=[0,255,0], thickness=3)
        show_img(img, 'DNN')"""
        return new_boxes

    def get_to_norm_contrast(self, img):
        b, g, r = cv2.split(img)

        b -= np.uint8(np.min(b) - 1)
        g -= np.uint8(np.min(g) - 1)
        r -= np.uint8(np.min(r) - 1)

        b *= np.uint8(2)
        lim = 255
        b[b > lim] = lim
        # s[s <= lim] += satur

        g *= np.uint8(2)
        g[g > lim] = lim

        r *= np.uint8(2)
        r[r > lim] = lim

        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                b[i, j], g[i, j], r[i, j] = [max(b[i, j], g[i, j], r[i, j])] * 3

        fin_contr = cv2.merge((b, g, r))
        show_img(fin_contr, 'Try_contr')
        return fin_contr

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

    def add_alpha_channel(self, img):
        alpha_data = np.zeros((img.shape[0:2]))
        alpha_data += np.uint8(255)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha_data
        show_img(rgba, 'Using alpha channel')
        return rgba

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
    def resize_img(self, image, x_col=1.0, x_rows=1.0, interpolation=cv2.INTER_CUBIC):
        cols, rows = image.shape[0:2]
        print(cols, rows)
        img = cv2.resize(image, (int(rows * x_rows), int(cols * x_col)), interpolation=interpolation)
        show_img(img, 'resize')
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

    def gaussian_blur(self, image, kernel=(5, 5), sigma=0.0):
        gaus = cv2.GaussianBlur(image, kernel, sigma)
        return gaus

    def median_blur(self, image, m=5):
        median = cv2.medianBlur(image, m)
        return median

    # Не особо улучшает качество(может лучше не использовать)
    def averaging_blur(self, image, kernel=(5, 5)):
        average = cv2.blur(image, kernel)
        return average

    def blob_detector(self, img):
        detector = cv2.SimpleBlobDetector()

        keypoints = detector.detect(img)
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        show_img(im_with_keypoints, 'Blob')
        return img

    def aprox_poly(self, img, orig_img):
        h, w = img.shape[0:2]

        def update(level):
            dst = np.zeros((h, w, 3), np.uint8)
            level -= 3
            cv2.drawContours(dst, contours, 0, (128, 255, 255),
                             3, cv2.LINE_AA, hierarchy)
            show_img(dst, 'cont')
            return dst

        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        contours = [cv2.approxPolyDP(con, 3, True) for con in contours]
        print(contours)
        level = 3
        image = update(level)
        cv2.drawContours(orig_img, contours, 0, (128, 255, 255),
                         3, cv2.FILLED, hierarchy, abs(level - 3))
        show_img(orig_img, 'aprox_poly')
        return image

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
