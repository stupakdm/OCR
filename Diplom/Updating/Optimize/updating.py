import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import imutils
from imutils.object_detection import non_max_suppression


class Update:


    def resize_image(self, img):
        H, W = img.shape[:2]
        print(img, img.shape)
        img = img[:H//2, :W//2, :]
        #show_img(img, 'after resize')
        return img

    def find_contours(self, img):
        H, W = img.shape[:2]
        img = self.bright_contrast(img, contrast=1.5)
        show_img(img, 'Contrast')
        img = self.black_white(img)
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_x, min_y, max_x, max_y = 0,0,H,W
        for cont in contours:
            print(cont)


    def check_image(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        show_img(hsv, 'HSV')

        hsv_min = np.array((107, 0, 130), np.uint8)
        hsv_max = np.array((110, 255, 190), np.uint8)

        thresh = cv2.inRange(hsv, hsv_min, hsv_max)
        H, W = thresh.shape[:2]
        print(H*W)
        print(np.sum(thresh)/255)
        show_img(thresh, 'thresh 1')
        if np.sum(thresh)/255 > H*W/15:
            heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
            thresh = cv2.erode(thresh, kernel=heightKernel, borderType=cv2.RETR_EXTERNAL)
            show_img(thresh, 'thresh 2')

        widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
        blueModifiedImage = cv2.dilate(thresh, widthKernel)
        show_img(blueModifiedImage, 'Blue')

        imageContours = cv2.findContours(blueModifiedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageContours = imutils.grab_contours(imageContours)

        lineContourInfo = None
        contourMaxWidth = 0
        for c in imageContours:
            (x, y, w, h) = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(blueModifiedImage, [box], 0, 255,2)
            if (w > contourMaxWidth):
                contourMaxWidth = w
                lineContourInfo = {'x': x, 'y': y, 'w': w, 'h': h}
                #cv2.rectangle(blueModifiedImage, (x, y), (x + w, y + h), 255, 2)
        new_image = np.zeros((H,W), np.float32)
        cv2.rectangle(new_image, (lineContourInfo['x'], lineContourInfo['y']),
                      (lineContourInfo['x']+lineContourInfo['w'], lineContourInfo['y']+lineContourInfo['h']), 255, 2)
        show_img(new_image, 'Rectangles')
        print('lineContourInfo', lineContourInfo)
            #return lineContourInfo

    def cropp_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)

        ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255,0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)

        print('corners', corners)
        res = np.hstack((centroids, corners))
        res = np.int0(res)
        print('res', res)
        img[res[:,1],res[:,0]] = [0,0,255]
        img[res[:,3], res[:,2]] = [0,255,0]

        show_img(img, 'Cropp')



    def corner_images(self, img):
        operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        operatedImage = np.float32(operatedImage)
        #show_img(operatedImage, 'Operated')
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
            #show_img(img, 'CornerHarris')
        return img

    def try_contrast_CLACHE(self, img):
        clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l2 = clahe.apply(l)

        lab = cv2.merge((l2, a, b))
        img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img2

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

    # Функция для преобразования МАСШТАБА изображения
    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized


    def calc_hist(self, img):
        img = self.black_white(img)
        h, w = img.shape[:2]
        pixelSequence = img.reshape([h * w, ])
        numberBins = 256
        histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                          facecolor='black', histtype='bar')
        plt.xlabel("gray label")
        plt.ylabel("number of pixels")
        plt.axis([0, 255, 0, np.max(histogram)])
        plt.show()

    def linear_change_hist(self, img, alpha = 1.0):
        out = alpha*img
        out[out>255] = 255
        out = np.around(out)
        out = out.astype(np.uint8)
        show_img(out, "change_hist")
        return out

    def equa_CLAHE_change_hist(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = self.black_white(img)
        #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        #l, a, b = cv2.split(lab)

        #l2 = clahe.apply(l)

        dst = clahe.apply(img)

        show_img(dst, 'Using CLAHE')

        equa = cv2.equalizeHist(img)
        show_img(equa, "Using equa")



    # DNN для нахождения границ с текстом
    def dnn_using(self, img):
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        orig = img.copy()
        # model = cv2.dnn.readNet(model = "frozen_east_text_detection.pb", config='resnet50.config.txt', framework = 'Tensorflow')
        model = cv2.dnn.readNet(model="frozen_east_text_detection.pb")
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
# print(img.shape)
    # print(img)
    plt.imshow(img)
    plt.title(title)
    plt.show()
    #pass