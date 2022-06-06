import uuid

import cv2
import imutils
import numpy as np
from imutils import contours

from updating_image import show_img


class Passport1:

    # Путь до оригинального изображения.

    # Путь до файлов с именем, фамилией, отчеством.
    nameFilePath = False
    surnameFilePath = False
    patronymicFilePath = False

    # Минимальное соотношение ширины к высоте рамки для фото.
    PHOTO_MIN_RATIO = 0.7

    # Минимальное соотношение длины к высоте красной линии,
    # которая разграничивает пасспорт на 2 части.
    LINE_SEPARATOR_MIN_RATIO = 8

    # Отступ по краям для результирующего изображения.
    # Если вырезать название вприктык, то распознавание будет плохое или
    # вовсе будет ошибка наподобие "Weak margin".
    RESULT_IMAGE_NAME_MARGIN = 3

    # Параметры для уменьшенного изображения
    RESIZED_IMAGE_HEIGHT = 600
    RESIZED_IMAGE_PHOTO_MIN_HEIGHT = 50
    RESIZED_IMAGE_PHOTO_MIN_WIDTH = 50
    RESIZED_IMAGE_NAME_MIN_WIDTH = 6
    RESIZED_IMAGE_NAME_MIN_HEIGHT = 3
    RESIZED_IMAGE_NAME_MAX_HEIGHT = 16
    RESIZED_IMAGE_NAME_MIN_X = 10
    RESIZED_IMAGE_NAME_MIN_RIHGTH_INDENT = 45

    # Расширение для создаваемых файлов.
    RESULT_IMAGES_EXTENSION = 'png'
    # Папку для сохранения изображений.
    RESULT_IMAGES_FOLDER = '/tmp'
    # Нахождение ФИО на изображении.
    # true - если найдено ФИО. Иначе false.
    def processFullName(self, image):
        #show_img(New_Rotate.rotate(self, image), 'rotated')
        #show_img(Update.sobel_operator(self, image), 'sobel_orig')
        image = self.makeCorrectOrientation(image)
        if (image is None):
            return False

        return self.processFullNameInternal(image)

    # Приводим изображение к правильной ориентации через последовательные повороты на 90 градусов.
    def makeCorrectOrientation(self, origImage):
        #resizedImage = imutils.resize(origImage.copy(), height=self.RESIZED_IMAGE_HEIGHT)
        resizedImage = origImage.copy()

        for degree in [0, 90, 180, 270]:
            rotatedImage = imutils.rotate_bound(resizedImage, degree)
            if (self.isRightOrientation(rotatedImage)):
                return imutils.rotate_bound(origImage, degree)

        return None

    # Правильная ли ориентация у изображения.
    # Основываемся на красной линии посередине паспорта и рамки для фото.
    def isRightOrientation(self, image):
        lineSeparatorInfo = self.getLineSeparatorInfo(image)
        if (lineSeparatorInfo is None):
            return False

        lineSeparatorRatio = lineSeparatorInfo['w'] / float(lineSeparatorInfo['h'])
        if (lineSeparatorRatio < self.LINE_SEPARATOR_MIN_RATIO):
            return False

        photoInfo = self.getPhotoInfo(image)
        if (photoInfo is None):
            return False

        # Рамка для фото должна быть ниже красной линии.
        return True if photoInfo['minY'] > lineSeparatorInfo['y'] else False

    # Нахождение красной линии посередине паспорта.
    # Если линия найдена, возвращается информация о ней в виде {x,y,w,h}, иначе None.
    def getLineSeparatorInfo(self, resizedImage):
        redImage = self.getRedImage(resizedImage)

        widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
        redModifiedImage = cv2.dilate(redImage, widthKernel)

        heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
        redModifiedImage = cv2.dilate(redModifiedImage, heightKernel)
        #show_img(redModifiedImage, 'redModifiedImage')
        imageContours = cv2.findContours(redModifiedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageContours = imutils.grab_contours(imageContours)

        lineContourInfo = None
        contourMaxWidth = 0
        for c in imageContours:
            (x, y, w, h) = cv2.boundingRect(c)
            if (w > contourMaxWidth):
                contourMaxWidth = w
                lineContourInfo = {'x': x, 'y': y, 'w': w, 'h': h}

        return lineContourInfo

    # Нахождение рамки для фото.
    # Если фото найдено, возвращается информация о ней в виде {minX,maxX,minY,maxY}, иначе None
    def getPhotoInfo(self, resizedImage):
        redImage = self.getRedImage(resizedImage)

        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        redModifiedImage = cv2.dilate(redImage, squareKernel)

        imageContours = cv2.findContours(redModifiedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageContours = imutils.grab_contours(imageContours)

        photoInfo = None
        maxContourWidth = 0

        for c in imageContours:
            maxInColumns = np.amax(c, axis=0)
            minInColumns = np.amin(c, axis=0)

            # Пример формата: [[25 631]]
            height = maxInColumns[0][0] - minInColumns[0][0]
            width = maxInColumns[0][1] - minInColumns[0][1]
            ratio = width / height if width < height else height / width

            if height > self.RESIZED_IMAGE_PHOTO_MIN_HEIGHT and \
                width > self.RESIZED_IMAGE_PHOTO_MIN_WIDTH and \
                ratio > self.PHOTO_MIN_RATIO and \
                width > maxContourWidth:
                    maxContourWidth = width

                    photoInfo = {
                        'minX': minInColumns[0][0],
                        'maxX': maxInColumns[0][0],
                        'minY': minInColumns[0][1],
                        'maxY': maxInColumns[0][1]
                    }

        return photoInfo

    # Получение только красной части на изображении
    def getRedImage(self, image):
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        redRange1 = cv2.inRange(hsvImage, np.array([0, 70, 50]), np.array([10, 255, 255]))
        redRange2 = cv2.inRange(hsvImage, np.array([170, 70, 50]), np.array([180, 255, 255]))
        redRange3 = cv2.inRange(hsvImage, np.array([160, 100, 100]), np.array([179, 255, 255]))

        redImage = cv2.bitwise_or(redRange1, redRange2)
        redImage = cv2.bitwise_or(redImage, redRange3)

        #show_img(redImage, 'Red_image')
        return redImage

    # Нахождение ФИО на изображении, которое уже находится в правильной ориентации,
    # имеет серединную красную линию и рамку для фото.
    def processFullNameInternal(self, origImage):
        resizedImage = imutils.resize(origImage.copy(), height=self.RESIZED_IMAGE_HEIGHT)

        lineSeparatorInfo = self.getLineSeparatorInfo(resizedImage)
        photoInfo = self.getPhotoInfo(resizedImage)

        fullNameMinX = photoInfo['maxX']
        fullNameMaxX = lineSeparatorInfo['x'] + lineSeparatorInfo['w']
        fullNameMinY = lineSeparatorInfo['y'] + lineSeparatorInfo['h']
        fullNameMaxY = photoInfo['maxY']

        # Вырезаем часть, где находится ФИО: ниже красной линии, и правее рамки для фото.
        fullNameImage = resizedImage[fullNameMinY:fullNameMaxY, fullNameMinX:fullNameMaxX]

        fullNameModifiedImage = cv2.cvtColor(fullNameImage, cv2.COLOR_BGR2GRAY)
        fullNameModifiedImage = cv2.threshold(fullNameModifiedImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        fullNameModifiedImage = cv2.morphologyEx(fullNameModifiedImage, cv2.MORPH_CLOSE, rectKernel)

        imageContours = cv2.findContours(fullNameModifiedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageContours = imutils.grab_contours(imageContours)

        (sortedContours, boundingBoxes) = contours.sort_contours(imageContours, method="top-to-bottom")
        countNameContours = 0

        origImageRatio = origImage.shape[0] / float(self.RESIZED_IMAGE_HEIGHT)

        boxes = []
        for contour in sortedContours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if w > self.RESIZED_IMAGE_NAME_MIN_WIDTH and \
                h > self.RESIZED_IMAGE_NAME_MIN_HEIGHT and \
                h < self.RESIZED_IMAGE_NAME_MAX_HEIGHT and \
                x > self.RESIZED_IMAGE_NAME_MIN_X and \
                fullNameImage.shape[1] - x > self.RESIZED_IMAGE_NAME_MIN_RIHGTH_INDENT:
                    countNameContours = countNameContours + 1

                    startX = int((x + fullNameMinX - self.RESULT_IMAGE_NAME_MARGIN) * origImageRatio)
                    endX = int((x + w + fullNameMinX + self.RESULT_IMAGE_NAME_MARGIN) * origImageRatio)
                    startY = int((y + fullNameMinY - self.RESULT_IMAGE_NAME_MARGIN) * origImageRatio)
                    endY = int((y + h + fullNameMinY + self.RESULT_IMAGE_NAME_MARGIN) * origImageRatio)

                    origImageCut = origImage[
                        startY:endY,
                        startX:endX
                    ]
                    boxes.append((startX, startY, endX, endY))
                    #img_sobel = Update.sobel_operator(self, origImageCut)
                    #img_canny = Update.getEdges(self, origImageCut)
                    #show_img(img_sobel, 'sobel')
                    #show_img(Update.black_white(self, img_sobel), 'bl_wh')
                    #show_img(Update.draw_contours(self, img_sobel, origImageCut), 'sobel_operator')
                    #show_img(Update.draw_contours(self, img_canny, origImageCut), 'canny_operator')
                    show_img(origImageCut, 'Find_FIO')
                    filePath = self.getUniqueFilePath()
                    cv2.imwrite(filePath, origImageCut)

                    if (countNameContours == 1):
                        self.surnameFilePath = filePath
                    elif (countNameContours == 2):
                        self.nameFilePath = filePath
                    elif (countNameContours == 3):
                        self.patronymicFilePath = filePath

                    if (countNameContours == 3) :
                        break

        return boxes if countNameContours == 3 else []


    def getProcessedNameFilePaths(self):
        return self.getProcessedImagesVariants(self.nameFilePath)

    def getProcessedSurnameFilePaths(self):
        return self.getProcessedImagesVariants(self.surnameFilePath)

    def getProcessedPatronymicFilePaths(self):
        return self.getProcessedImagesVariants(self.patronymicFilePath)

    # Обработка изображения для улучшенного распознавания
    # Возвращается массив путей для преобразованных файлов
    def getProcessedImagesVariants(self, filePath):
        image = cv2.imread(filePath)
        grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscaled, (5, 5), 0)

        otsuThreshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        otsuThresholdFilePath = self.getUniqueFilePath()
        cv2.imwrite(otsuThresholdFilePath, otsuThreshold)

        customThreshold = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)[1]
        customThresholdFilePath = self.getUniqueFilePath()
        cv2.imwrite(customThresholdFilePath, customThreshold)

        customThreshold2 = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]
        customThresholdFilePath2 = self.getUniqueFilePath()
        cv2.imwrite(customThresholdFilePath2, customThreshold2)

        return [otsuThresholdFilePath, customThresholdFilePath, customThresholdFilePath2]

    def getUniqueFilePath(self):
        return self.RESULT_IMAGES_FOLDER + '/' + str(uuid.uuid4()) + '.' + self.RESULT_IMAGES_EXTENSION
"""
if __name__ == '__main__':
    #ps = Passport('templates/orig7.jpeg')
    for i in range(8, 9):
        ps = Passport1(f"templates/orig{i}.jpeg")
        try:
            ps.processFullName()
        except Exception as msg:

            print(i, msg)"""

