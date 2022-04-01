# from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from rotate_image import RotateImage, show_img, New_Rotate
from Update_photo import Update
from SearchBD import Search
from Choose_correct_word import Doctr
import uuid
import imutils
from imutils import contours

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract as pt


class Passport(RotateImage, New_Rotate,  Update, Doctr, Search):
    filePath = False

    new_file_path = False

    def __init__(self, filepath):
        self.filePath = filepath
        Doctr.choose_model(self)
        Search.start(self)
        # Doctr.add_files(self, self.filePath)
        # Doctr.find_contours(self)

    def test_quality(self):
        orig_image = self.__read_img()
        print("For Origin")
        cols, rows = orig_image.shape[0:2]
        print(cols, rows)
        change_size = np.array(range(10, 25, 5))/10
        hsv = (-30, 0, 30)
        for i in change_size:
            resize = Update.resize_img(self, orig_image, cols = int(cols*i), rows = int(rows*i), interpolation = cv2.INTER_CUBIC)
            #show_img(resize, f"Resize {i}")
            #chng_hsv = Update.change_hsv(self, resize, hue=hsv[0], value=hsv[0], satur=hsv[0])
            #black_white = Update.black_white(self, resize)
            #dilate = Update.dilation(self, black_white, k=(1,1))
            #erode = Update.erosion(self, dilate, k =(1,1))
            #show_img(erode, 'After_all')
            unsharp = Update.unsharp_mask(self, resize)
            print(f"OCR+origin+resize+{i}+black_white+dilate/erode+unsharp")  #hsv+{hsv[0]}+{hsv[0]}+{hsv[0]}
            print(self.ocr_core(unsharp))
        rotated = New_Rotate.rotate(self, orig_image)
        show_img(rotated, 'Rotated')
        #Update.change_hsv(self, unsharp, hue=h, value=v, satur=s), f'hue={h}, satur={s}, value={v}')
        '''rot_image = self.__rotate_img()
        print("For Rotate")
        cols, rows = orig_image.shape[0:2]
        for i in change_size:
            resize = Update.resize_img(self, rot_image, cols = int(cols*i), rows = int(rows*i))
            #show_img(resize, f"Resize {i}")
            chng_hsv = Update.change_hsv(self, resize, hue=hsv[0], value=hsv[0], satur=hsv[0])
            unsharp = Update.unsharp_mask(self, chng_hsv)
            contrast = Update.bright_contrast(self, unsharp, contrast=1.5)
            print(f"OCR+rotate+resize+{i}+hsv+{hsv[0]}+{hsv[0]}+{hsv[0]}+unsharp+contrast")
            print(self.ocr_core(contrast))
        '''
        #hsv = (-30, 0, 30)



    def __read_img(self):
        self.orig_img = cv2.imread(self.filePath)
        print('shape orig', self.orig_img.shape)
        # print(self.orig_img.shape)
        # show_img(self.orig_img, 'Original')
        #print('shape orig', self.orig_img.shape)
        #upd = Update.resize_img(self, self.orig_img, cols = self.orig_img.shape[0]*2, rows = self.orig_img.shape[1]*2)
        #print('shape resize', upd.shape)
        '''
        print("colors")
        for i in Update.colors:
            show_img(Update.change_color(self, upd, choice=i), i)

        print("erose")
        for i in Update.boards:
            erose = Update.erosion(self, upd, k = (5, 5), board=i)
            show_img(erose, i)
            print(f"Erose {i}\n", Passport.ocr_core(self, erose))

        print("dilation")
        for i in Update.boards:
            dilate = Update.dilation(self, upd, k = (5, 5), board=i)
            show_img(dilate, i)
            print(f"Dilate {i}\n", Passport.ocr_core(self, dilate))

        print("Morphology")
        for i in Update.ex:
            morph = Update.morphology(self, upd, k = (5, 5), choice=i)
            show_img(morph, i)
            print(f"Morph {i}\n", Passport.ocr_core(self, morph))
            '''
        # show_img(upd, 'Resize')
        #show_img(Update.black_white(self, self.orig_img), 'black')
        #hsv = (-30, 0, 30)
        #change_contours = Update.change_hsv(self, upd, hue=-30, value=-30, satur=-30)
        #show_img(change_contours, 'HSV')
        #self.chng = Update.extract_contours_text(self, change_contours)
        #show_img(self.chng, 'New_image')
        #cv2.imwrite('templates/new.jpg', self.chng)
        '''for h in hsv:
            for s in hsv:
                for v in hsv:
                    show_img(Update.change_hsv(self, upd, hue = h, value = v, satur=s), f'hue={h}, satur={s}, value={v}')
        '''
        return self.orig_img
        # Функция для распознавания слов и контуров с помощью doctr
        #Doctr.find_contours(self, self.filePath)

    def __rotate_img(self):
        image = self.makeCorrectOrientation(self.orig_img) #Rotate
        #self.image = self.chng
        self.filepath_oriented = "templates/Right_corr.jpg"
        cv2.imwrite(self.filepath_oriented, image)
        # print(self.image)
        '''if (image is None):
            return False
        show_img(self.orig_img, 'Original')
        #Doctr.find_contours(self, filepath_oriented)
        print("Original\n", Passport.ocr_core(self, self.orig_img))
        print("Rotated\n", Passport.ocr_core(self, self.image))
        change_contours = Update.change_hsv(self, self.image, hue=-30, value=-30, satur=-30) #HSV
        show_img(change_contours, 'HSV2')
        chng = Update.extract_contours_text(self, change_contours)  # HSV find contours
        show_img(chng, "Rotated+HSV_filter")
        unsharp = Update.unsharp_mask(self, chng)  #Unsharp
        hsv = (-30, 0, 30)'''
        '''for h in hsv:
            for s in hsv:
                for v in hsv:
                    show_img(Update.change_hsv(self, unsharp, hue=h, value=v, satur=s), f'hue={h}, satur={s}, value={v}')
        '''
        #chng = Update.extract_contours_text(self, unsharp)  #HSV-filter
        '''show_img(unsharp, "Rotated+HSV_filter+Unsharp")
        print("Rotated+Unsharp+HSV_filter", Passport.ocr_core(self, chng))   #Detect text
        '''
        #show_img(Update.black_white(self, self.image), 'try black_white')
        #show_img(Update.black_white(self, unsharp), 'try black_white+unsharp')

        '''
        print("origin:")
        print("origImageCut\n", Passport.ocr_core(self, self.orig_img))
        print("unsharp:")
        print("Unsharp\n", Passport.ocr_core(self, Update.black_white(self, unsharp)))'''
        #self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        #show_img(Update.change_hsv(self, self.image, hue = 50, value=50), 'HueSV')
        #show_img(self.hsv, 'HSV')
        # Функции для нахождения ФИО
        #print("New Img:", self.processFullNameInternal(self.image))

        # show_img(self.image, 'Original')
        #print("Orig Img:", self.processFullNameInternal(self.orig_img))
        return image
        # Doctr.find_contours(self, filepath_oriented)
        # return self.processFullNameInternal(self.image)

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

        # show_img(fullNameImage, 'full Name Image')
        fullNameModifiedImage = cv2.cvtColor(fullNameImage, cv2.COLOR_BGR2GRAY)
        fullNameModifiedImage = \
            cv2.threshold(fullNameModifiedImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
        fullNameModifiedImage = cv2.morphologyEx(fullNameModifiedImage, cv2.MORPH_CLOSE, rectKernel)

        imageContours = cv2.findContours(fullNameModifiedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageContours = imutils.grab_contours(imageContours)

        (sortedContours, boundingBoxes) = contours.sort_contours(imageContours, method="top-to-bottom")
        countNameContours = 0

        origImageRatio = origImage.shape[0] / float(self.RESIZED_IMAGE_HEIGHT)
        icon = 0
        for с in sortedContours:
            (x, y, w, h) = cv2.boundingRect(с)

            if w > self.RESIZED_IMAGE_NAME_MIN_WIDTH and \
                    h > self.RESIZED_IMAGE_NAME_MIN_HEIGHT and \
                    h < self.RESIZED_IMAGE_NAME_MAX_HEIGHT and \
                    x > self.RESIZED_IMAGE_NAME_MIN_X and \
                    fullNameImage.shape[1] - x > self.RESIZED_IMAGE_NAME_MIN_RIHGTH_INDENT:
                countNameContours = countNameContours + 1

                origImageCut = origImage[
                               int(((y + fullNameMinY - self.RESULT_IMAGE_NAME_MARGIN) * origImageRatio)):int(
                                   ((y + h + fullNameMinY + self.RESULT_IMAGE_NAME_MARGIN) * origImageRatio)),
                               int(((x + fullNameMinX - self.RESULT_IMAGE_NAME_MARGIN) * origImageRatio)):int(
                                   ((x + w + fullNameMinX + self.RESULT_IMAGE_NAME_MARGIN) * origImageRatio))
                               ]
                # Изменить качество изображений на местах ФИО
                # Это улучшает качество
                unsharp = Update.unsharp_mask(self, origImageCut)
                upd = Update.dilation(self, unsharp, k = (10, 15), board=Update.boards[0])
                upd = Update.erosion(self, upd, k = (3, 3), board=Update.boards[0])
                show_img(upd, 'UPD')
                icon += 1
                if icon == 1:
                    print("origImageCut\n", Search.look_family(self, Passport.ocr_core(self, upd)))
                if icon == 2:
                    print("origImageCut\n", Search.look_name(self, Passport.ocr_core(self, upd)))
                if icon == 3:
                    print("origImageCut\n", Search.look_surname(self, Passport.ocr_core(self, upd)))
                print("origImageCut\n", Passport.ocr_core(self, unsharp))
                show_img(origImageCut, "origImangeCut" + str(countNameContours))
                # upd = Update.sharp_image(self, origImageCut, t = 9)
                # threshs = (cv2.THRESH_BINARY_INV,  cv2.THRESH_TRUNC,  cv2.THRESH_TOZERO,  cv2.THRESH_TOZERO_INV)
                # for i in threshs:
                # show_img(Update.img_threshold(self, origImageCut), "origImangeCutThresh"+str(countNameContours))
                # show_img(Update.fix_pixels(self, origImageCut), "origImage+fix")
                #unsharp = Update.unsharp_mask(self, origImageCut)
                #show_img(Update.unsharp_mask(self, origImageCut), "OrigImageSharp")
                # show_img(())
                # show_img(Update.bilaterial_filter(self, Update.true_sharp(self, origImageCut, origImageCut)), "origImageTrueSharp "+str(countNameContours))
                # show_img(upd, "origImageCut+sharp_1"+str(countNameContours))
                # show_img(Update.gaussian_blur(self, origImageCut), "OrigImageCut+sharp_2" + str(countNameContours))
                # show_img(Update.median_blur(self, upd), "upd+sharp_2"+str(countNameContours))
                # show_img(Update.unsharp_mask(self, origImageCut), "origImageCut+sharp_2"+str(countNameContours))
                filePath = self.getUniqueFilePath()
                cv2.imwrite(filePath, origImageCut)

                if (countNameContours == 1):
                    self.surnameFilePath = filePath
                elif (countNameContours == 2):
                    self.nameFilePath = filePath
                elif (countNameContours == 3):
                    self.patronymicFilePath = filePath

                if (countNameContours == 3):
                    break

        return True if countNameContours == 3 else False

    def getUniqueFilePath(self):
        return self.RESULT_IMAGES_FOLDER + '/' + str(uuid.uuid4()) + '.' + self.RESULT_IMAGES_EXTENSION

    def ocr_core(self, img):
        """This function will handle the core OCR"""

        text = pt.pytesseract.image_to_string(img, lang='rus')
        return text.strip().lower().capitalize()


'''def ocr_core(img):
    """This function will handle the core OCR"""

    text = pt.pytesseract.image_to_string(img, lang='rus')
    return text'''


def captch_ex(file_name):
    img = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    # for cv2.x.x

    print(dilated)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    # for cv3.x.x comment above line and uncomment line below

    # image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    plt.imshow(img)
    plt.title('captcha_result')
    plt.show()
    # cv2.imshow('captcha_result', img)
    # cv2.waitKey()


def transform_image(filename):
    pil_img = Image.open(filename)
    opencv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2RGB)  # np.array(pil_img)
    # opencv_image = opencv_image[:, :, ::-1].copy()
    print(opencv_image)
    plt.imshow(opencv_image)
    plt.title('Image')
    plt.show()

    img = cv2.resize(opencv_image,
                     (opencv_image.shape[1] * 2, opencv_image.shape[0] * 2))  # Увеличить для лучшего результата
    # img = cv2.Canny(img, 90, 90)
    # plt.imshow(img)
    # plt.title('Image')
    # plt.show()
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary_image = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    plt.imshow(binary_image)
    plt.title('Image')
    plt.show()
    # cv2.imshow('Result', binary_image)
    # cv2.waitKey(0)

    inverted_bin = cv2.bitwise_not(binary_image)

    # Some noise reduction
    kernel = np.ones((2, 2), np.uint8)
    processed_img = cv2.erode(inverted_bin, kernel, iterations=1)
    processed_img = cv2.dilate(processed_img, kernel, iterations=1)

    plt.imshow(processed_img)
    plt.title('Image')
    plt.show()

    # cv2.imshow('Result', processed_img)
    # cv2.waitKey(0)

    return processed_img
