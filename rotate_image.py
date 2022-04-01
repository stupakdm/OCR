import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from Update_photo import Update
from imutils import contours



class New_Rotate(Update):

    def largest_rotated_vec(self):
        pass

    def rotate_image(self, image, angle):
        (h,w) = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def crop_image(self, img, degree):
        pass

    def skew_correction(self, image):
        bl_wh = Update.black_white(self, image)


        thresh = cv2.threshold(bl_wh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh>0))
        #print(coords)
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90+angle)

        else:
            angle = -angle

        return angle

    def left_red_contours(self, image):
        pass
        #gray = cv2.bitwise_not(image)
        #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


    def cvt_image(self, image):
        hsv = Update.change_hsv(self, image, hue=-30, value=-30, satur=-30)
        return Update.extract_red_contours(self, hsv)

    def rotate(self, orig_image):
        bl_wh = self.cvt_image(orig_image)
        dilate = Update.dilation(self, bl_wh)
        #show_img(dilate, 'Dilate')
        erose = Update.erosion(self, bl_wh, k=(6,6), board='REPLICATE')
        #show_img(erose, 'black_white')
        angle = self.skew_correction(bl_wh)
        rotated = self.rotate_image(orig_image, angle)
        return rotated


# Получение только красной части на изображении
def getRedImage(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    redRange1 = cv2.inRange(hsvImage, np.array([0, 70, 50]), np.array([10, 255, 255]))
    redRange2 = cv2.inRange(hsvImage, np.array([170, 70, 50]), np.array([180, 255, 255]))
    redRange3 = cv2.inRange(hsvImage, np.array([160, 100, 100]), np.array([179, 255, 255]))

    redImage = cv2.bitwise_or(redRange1, redRange2)
    redImage = cv2.bitwise_or(redImage, redRange3)
    # print("redImage: ", end="")
    # print(redImage)
    return redImage


class Base_Coef:
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
    RESULT_IMAGES_FOLDER = '/templates'

    def rotate_image(self, image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    def largest_rotated_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    # Обрезать фото по бокам
    def crop_around_center(self, image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if width > image_size[0]:
            width = image_size[0]

        if height > image_size[1]:
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]


class FindCountours(Base_Coef):

    # Нахождение красной линии посередине паспорта.
    # Если линия найдена, возвращается информация о ней в виде {x,y,w,h}, иначе None.
    def getLineSeparatorInfo(self, resizedImage):
        redImage = getRedImage(resizedImage)

        widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
        redModifiedImage = cv2.dilate(redImage, widthKernel)

        heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
        redModifiedImage = cv2.dilate(redModifiedImage, heightKernel)

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

    def redImage(self, img):
        redImage = getRedImage(img)

        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        redModifiedImage = cv2.dilate(redImage, squareKernel)

        imageContours = cv2.findContours(redModifiedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageContours = imutils.grab_contours(imageContours)

        return imageContours

    ''''# Получение только красной части на изображении
    def getRedImage(self, image):
            hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            redRange1 = cv2.inRange(hsvImage, np.array([0, 70, 50]), np.array([10, 255, 255]))
            redRange2 = cv2.inRange(hsvImage, np.array([170, 70, 50]), np.array([180, 255, 255]))
            redRange3 = cv2.inRange(hsvImage, np.array([160, 100, 100]), np.array([179, 255, 255]))

            redImage = cv2.bitwise_or(redRange1, redRange2)
            redImage = cv2.bitwise_or(redImage, redRange3)
            # print("redImage: ", end="")
            # print(redImage)
            return redImage'''


class RotateImage(FindCountours):
    # Минимальное соотношение ширины к высоте рамки для фото.

    def makeCorrectOrientation(self, origImage):
        resizedImage = imutils.resize(origImage.copy(), height=self.RESIZED_IMAGE_HEIGHT)
        # show_img(resizedImage, )

        for degree in [0, 45, 90, 135, 180, 225, 270, 315]:
            rotatedImage = imutils.rotate_bound(resizedImage, degree)

            k = self.isRightOrientation(rotatedImage)
            if type(k) != bool:
                return k

        return None

    def isRightOrientation(self, image):
        lineSeparatorInfo = self.getLineSeparatorInfo(image)
        if (lineSeparatorInfo is None):
            return False
        print('ok1')
        lineSeparatorRatio = lineSeparatorInfo['w'] / float(lineSeparatorInfo['h'])
        #if (lineSeparatorRatio < self.LINE_SEPARATOR_MIN_RATIO):
         #   return False
        print('ok2')
        photoInfo = self.getPhotoInfo(image)
        if (photoInfo is None):
            return False
        print('ok3')
        # Рамка для фото должна быть ниже красной линии.
        if photoInfo['minY'] <= lineSeparatorInfo['y']:
            return False
        print('ok4')
        img = image[photoInfo['minY']:photoInfo['maxY'], photoInfo['minX']:photoInfo['maxX'], :]
        crop_img = self.crop_img(image, self.rotation(img))
        return crop_img
        #return True if photoInfo['minY'] > lineSeparatorInfo['y'] else False

    # Если линия найдена, возвращается информация о ней в виде {x,y,w,h}, иначе None.
    def getLineSeparatorInfo(self, resizedImage):
        redImage = getRedImage(resizedImage)

        widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
        redModifiedImage = cv2.dilate(redImage, widthKernel)

        heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
        redModifiedImage = cv2.dilate(redModifiedImage, heightKernel)

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

    def rotation(self, img):
        delta = 2
        rows, cols, ch = img.shape
        height, width = rows, cols
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 0, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        # rotatedImage = imutils.rotate_bound(img, 0)
        beg_left = self.find_left_angle(dst)
        degree_1 = 0
        t = 1
        while degree_1 > -90:
            M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree_1, 1)
            dst = cv2.warpAffine(img, M, (cols, rows))
            # rotatedImage = imutils.rotate_bound(img, degree)
            left = self.find_left_angle(dst)

            # show_img(dst, 'Degree ' +str(degree_1))
            if sum(left) <= sum(beg_left):
                beg_left = left
            else:
                degree_1 += t
                # show_img(dst, 'Rotated degree1 ' + str(degree_1))
                break
            degree_1 -= t
        degree_2 = 0
        t = 1
        while degree_2 < 90:
            M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree_2, 1)
            dst = cv2.warpAffine(img, M, (cols, rows))
            # rotatedImage = imutils.rotate_bound(img, degree)
            left = self.find_left_angle(dst)

            # show_img(dst, 'Degree ' + str(degree_2))
            if sum(left) <= sum(beg_left):
                beg_left = left
            else:
                degree_2 -= t
                # show_img(dst, 'Rotated degree2 ' + str(degree_2))
                # show_img(image_rotated_cropped, 'Cropped and rotated ' + str(degree_2))
                break
            degree_2 += t
        # if (left[0] <= delta and left[1] <= delta):
        # show_img(rotatedImage, 'Rotated')

        if degree_2 < degree_1:
            return degree_2
        else:
            return degree_1
        # return degree

    def find_left_angle(self, img):
        imageCountours = self.redImage(img)
        left_x = -1
        left_y = -1
        fl = 0
        print("IMAGECONTOURS")
        # print(len(imageCountours))
        for c in imageCountours:
            # print(c)
            print(len(c))

            for i in range(len(c)):
                if fl == 0:
                    fl = 1
                    left_x = c[i][0][1]
                    left_y = c[i][0][0]
                    # print(left_y)
                    # print(left_x)
                for j in range(len(c[i])):
                    if np.sum(c[i][j]) < left_y + left_x:
                        left_x = c[i][j][1]
                        left_y = c[i][j][0]
                        '''
                    if c[i][j][1] < left_x:
                        left_x = c[i][j][1]
                        left_y = c[i][j][0]
                    elif c[i][j][1] == left_x and c[i][j][0] < left_y:
                        left_y = c[i][j][0]
                        left_x = c[i][j][1]'''
        print(left_x, left_y)
        return (left_x, left_y)

    def crop_img(self, img, degree):
        # degree = self.rotation(img)
        # print("Degree:", degree)
        print("Degree: ", degree)
        rows, cols, ch = img.shape
        height, width = rows, cols
        show_img(img, 'Origin')
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        show_img(dst, 'Rotated full img')

        image_rotated_cropped = self.crop_around_center(
            dst,
            *self.largest_rotated_rect(
                width,
                height,
                math.radians(degree)
            )
        )
        show_img(image_rotated_cropped, 'Rotated and croped')
        return image_rotated_cropped

    def getPhotoInfo(self, resizedImage):
        redImage = getRedImage(resizedImage)

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
        # print(resizedImage.shape)

        '''
        degree = self.rotation(img)
        print("Degree:", degree)

        rows, cols, ch = orig.shape
        height, width = rows, cols
        show_img(orig, 'Origin')
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree, 1)
        dst = cv2.warpAffine(orig, M, (cols, rows))
        show_img(dst, 'Rotated full img')

        image_rotated_cropped = self.crop_around_center(
            dst,
            *self.largest_rotated_rect(
                width,
                height,
                math.radians(degree)
            )
        )
        show_img(image_rotated_cropped, 'Rotated and cropped')
        # show_img(img, "resized")
        # print("Contours", imageContours)
        # cv2.imshow(resizedImage)
        # cv2.waitKey(0)
        # for degree in range(0, 90):
        #    rotatedImage = imutils.rotate_bound(img, degree)
        # if (self.isRightOrientation(rotatedImage)):
        #    return imutils.rotate_bound(origImage, degree)
        '''
        print("photoInfo:", end="")
        print(photoInfo)
        return photoInfo


def show_img(img, title):
    # print(img.shape)
    # print(img)
    plt.imshow(img)
    plt.title(title)
    plt.show()


a = '''
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def demo():
    """
    Demos the largest_rotated_rect function
    """

    image = cv2.imread("templates/1.jpeg")
    image_height, image_width = image.shape[0:2]

    show_img(image, "Original Image")

    print("Press [enter] to begin the demo")
    print("Press [q] or Escape to quit")


    for i in np.arange(0, 360, 0.5):
        image_orig = np.copy(image)
        image_rotated = rotate_image(image, i)
        image_rotated_cropped = crop_around_center(
            image_rotated,
            *largest_rotated_rect(
                image_width,
                image_height,
                math.radians(i)
            )
        )

        show_img(image_orig, "Original Image")
        show_img(image_rotated, "Rotated Image")
        show_img(image_rotated_cropped, "Cropped Image")

    print("Done")


if __name__ == "__main__":
    demo()'''
