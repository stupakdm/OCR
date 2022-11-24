import cv2
import numpy as np
import imutils
import math
from Diplom.updating_diplom.optimize_diplom.image_updating_dp import show_img

class Rotate:
    def rotate_image_true(self, image, angle):

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

    #Обрезать фото по бокам

    def crop_around_center(self, image, width, height):

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

    def try_find_new_degree(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cont in contours:
            rect = cv2.minAreaRect(cont)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #center = (int(rect[0][0]), int(rect[0][1]))
            #area = int(rect[1][0] * rect[1][1])

            edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
            edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

            print('Edges', edge1, edge2)
            usedEdge = edge2
            reference = (1, 0)


            angle = 180.0 / math.pi * math.acos((reference[0] * usedEdge[0] + reference[1] * usedEdge[1])
                                                / (cv2.norm(reference) * cv2.norm(usedEdge)))
            if angle > 60:
                angle = angle - 90
            if angle < -60:
                angle = 90 - angle
            return angle

    def find_biggest_rect(self, image):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_min = np.array((50, 50, 20), np.uint8)
        rgb_max = np.array((120, 100, 100), np.uint8)
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #hsv_min = np.array((107, 0, 130), np.uint8)
        #hsv_max = np.array((110, 255, 190), np.uint8)
        thresh = cv2.inRange(image, rgb_min, rgb_max)
        #thresh = cv2.inRange(hsv, hsv_min, hsv_max)
        #H, W = thresh.shape[:2]

        #if np.sum(thresh)/255 > H*W/15:
        #    heightKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
        #    thresh = cv2.erode(thresh, kernel=heightKernel, borderType=cv2.RETR_EXTERNAL)

        widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 25))
        #for i in range(10):
        #    widthKernel[i] = np.array([0]*20)
        #thresh = cv2.morphologyEx(thresh, cv2., widthKernel)

        blueModifiedImage = cv2.dilate(thresh, widthKernel, iterations=2, borderType=cv2.RETR_EXTERNAL)
        #blueModifiedImage = cv2.morphologyEx(blueModifiedImage, cv2.MORPH_CLOSE, widthKernel)

        #show_img(blueModifiedImage, 'blueModifiedImage')
        imageContours = cv2.findContours(blueModifiedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageContours = imutils.grab_contours(imageContours)

        lineContourInfo = None
        contourMaxWidth = 0
        max_cont = None
        if len(imageContours) !=0:
            max_cont = imageContours[0]
        for c in imageContours:
            (x, y, w, h) = cv2.boundingRect(c)
            #rect = cv2.minAreaRect(c)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            # cv2.drawContours(blueModifiedImage, [box], 0, 255,2)
            if (w > contourMaxWidth):
                max_cont = c
                contourMaxWidth = w
                lineContourInfo = {'x': x, 'y': y, 'w': w, 'h': h}
        #(x, y, w, h) = cv2.boundingRect(max_cont)
        rect = cv2.minAreaRect(max_cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_image[:,:] = [0]
        cv2.drawContours(new_image, [box], 0, 255, 3)
        #cv2.drawContours(new_image, (lineContourInfo['x'], lineContourInfo['y']), )
        #cv2.rectangle(new_image, (lineContourInfo['x'], lineContourInfo['y']),
        #              (lineContourInfo['x'] + lineContourInfo['w'], lineContourInfo['y'] + lineContourInfo['h']), 255, thickness=3)
        #show_img(new_image, 'Rectangles')
        print('lineContourInfo', lineContourInfo)
        return new_image

    def largest_rotated_rect(self, w, h, angle):


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


    def cropped_rotated_image(self, orig_img, degree):
        rows, cols, ch = orig_img.shape
        height, width = rows, cols

        #show_img(orig_img, 'Origin')


        rotated = self.rotate_image_true(orig_img, degree)#(-1)*degree)
        #show_img(rotated, 'Rotated full img')

        image_rotated_cropped = self.crop_around_center(
            rotated,
            *self.largest_rotated_rect(
                width,
                height,
                math.radians(degree)
            )
        )

        return image_rotated_cropped

    '''def find_avg_color(self, image):
        avr_color_row = np.average(image, axis = 0)
        average_color = np.average(avr_color_row, axis=0)
        print("avg_color: ", average_color)
        return average_color'''

    def rotate(self, image_to_rotate, orig_image, key=1):
        RESIZED_IMAGE_HEIGHT = 600
        # orig_image = imutils.resize(orig_image.copy(), height=RESIZED_IMAGE_HEIGHT, width=int(w*RESIZED_IMAGE_HEIGHT/h))
        image_to_rotate = imutils.resize(image_to_rotate.copy(), height=RESIZED_IMAGE_HEIGHT)
        #self.find_avg_color(orig_image)
        #self.find_avg_color(image_to_rotate)
        orig_image = imutils.resize(orig_image.copy(), height=RESIZED_IMAGE_HEIGHT)
        original = orig_image.copy()
        rect = self.find_biggest_rect(image_to_rotate)
        #print('rect:', rect)
        degree = self.try_find_new_degree(rect)
        print('Degree: ', degree)
        if degree == 0:
            return orig_image
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        cropped_rotated = self.cropped_rotated_image(original, degree)
        return cropped_rotated