from passport.modify_passport.image_updating_passport.red_contour_ps import Passport1
from passport.modify_passport.image_updating_passport.Rotating_ps import New_Rotate
from passport.modify_passport.image_updating_passport.updating_image_ps import Update

from cv2 import imread
import pytesseract as pt


class Info_functions:
    '''
    Класс функций, применяющиеся для get_all_info_ps

    '''
    def __init__(self):
        self.passport_rotate1 = Passport1()
        self.rotate_class = New_Rotate()
        self.update = Update()

    # 0.5 Cropp
    def cropp(self, img):
        cropped = self.update.corner_images(img)
        return cropped


    #     Check orientation
    def check_orientation(self, img):
        check_image, check_flag = self.passport_rotate1.makeCorrectOrientation(img)
        if not check_flag:
            print('Not found')
        else:
            print('Found')
        return check_image, check_flag

        # 1 Rotate
    def rotation(self, img, key=1):
        # key = 2
        rotated = self.rotate_class.rotate(img, key=key)
        return rotated

        # 1.5 Rotate for boxes
    def rotation_boxes(self, img):
        new_img = self.update.box_update(img)

        rotated = self.rotate_class.another_rotate(img, new_img)
        return rotated

    def preProcess_img(self, img, key=1):
        '''
        Улучшение качества изображения

        :param img:
        :param key:
        :return:
        '''
        # 2.1 DPI

        img = self.update.change_dpi(img.copy(), dpi=450)

        img = self.update.change_dpi(img.copy(), dpi=600)

        # 2.2 Unsharp
        img = self.update.unsharp_mask(img.copy(), sigma=1.5, threshold=1, amount=2.0,
                                                      kernel_size=(3, 3))


        # 2.5 Contrast   (Нужно подобрать пар-ры)
        img = self.update.bright_contrast(img.copy(), contrast=1.1, bright=0)


        return img


    def upgrade_image(self, path):
        orig_image = self.read_img(path)
        img = self.cropp(orig_image.copy())

        check_image, check_flag = self.check_orientation(orig_image)

        img = self.rotation(img, key=1)

        try:
            img = self.preProcess_img(img)
        except:
            pass

        return check_image, check_flag, img


    def divide_on_words(self, text):
        text = text.split('\n')
        return list(map(str.strip, text))


    def read_img(self, filepath):
        orig_img = imread(filepath)
        print('shape orig', orig_img)
        return orig_img


    def clean_text(self, text):
        j = 0
        length = len(text)
        while j < length:
            if '<' in text[j] or text[j] == '':
                del text[j]
                length = len(text)
            else:
                j += 1
        return text


class Pytesseract:

    def __init__(self):
        pass

    def ocr_core_text(self, img, params=1):
        """This function will handle the core OCR"""
        # rus = 'абвгдеёжзийклмнопрстуфхцчшщЪыьэюя'
        if params == 1:
            text = pt.pytesseract.image_to_string(img, lang='rus')
        else:
            text = pt.pytesseract.image_to_string(img, config='-l rus --psm 5 --oem 1 --dpi 300')
        # text = pt.pytesseract.image_to_string(img, lang='rus', config = f'--psm 12 --oem 3 -c tessedit_char_whitelist={rus}{rus.upper()}ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return text.strip().lower().capitalize()

    def ocr_core_boxes(self, img):
        boxes = pt.pytesseract.image_to_boxes(img)
        return boxes