import pytesseract as pt
#from paddleocr import PaddleOCR
"""
class Paddle:
    '''
    Paddle - новая китайская нейронная сеть для распознавания, нахождения текстовых блоков
    '''

    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ru')


    def detect_text(self, img):
        result = self.ocr.ocr(img, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)
        return result"""

class Tesseract:
    '''
    Tesseract - нейронная сеть для распознавания слов по текстовым блокам
    '''
    def __init__(self):
        pass

    def detect_text(self, img, params=1):
        """This function will handle the core OCR"""
        # rus = 'абвгдеёжзийклмнопрстуфхцчшщЪыьэюя'
        if params == 1:
            text = pt.pytesseract.image_to_string(img, lang='rus')
        else:
            text = pt.pytesseract.image_to_string(img, config='-l rus --psm 5 --oem 1 --dpi 300')
        # text = pt.pytesseract.image_to_string(img, lang='rus', config = f'--psm 12 --oem 3 -c tessedit_char_whitelist={rus}{rus.upper()}ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return text.strip().lower().capitalize()