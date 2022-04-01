try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract as pt

def ocr_core(img):
    """This function will handle the core OCR"""

    for i in img:
        text = pt.pytesseract.image_to_string(i, lang='rus')
        yield text
