
import Defines
import pytesseract

from Scan import scan_img

pytesseract.pytesseract.tesseract_cmd = Defines.tesseract_path

def get_text_from_img(path, lng):
    img = scan_img(path)
    text = ocr(img, lng)
    
    return text

def ocr(path, lng):
    #img = Image.open(path)
    img = path
    # norm_img = np.zeros((img.shape[0], img.shape[1]))
    # img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    # img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    # img = cv2.GaussianBlur(img, (1, 1), 0)

    # select language eng+deu
    # select oem --oem 1 for LSTM --oem 0 for Legacy Tesseract
    # select psm 3

    text = pytesseract.image_to_string(img, lng)

    # TODO remove
    text = text.replace('', '')

    return text

