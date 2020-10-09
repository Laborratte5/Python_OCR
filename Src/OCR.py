from PIL import Image
import pytesseract
import imutils
from skimage.filters import threshold_local
from rake_nltk import Rake

import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\User\AppData\Local\Tesseract-OCR\tesseract.exe'


def get_text_from_img(path, lng):
    img = scan_img(path)
    text = ocr(img, lng)
    text = '''Donald J. Trump is the 45th President of the United States. He believes the United States has incredible potential and will go on to exceed even its remarkable achievements of the past.
Donald J. Trump defines the American success story. Throughout his life he has continually set the standards of business and entrepreneurial excellence, especially in real estate, sports, and entertainment. Mr. Trump built on his success in private life when he entered into politics and public service. He remarkably won the Presidency in his first ever run for any political office.

A graduate of the University of Pennsylvania’s Wharton School of Finance, Mr. Trump followed in his father’s footsteps into the world of real estate development, making his mark in New York City. There, the Trump name soon became synonymous with the most prestigious of addresses in Manhattan and, subsequently, throughout the world.

Mr. Trump is also an accomplished author. He has written more than fourteen bestsellers.  His first book, The Art of the Deal, is considered a business classic.

Mr. Trump announced his candidacy for the Presidency on June 16, 2015. He then accepted the Republican nomination for President of the United States in July of 2016, having defeated 17 other contenders during the Republican primaries.

On November 8, 2016, Mr. Trump was elected President in the largest Electoral College landslide for a Republican in 28 years. Mr. Trump won more than 2,600 counties nationwide, the most since President Ronald Reagan in 1984. He received the votes of more than 62 million Americans, the most ever for a Republican candidate.

President Trump has delivered historic results in his first term in office despite partisan gridlock in the Nation’s Capital, and resistance from special interests and the Washington Establishment.

He passed record-setting tax cuts and regulation cuts, achieved energy independence, replaced NAFTA with the United-States-Mexico-Canada Agreement, invested $2 trillion to completely rebuild the Military, launched the Space Force, obliterated the ISIS Caliphate, achieved a major breakthrough for peace in the Middle East, passed the most significant Veterans Affairs reforms in half a century, confirmed over 250 federal judges, including 2 Supreme Court Justices, signed bipartisan Criminal Justice Reform, lowered drug prices, protected Medicare and Social Security, and secured our nation’s borders.

To vanquish the COVID-19 global pandemic, President Trump launched the greatest national mobilization since World War II. The Trump Administration enacted the largest package of financial relief in American history, created the most advanced testing system in the world, developed effective medical treatments to save millions of lives, and launched Operation Warp Speed to deliver a vaccine in record time and defeat the Virus.

President Trump has been married to his wife, Melania, for 15 years, and they are parents to their son, Barron. Mr. Trump also has four adult children, Don Jr., Ivanka, Eric, and Tiffany, as well as 10 grandchildren.'''
    rake = Rake()
    rake.extract_keywords_from_text(text)
    #gprint(rake.get_ranked_phrases())
    ranked_phrases = rake.get_ranked_phrases_with_scores()

    text = ""
    print(ranked_phrases)
    for i in range(len(ranked_phrases) // 2):
        text += "'" + ranked_phrases[i][1] + "',"
    #text = "".join(s for v, s in ranked_phrases if v >= median)

    #text = rake.get_ranked_phrases_with_scores()

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

    return text


# See https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/

# Transform and crop image
def scan_img(file):
    # Edge detection
    # crop image
    image = cv2.imread(file)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # grayscale convertion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection using Canny algorithm
    edged = cv2.Canny(gray, 75, 200)

    # Find contours
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    screenCnt = None
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # show the contour (outline) of the piece of paper
    if screenCnt is not None:
        #cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        # apply the four point transform to obtain a top-down
        # view of the original image
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # TODO threshold_local for better color correction
        T = threshold_local(warped, 11, offset=10, method="gaussian")
        warped = (warped > T).astype("uint8") * 255
        # cv2.imshow("warped", warped)
        # cv2.waitKey()
        return warped

    return orig


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect
