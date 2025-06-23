import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def warp_perspective(img, keypoints):
    # tl, tr, br, bl = keypoints
    tl = tuple(map(int, keypoints[0]))
    bl = tuple(map(int, keypoints[3]))
    tr = tuple(map(int, keypoints[1]))
    br = tuple(map(int, keypoints[2]))
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, img.shape[0]], [img.shape[1], 0], [img.shape[1], img.shape[0]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    return warped

def get_text(img) -> tuple:
    """
    :param img: np.ndarray
    :return: tuple: text and confidence
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PaddleOCR
    img = cv2.resize(img, (640, 480))  # Resize image to
    results = ocr.ocr(img, cls=True)
    if results[0]:
        if len(results[0]) == 1: # If there's only 1 row of text then return it
            return results[0][0][1]
        elif len(results[0]) == 2:
            """
                - If there are 2 rows of text then we need to check which one is the first row, which is the second row 
                - The first row contains fewer characters than the second row or the first row contains 1 alphabet 
                  character, the second row contains only digits.
             """
            text1, conf1 = results[0][0][1]
            text2, conf2 = results[0][1][1]

            if len(text1) < len(text2) or any(char.isalpha() for char in text1):
                return text1 + text2, (conf1 + conf2) / 2
            else:
                return text2 + text1, (conf1 + conf2) / 2

    return None, None

