import cv2
import numpy as np

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    # ปรับความสว่างและความคมชัด
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def preprocess_image(image):
    # การประมวลผลภาพเบื้องต้น เช่น การแปลงภาพเป็นขาวดำ หรือกรองขอบ
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)

