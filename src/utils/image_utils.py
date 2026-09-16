import cv2


def resize_image(image, size=(224, 224)):
    return cv2.resize(image, size)


def save_image(path, image):
    cv2.imwrite(path, image)


def load_image(path):
    return cv2.imread(path)