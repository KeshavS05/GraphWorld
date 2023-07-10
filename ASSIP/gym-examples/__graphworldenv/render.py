import cv2
import numpy as np

def draw_grid(img, scale_length, scale_height, line_color=(255, 255, 255), thickness=1, type_=cv2.LINE_AA):
    '''(ndarray, 3-tuple, int, int) -> void'''
    pxstep = scale_length
    pystep = scale_height
    x = scale_length
    y = scale_height
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pystep

def render(length, height):
    image_size = 800

    if image_size / length == int(image_size/length):
        scale_length = int(image_size / length)
    else:
        scale_length = int(image_size / length) + 1

    if image_size / height == int(image_size/height):
        scale_height = int(image_size / height)
    else:
        scale_height = int(image_size / height) + 1       

    img = np.zeros((length * scale_length, height * scale_height, 3), np.uint8)
    draw_grid(img, scale_length, scale_height)
    cv2.imwrite('gridTest.jpg', img)
