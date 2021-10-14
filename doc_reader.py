from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import json


document_columns_dict = {
    "Выписка из протокола": True,
    "дата": True, "номер": True, "пункт": True,
    "Наименование объекта": True,
    "Авторы проекта": True,
    "Генеральная проектная организация": True,
    "Застройщик": True,
    "Рассмотрение на рабочей комиссии": False,
    "Референт": True,
    "Докладчик": True,
    "Выступили": True
}


def skew_correction(image_name):
    '''Поворот изображения к вертикали.

    args: название изображения 
    return: массив пикселей изображения
    '''
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    coords = np.column_stack(np.where(img_bin == 255))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle == 90:
        angle = 0
    elif 45 < angle < 90:
        angle = 90 - angle
    else:
        angle = -angle

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def find_main_lines(img, type):
    '''Поиск линий рамки. 

    args: img = массив пикселей изображения, type = тип линий для поиска 
    return: массив линий.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]

    if type == 'h':
        structuring_element = np.ones((1, 30), np.uint8)
    elif type == 'v':
        structuring_element = np.ones((25, 1), np.uint8)

    erode_image = cv2.erode(gray, structuring_element, iterations=1)
    lines = cv2.dilate(erode_image, structuring_element, iterations=1)

    return lines


def merge_lines(horizontal_lines, vertical_lines):
    '''Соединение вертикальных и горизонтальных линий для получения . 
    
    args: horizontal_lines = массив горизонтальных линий, vertical_lines = массив вертикальных линий 
    return: массив пикселей изображения(контуры рамок).
    '''
    structuring_element = np.ones((1, 1), np.uint8)
    merge_image = horizontal_lines + vertical_lines
    merge_image = cv2.dilate(merge_image, structuring_element, iterations=2)

    return merge_image


def read_img_from_border(cnts, img, united_image):
    '''Соединение вертикальных и горизонтальных линий. 
    
    args: cnts = контуры рамок, img = исходное изображение, united_image = изображение с определенными рамками 
    return: текст считаный с рамок
    '''
    temp_img_name = 'temp_img.png'
    data = []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        img_crop = find_cell_contours(img, united_image, x, y, w, h)

        cv2.imwrite(temp_img_name, img_crop)
        text = pytesseract.image_to_string(
            Image.open(r'{}'.format(temp_img_name)), lang='rus')
        os.remove(temp_img_name)

        data.append(text.strip())

    return data


def find_cell_contours(img, united_image, x, y, w, h):
    '''Удаление рамки с вырезанного изображения. 
    
    args: img = исхожное изображение, united_image = изображение с определенными рамками, x, y, w, h
    return: изображение без рамки
    '''
    frame_image = img[y:y + h, x:x + w]
    crop_image = united_image[y:y + h, x:x + w]



    white_pixels = np.where(frame_image == 255)
    y = white_pixels[0]
    x = white_pixels[1]
    for i in range(len(y)):
        crop_image[y[i]][x[i]] = 255

    return crop_image


def match_dict_to_list(dict, list):
    '''Сопоставление dict и list
    
    args: dict, list
    return: считаные данные из изображения в формате json.
    '''
    list.reverse()
    i = 0

    for k in dict:
        if (dict[k]):
            dict[k] = list[i]
            i += 1
        else:
            dict[k] = ' '

    data_json = json.dumps(dict, ensure_ascii=False)

    return data_json


if __name__ == '__main__':
    img_name = '1.jpg'
    img = skew_correction(img_name)
    horizontal_lines = find_main_lines(img, 'h')
    vertical_lines = find_main_lines(img, 'v')
    united_image = merge_lines(horizontal_lines, vertical_lines)
    cnts, _ = cv2.findContours(
        united_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    data = read_img_from_border(cnts, img, united_image)
    data_json = match_dict_to_list(document_columns_dict, data)
    print(data_json)
