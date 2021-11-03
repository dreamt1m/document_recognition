import re
import sys
import os

import cv2
import numpy as np
import pytesseract

DEFRAGMENTATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (300, 1))
SPLIT_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 75))
EXTRACT_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 75))


def get_contours(image, kernel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_fragments(image, kernel):
    images = []
    for cnt in get_contours(image, kernel):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100:
            continue
        cropped = image[y:y + h, x:x + w]
        if len(get_contours(cropped, SPLIT_KERNEL)) > 2:
            images.extend(get_fragments(cropped, SPLIT_KERNEL))
        else:
            images.append(cropped.copy())
    return list(reversed(images))


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def delete_spaces(s):
    return re.sub(r"\s*\n\s*", ' ', s.strip())


def extract_string(image, thresh=90):
    cnt = get_contours(image, EXTRACT_KERNEL)[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = image[y:y + h, x:x + w]
    cropped = cv2.resize(cropped, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    cropped = unsharp_mask(cropped, kernel_size=(7, 7))
    _, cropped = cv2.threshold(cropped, thresh, 255, cv2.THRESH_BINARY)
    info = pytesseract.image_to_string(cropped, lang='rus', config='--oem 1')
    info = ''.join((filter(lambda a: a not in ['|', '_'], info)))
    info = delete_spaces(info)
    return info if info != '' else ' '


def extract_number(image, thresh=188):
    cnt = get_contours(image, EXTRACT_KERNEL)[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = image[y:y + h, x:x + w]
    cropped = cv2.resize(cropped, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    cropped = unsharp_mask(cropped, kernel_size=(5, 3))
    _, cropped = cv2.threshold(cropped, thresh, 255, cv2.THRESH_BINARY)
    info = delete_spaces(
        pytesseract.image_to_string(cropped, config='--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789'))
    return info if info != '' else ' '


def form_json(images):
    dct = dict()
    dct['Выписка из протокола'] = extract_string(images[0])
    dct['дата'] = extract_string(images[3])
    dct['номер'] = str(extract_number(images[2]))
    dct['пункт'] = str(extract_number(images[1]))
    dct['Наименование объекта'] = extract_string(images[4])
    dct['Авторы проекта'] = extract_string(images[5], 70)
    dct['Генеральная проектная организация'] = extract_string(images[6])
    dct['Застройщик'] = extract_string(images[7])
    dct['Рассмотрение на рабочей комиссии'] = ' '
    dct['Референт'] = extract_string(images[9])
    dct['Докладчик'] = extract_string(images[10])
    dct['Выступили'] = extract_string(images[11])
    return str(dct).replace('\',', '\',\n').replace('}', '\n}').replace('\"', '\\\"').replace('\'', '\"')


def main(path = ""):
    if path != "":
        image = cv2.imread(path)
        image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        images = get_fragments(image, DEFRAGMENTATION_KERNEL)
        file = open(f"{path}.json", "w+")
        file.write(form_json(images))
        file.close()
        print(f"Result moved to {path}.json")
    else:
        path = "cv"
        file = open("result.json", "w+")
        file.write("{\n")

        jsons = []
        files = os.listdir(path)
        files.sort()
        jpgs = filter(lambda x: x.endswith('.jpg'), files)
        for img in jpgs:
            img = cv2.imread(f"{path}/{img}")
            img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
            imgs = get_fragments(img, DEFRAGMENTATION_KERNEL)            
            jsons.append(form_json(imgs))
        file.write(",\n".join(jsons))
        file.write("\n}")
        file.close()
        print("Done!")



if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
