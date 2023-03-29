# extract.py
# Chet Russell
# Based on code written by Haniye Kashgarani
# Last edited: Mar 29, 2023

import PIL.ImageOps
from PIL.ExifTags import TAGS
import cv2
import os
import pandas as pd
import time
import pytesseract
import PIL.ExifTags
from PIL.ExifTags import TAGS
from pathlib import Path
from collections import defaultdict
from pprint import pprint

def crop(src_dir, dst_dir):
    #src_dir= "./Garrett_CamWTP07"
    #dst_dir= "./garret"

    allfiles=[]

    for root, dirs, files in os.walk(src_dir):
        print("1")
        for f in files:
            print(f)
            if f.endswith(".JPG") or f.endswith(".jpg") or f.endswith(".jpeg"):
                #print(f)
                allfiles.append(os.path.join(src_dir,f))
                img = cv2.imread(os.path.join(src_dir,f),cv2.IMREAD_UNCHANGED)
                #print('Original Dimensions : ',img.shape)
                img = cv2.resize(img, (422,237))
                status = cv2.imwrite(os.path.join(dst_dir,f),img)
                #print("Image written to file-system : "+os.path.join(dst_dir,f),status,'\n')

def meta_data(src_dir, dst_dir):

    # Grabs temperature
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".JPG") or f.endswith(".jpg") or f.endswith(".jpeg"):
                # crop image and save to destination directory
                img = cv2.imread(os.path.join(src_dir,f),cv2.IMREAD_UNCHANGED)
                #img = img[:30, -200:]
                img = img[2272:, 1360:1650]
                scale_percent = 50
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                status = cv2.imwrite(os.path.join(dst_dir,f),img)
                print("Image written to file-system : "+os.path.join(dst_dir,f),status,'\n')

    # keys = ['Name', 'Date', 'Time', 'Temp']
    images = defaultdict(list)
    # images = dict.fromkeys(keys, [])

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    # for root, dirs, files in os.walk(dst_dir):
    for f in src_dir.glob('*.[jJ][pP][gG]'):
        # images['Name'].append(f.name)
        img = PIL.Image.open(f.absolute())
        exif_table = {TAGS.get(k) : v for k, v in img.getexif().items()}
        creation_date = exif_table["DateTime"]
        date, time = creation_date.split()
        temp_img_path = dst_dir / f.name
        temp_img = PIL.Image.open(temp_img_path)

        custom_config = r'-l eng --psm 13 --oem 0'
        inv_img = PIL.ImageOps.invert(temp_img)
        temp = pytesseract.image_to_string(inv_img, config=custom_config).strip()

        images["Name"].append(f.name)
        images["Date"].append(date)
        images["Time"].append(time)
        images["Temp"].append(temp)
        #images["Animal"].append('')

        temp_img.close()
        img.close()

    pprint(images)

    return images

    #df = pd.DataFrame(images)
    ##, columns = ['FileName', 'Date', 'Time', 'Temperature'])
    #df.to_csv('results.csv', index=False)