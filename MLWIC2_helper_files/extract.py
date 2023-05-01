# extract.py
# Chet Russell
# Based on code written by Haniye Kashgarani
# Last edited: May 1, 2023

import PIL.ImageOps
from PIL.ExifTags import TAGS
from PIL import Image
import cv2
import os
import pandas as pd
import time
import pytesseract
import PIL.ExifTags
from pathlib import Path
from collections import defaultdict
from pprint import pprint
import array as arr


def crop(f, im_name, dst_dir):
    allfiles = []

    # for root, dirs, files in os.walk(src_dir):
    #    print("1")
    #    for f in files:
    #        print(f)
    if f.endswith(".JPG") or f.endswith(".jpg") or f.endswith(".jpeg"):
        # allfiles.append(os.path.join(src_dir,f))
        # img = cv2.imread(os.path.join(src_dir,f),cv2.IMREAD_UNCHANGED)
        ##print('Original Dimensions : ',img.shape)
        # img = cv2.resize(img, (422,237))
        # status = cv2.imwrite(os.path.join(dst_dir,f),img)
        ##print("Image written to file-system : "+os.path.join(dst_dir,f),status,'\n')

        image = Image.open(f)
        x, y = image.size
        rimage = image.copy()
        rimage.thumbnail((256, 256), resample=Image.LANCZOS)
        # rimage = Image.new('RGBA', (x, x), (0, 0, 0, 0))
        # rimage.paste(image, (0, int((x - y) / 2)))
        # rimage = rimage.resize((256,256))
        rimage.save(dst_dir + im_name)


def im_meta_data(f, im_name, dst_dir):

    # Grabs temperature
    # for root, dirs, files in os.walk(src_dir):
    #    for f in files:
    if f.endswith(".JPG") or f.endswith(".jpg") or f.endswith(".jpeg"):
        image = Image.open(f)
        # crop image and save to destination directory
        img = cv2.imread(os.path.join(f), cv2.IMREAD_UNCHANGED)

        # iterating over all EXIF data fields
        exifdata = image.getexif()
        tags = []
        md = []
        for tag_id in exifdata:
            # get the tag name, instead of human unreadable tag id
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)

            # decode bytes
            if isinstance(data, bytes):
                data = data.decode()
            tags.append(tag)
            md.append(data)

        # create dictionary to contain metadata of image
        d = dict(zip(tags, md))

        if d["Make"] == "BROWNING":
            img = img[2272:, 1360:1650]
        elif d["Make"] == "RECONYX":
            img = img[:30, -200:]

        # image resizing
        scale_percent = 50
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        status = cv2.imwrite(os.path.join(dst_dir, im_name), img)
        print(
            "Image written to file-system : " + os.path.join(dst_dir, im_name),
            status,
            "\n",
        )


def meta_dict(f, im_name, src_dir, dst_dir, dictionary):

    images = dictionary

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    img = PIL.Image.open(f)
    exif_table = {TAGS.get(k): v for k, v in img.getexif().items()}
    creation_date = exif_table["DateTime"]
    date, time = creation_date.split()
    temp_img_path = dst_dir / im_name
    temp_img = PIL.Image.open(temp_img_path)

    # Tesseract stuff.
    custom_config = r"-l eng --psm 13 --oem 0"
    inv_img = PIL.ImageOps.invert(temp_img)
    temp = pytesseract.image_to_string(inv_img, config=custom_config).strip()

    # Temperature readings and conversions.
    if temp[-1] == "F":
        tempf = temp.replace("F", "")
        tempc = round((int(temp[:-2]) - 32) / 1.8)
    elif temp[-1] == "C":
        tempf = round((int(temp[:-2]) * 1.8) + 32)
        tempc = temp.replace("C", "")

    # Adding string versions of the temperature variables to use for the degree check later.
    str_c = str(tempc)
    str_f = str(tempf)

    # Adding basic metadata to the dictionary.
    images["Name"].append(im_name)

    date = date.replace(":", "/")
    images["Date"].append(date)
    images["Time"].append(time)

    # Check if each temperature has a degree sign. If so, remove it.
    if str_c[-1] == "\N{DEGREE SIGN}":
        images["Temp(C)"].append(tempc[:-1])
    else:
        images["Temp(C)"].append(tempc)

    if str_f[-1] == "\N{DEGREE SIGN}":
        images["Temp(F)"].append(tempf[:-1])
    else:
        images["Temp(F)"].append(tempf)

    # Closing both images
    temp_img.close()
    img.close()

    return images
