# extract.py
# Chet Russell
# Based on code written by Haniye Kashgarani
# Last edited: April 11, 2024

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
    allfiles=[]

    if f.endswith(".JPG") or f.endswith(".jpg") or f.endswith(".jpeg"):
        image = Image.open(f)
        x, y = image.size
        rimage = image.copy()
        rimage.thumbnail((256, 256), resample=Image.LANCZOS)
        rimage.save(dst_dir + im_name)

def im_meta_data(f, im_name, dst_dir):

    # Grabs temperature
    if f.endswith(".JPG") or f.endswith(".jpg") or f.endswith(".jpeg"):
        image = Image.open(f)

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

        nodata = False

        try:
            img = cv2.imread(os.path.join(f),cv2.IMREAD_UNCHANGED)
            if d["Make"] == "BROWNING":
                img = img[2272:, 1360:1650]
            elif d["Make"] == "RECONYX":
                img = img[:30, -200:]
            else: 
                img = None
        except:
            nodata = True

        # image resizing
        if nodata != True:
            scale_percent = 50
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            status = cv2.imwrite(os.path.join(dst_dir,im_name),img)
            print("Image written to file-system : "+os.path.join(dst_dir,im_name),status,'\n')
        else:
            print("Temperature value cannot be found: "+os.path.join(dst_dir,im_name)+'\n')



def meta_dict(f, im_name, src_dir, dst_dir, dictionary):

    images = dictionary

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    img = PIL.Image.open(f)

    try:
        exif_table = {TAGS.get(k) : v for k, v in img.getexif().items()}
        creation_date = exif_table["DateTime"]
        date, time = creation_date.split()
        temp_img_path = dst_dir / im_name
        temp_img = PIL.Image.open(temp_img_path)
    except:
        temp_img = None
        date = "N/A"
        time = "N/A"

    tempc = "N/A"
    tempf = "N/A"

    if temp_img != None:

        # Tesseract stuff.
        custom_config = r'-l eng --psm 13 --oem 0'
        inv_img = PIL.ImageOps.invert(temp_img)
        temp = pytesseract.image_to_string(inv_img, config=custom_config).strip()

        # Temperature readings and conversions.
        print(temp)
        s = ''.join(x for x in temp if x.isdigit())
        if temp[-1] == "F":
            tempf = s
            tempc = str(round(int(s) - 32) / 1.8)
        elif temp[-1] == "C":
            tempf = str(round(int(s) * 1.8) + 32)
            tempc = s

    # Adding basic metadata to the dictionary.
    images["Name"].append(im_name)

    # Reformatting the date to work with excel
    date = date.replace(":", "/")
    images["Date"].append(date)
    images["Time"].append(time)

    # Check if each temperature has a degree sign. If so, remove it.
    if tempc != "N/A" and tempf != "N/A":
        if tempc[-1] == u"\N{DEGREE SIGN}":
            images["Temp(C)"].append(int(tempc[:-1]))
        else:
            images["Temp(C)"].append(int(tempc))

        if tempf[-1] == u"\N{DEGREE SIGN}":
            images["Temp(F)"].append(int(tempf[:-1]))
        else:
            images["Temp(F)"].append(int(tempf))

    # Closing both images
    if temp_img != None:
        temp_img.close()
    img.close()

    return images