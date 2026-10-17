# -*- coding: utf-8 -*-
"""
Prepare the image datasets and labels used to train YOLO26.

The disabled block records the original preparation: convert annotations,
resize external images to a minimum dimension of 640, split by image, and
create non-overlapping MICOLI crops.

The active section uses the saved splits to create overlapping MICOLI crops,
then assigns new random filename prefixes to all training and validation sets.
Training files parked in 03_assets/02_data/99_temp are returned to the main folder first.
"""



from PIL import Image
import pandas as pd
import os
import shutil
import glob
import torchvision.transforms as T

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import re
import copy
import random
from pathlib import Path

random.seed(2398)
np.random.seed(2398)

# change working directory
who = "Carla_windows"

USERS = {
    "Carla_windows": Path(r"C:\Users\caco"),
    "Carla_mac": Path("/Users/carlacoccia"),
    "Carla_linux": Path("/mnt/c/Users/caco"),
}

if who not in USERS:
    raise ValueError(f"Unknown user profile: {who}")

home = USERS[who]


path_to_folder = Path(__file__).resolve().parents[1]
os.chdir(path_to_folder)

# Directories
raw_data_dir = os.path.join(path_to_folder, "03_assets", "01_raw_data") # Raw data
temp_dir = os.path.join(raw_data_dir, "99_temp") # Raw data tempdir to rename files, create labels & split train/validation
data_dir = "03_assets/02_data/data" # Final data folder used for training



def coco_to_yolo(x1, y1, w, h, img_w, img_h):
    """Transform from COCO format (TACO-format) to yolo format"""
    return [(x1 + w/2)/img_w,
            (y1 + h/2)/img_h,
            w/img_w,
            h/img_h]

# Original preparation; leave disabled when using the supplied processed data.
if False:
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                  TACO DATASET
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    # Define source directory
    TACO_main_dir = os.path.join(raw_data_dir, "01_TACO_data")
    TACO_data_dir = os.path.join(TACO_main_dir,"data")

    #-------------
    # Read in annotations and convert to yolo
    #-------------

    # Read in bounding boxes
    df = pd.read_csv(os.path.join(TACO_main_dir,"annotations_taco.csv"))
    df.head()

    # Convert from coco to yolo
    df = pd.concat([df, pd.DataFrame.\
                    from_records(df.apply(lambda x: coco_to_yolo(x["x"],
                                                                 x["y"],
                                                                 x["width"],
                                                                 x["height"],
                                                                 x["img_width"],
                                                                 x["img_height"]
                                                                 ),
                              axis=1),
                              columns=["xcenter", "ycenter", "w", "h"])],axis=1)

    # Select and rename columns
    df = df[['img_id', 'img_file', 'img_width', 'img_height', 'cat_name',
             "xcenter", "ycenter", "w", "h"]]
    df.columns = ["img_id", "filename", "img_width", "img_height", "class",
                    "xcenter", "ycenter", "w", "h"]

    df.head()

    len(df) # number of TACO annotations
    len(df.img_id.unique()) # number of TACO images

    #-------------
    # Move TACO images to temp dir and create lables
    #-------------

    # Get taco photo file names
    files = glob.glob(os.path.join(TACO_data_dir, '**', '*.[jJ][pP][gG]'), recursive=True)
    files = [Path(s).relative_to(TACO_data_dir).as_posix() for s in files
             if not re.search(r"delete", s)] # Relative paths matching the annotation CSV

    # Filter out cleaned taco photos to be used (some taco files were not coded correctly)
    df = df[df["filename"].isin(files)]

    # Give own id and move all images to temporary directory
    ids = df[["filename", "img_id"]].drop_duplicates()
    ids["new_path"] = "taco-" + ids["img_id"].astype(str).str.zfill(5) + ".jpg"

    # Copy images to temp directory
    for i, x in ids.iterrows():
      shutil.copy(os.path.join(TACO_data_dir,x["filename"]),
                  os.path.join(temp_dir,"images",x["new_path"]))

    # Delete batch folders
    # for folder in glob.glob('batch*'):
    #  os.rmdir(folder)

    # Rename images
    df.filename =  "taco-" + df.img_id.astype(str).str.zfill(5) + ".jpg"

    print(df["class"].unique())
    df["class"] = 0  # just one class of trash, not several

    # Resize
    for filename in df.filename.unique():
        filename = os.path.join(temp_dir,"images",filename)
        img = Image.open(filename)
        resize = min(img.size)/640
        size = [int(x/resize) for x in img.size]
        transform = T.Resize(size = (size[1], size[0]))
        img = transform(img)
        img.save(filename)

    # Save .txt file of all labels
    np.savetxt(os.path.join(data_dir,"labels",'all_labels_TACO.txt'),
               df[["class", "xcenter", "ycenter", "w", "h"]].values,
               fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])

    # Save label file for each image
    for filename in df["filename"].unique():
      subdf = df.loc[df["filename"] == filename]
      filename = filename.replace("jpg","txt").replace("JPG","txt")
      np.savetxt(os.path.join(temp_dir,"labels",filename),
                 subdf[["class", "xcenter", "ycenter", "w", "h"]].values,
               fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])


    #-------------
    # Perform train-validation split
    #-------------
    train_inds, test_inds = next(GroupShuffleSplit(test_size=0.30, random_state = 7)\
                                 .split(df, groups=df['img_id']))

    train_taco = df.iloc[train_inds]
    valid_taco = df.iloc[test_inds]

    # Export the csv files (we'll use these to shuffle data in a second step)
    train_taco[['filename', 'img_id']].drop_duplicates().to_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","train_imgs_TACO.csv"))
    valid_taco[['filename', 'img_id']].drop_duplicates().to_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","valid_imgs_TACO.csv"))

    #-------------
    # Move photos into corresponding subfolder of final data folder for training (that needs to have a specific structure)
    #-------------

    # Move training images from temp folder to data folder
    for filename in train_taco.filename.unique():
        shutil.move(os.path.join(temp_dir,"images", filename),
                    os.path.join(data_dir, "images","train", filename))

        filename = filename.replace(".jpg", ".txt")

        shutil.move(os.path.join(temp_dir, "labels", filename),
                    os.path.join(data_dir, "labels", "train", filename))

    # Move validation images from temp folder to data folder
    for filename in valid_taco.filename.unique():
        shutil.move(os.path.join(temp_dir, "images", filename),
                    os.path.join(data_dir, "images", "valid","taco", filename))

        filename = filename.replace(".jpg", ".txt")

        shutil.move(os.path.join(temp_dir, "labels", filename),
                    os.path.join(data_dir, "labels", "valid", "taco", filename))



    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            Extended TACO DATASET                          
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    # Define source directory
    extended_TACO_main_dir = os.path.join(raw_data_dir, "03_extended_TACO")
    extended_TACO_data_dir = os.path.join(extended_TACO_main_dir, "data")

    #-------------
    # Read in annotations and convert to yolo
    #-------------
    df = pd.read_csv(os.path.join(extended_TACO_main_dir,"annotations_extended_taco_cleaned.csv"))
    df.head()

    # Convert from coco to yolo
    df = pd.concat([df, pd.DataFrame.\
                    from_records(df.apply(lambda x: coco_to_yolo(x["x"],
                                                                 x["y"],
                                                                 x["width"],
                                                                 x["height"],
                                                                 x["img_width"],
                                                                 x["img_height"]
                                                                 ),
                              axis=1),
                              columns=["xcenter", "ycenter", "w", "h"])],axis=1)

    df = df[['img_id', 'img_file', 'img_width', 'img_height', 'cat_name',
             "xcenter", "ycenter", "w", "h"]]
    df.columns = ["img_id", "filename", "img_width", "img_height", "class",
                    "xcenter", "ycenter", "w", "h"]

    len(df) # number of TACO annotations
    len(df.img_id.unique()) # number of TACO images


    #-------------
    # Move TACO extended images to temp dir and create lables
    #-------------

    # Rename and move all images to temporary directory
    ids = df[["filename", "img_id"]].drop_duplicates()

    for i, x in ids.iterrows():
      shutil.copy(os.path.join(extended_TACO_data_dir,x["filename"]),
                  os.path.join(temp_dir,"images",x["filename"]))

    df["class"] = 0 # just one class of trash, not several

    # Resize without changing proportions (min dimension = 640) for faster training
    for filename in df.filename.unique():
        filename = os.path.join(temp_dir,"images",filename)
        img = Image.open(filename)
        resize = min(img.size)/640
        size = [int(x/resize) for x in img.size]
        transform = T.Resize(size = (size[1], size[0]))
        img = transform(img)
        img = img.convert('RGB')
        img.save(filename)

    # Save txt of all labels
    np.savetxt(os.path.join(data_dir,"labels",'all_labels_TACO_extended.txt'),
               df[["class", "xcenter", "ycenter", "w", "h"]].values,
               fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])

    #save label file for each image
    for filename in df["filename"].unique():
      subdf = df.loc[df["filename"]==filename]
      filename = filename.replace(".jpg", ".txt")
      np.savetxt(os.path.join(temp_dir,"labels",filename),
                 subdf[["class", "xcenter", "ycenter", "w", "h"]].values,
               fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])



    #-------------
    # Perform train-validation split
    #-------------

    train_inds, test_inds = next(GroupShuffleSplit(test_size=0.30, random_state = 7)\
                                 .split(df, groups=df['img_id']))

    train_taco_extended = df.iloc[train_inds]
    valid_taco_extended = df.iloc[test_inds]

    # Export the csv files (we'll use these to shuffle data in a second step)
    train_taco_extended[['filename', 'img_id']].drop_duplicates().to_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","train_imgs_TACO_extended.csv"))
    valid_taco_extended[['filename', 'img_id']].drop_duplicates().to_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","valid_imgs_TACO_extended.csv"))


    #-------------
    # Move images to final folder
    #-------------

    # Move training photos from temp folder to data folder
    for filename in train_taco_extended.filename.unique():
        shutil.move(os.path.join(temp_dir,"images", filename),
                    os.path.join(data_dir, "images","train", filename))

        filename = filename.replace(".jpg", ".txt")

        shutil.move(os.path.join(temp_dir, "labels", filename),
                    os.path.join(data_dir, "labels", "train", filename))

    # Move validation photos from temp folder to data folder
    for filename in valid_taco_extended.filename.unique():
        shutil.move(os.path.join(temp_dir, "images", filename),
                    os.path.join(data_dir, "images", "valid","taco_extended", filename))

        filename = filename.replace(".jpg", ".txt")

        shutil.move(os.path.join(temp_dir, "labels", filename),
                    os.path.join(data_dir, "labels", "valid", "taco_extended", filename))



    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                 PLASTOPOL DATASET
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    # Define source directory
    plastopol_main_dir = os.path.join(raw_data_dir, "04_plastopol")
    plastopol_data_dir= os.path.join(plastopol_main_dir,"images")

    #-------------
    # Read in annotations and convert to yolo
    #-------------

    df = pd.read_csv(os.path.join(plastopol_main_dir,"annotations_plastopol.csv"))
    df.head()

    # Convert to yolo
    df = pd.concat([df, pd.DataFrame.\
                    from_records(df.apply(lambda x: coco_to_yolo(x["x"],
                                                                 x["y"],
                                                                 x["width"],
                                                                 x["height"],
                                                                 x["img_width"],
                                                                 x["img_height"]
                                                                 ),
                              axis=1),
                              columns=["xcenter", "ycenter", "w", "h"])],axis=1)

    df = df[['img_id', 'img_file', 'img_width', 'img_height',
             "xcenter", "ycenter", "w", "h"]]
    df.columns = ["img_id", "filename", "img_width", "img_height",
                    "xcenter", "ycenter", "w", "h"]

    df.head()

    len(df) # number of Plastopol annotations
    len(df.img_id.unique()) # number of Plastopol images


    #-------------
    # Move PlASTOPOL images to temp dir and create lables
    #-------------

    files = glob.glob(os.path.join(plastopol_main_dir, '**', '*.[jJ][pP][gG]'), recursive=True)
    #filter out delete files
    files = [os.path.basename(s.replace(plastopol_data_dir+"/","")) for s in files if not re.search(r"delete", s)]
    df = df[df["filename"].isin(files)]

    # Rename and move all images to temporary directory
    ids = df[["filename", "img_id"]].drop_duplicates()

    for i, x in ids.iterrows():
      shutil.copy(os.path.join(plastopol_data_dir,x["filename"]),
                  os.path.join(temp_dir,"images",x["filename"]))

    df["class"] = 0

    # Resize without changing proportions (min dimension = 640) for faster training
    for filename in df.filename.unique():
        filename = os.path.join(temp_dir,"images",filename)
        img = Image.open(filename)
        resize = min(img.size)/640
        size = [int(x/resize) for x in img.size]
        transform = T.Resize(size = (size[1], size[0]))
        img = transform(img)
        img = img.convert('RGB')
        img.save(filename)

    # Save all labels as .txt file
    np.savetxt(os.path.join(data_dir,"labels",'all_labels_plastopol.txt'),
               df[["class", "xcenter", "ycenter", "w", "h"]].values,
               fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])

    # Save label file for each image
    for filename in df["filename"].unique():
      subdf = df.loc[df["filename"]==filename]
      filename = filename.replace(".jpg", ".txt")
      np.savetxt(os.path.join(temp_dir,"labels",filename),
                 subdf[["class", "xcenter", "ycenter", "w", "h"]].values,
               fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])


    #-------------
    # Perform train-validation split
    #-------------

    train_inds, test_inds = next(GroupShuffleSplit(test_size=0.30, random_state = 7)\
                                 .split(df, groups=df['img_id']))

    train_plastopol = df.iloc[train_inds]
    valid_plastopol = df.iloc[test_inds]

    # Export the csv files (we'll use these to shuffle data in a second step)
    train_plastopol[['filename', 'img_id']].drop_duplicates().to_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","train_imgs_plastopol.csv"))
    valid_plastopol[['filename', 'img_id']].drop_duplicates().to_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","valid_imgs_plastopol.csv"))


    #-------------
    # Move images to final folder
    #-------------

    # Move training photos and labels from temp folder to data folder
    for filename in train_plastopol.filename.unique():
        shutil.move(os.path.join(temp_dir,"images", filename),
                    os.path.join(data_dir, "images","train", filename))

        filename = filename.replace(".jpg", ".txt")

        shutil.move(os.path.join(temp_dir, "labels", filename),
                    os.path.join(data_dir, "labels", "train", filename))

    # Move validation photos and labels from temp folder to data folder
    for filename in valid_plastopol.filename.unique():
        shutil.move(os.path.join(temp_dir, "images", filename),
                    os.path.join(data_dir, "images", "valid","plastopol", filename))

        filename = filename.replace(".jpg", ".txt")

        shutil.move(os.path.join(temp_dir, "labels", filename),
                    os.path.join(data_dir, "labels", "valid", "plastopol", filename))



    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                             MICOLI bounding boxes 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    MICOLI_main_dir = os.path.join(raw_data_dir, "02_micoli")

    # The first part of this data cleaning only has to be done once,
    # after copying the Nadjas bounding_boxes folder from the main
    # MICOLI folder (downloaded from Labelbox) to the Training folder.
    # If this has already been done once, set first_step to False
    first_step = False

    if first_step:
        if os.path.exists(os.path.join(MICOLI_main_dir,"images/no trash")):
            photos_no_trash = os.listdir(os.path.join(MICOLI_main_dir,"images/no trash"))

            # Create empty txt file in label folder
            for photo in photos_no_trash:
              annotation = photo.replace(".jpg",".txt")
              file=open(os.path.join(MICOLI_main_dir,"labels",annotation), "w")
              file.close()

            # Move no trash pictures to all the others and delte no trash folder
            for photo in photos_no_trash:
                shutil.move(os.path.join(MICOLI_main_dir,"images/no trash",photo),
                          os.path.join(MICOLI_main_dir,"images", photo))

            os.rmdir(os.path.join(MICOLI_main_dir,"images/no trash"))

        micoli_photos = [os.path.join(MICOLI_main_dir, "images", i) for i in
                         os.listdir(os.path.join(MICOLI_main_dir,"images")) if i.endswith(".jpg")]
        micoli_labels = [os.path.join(MICOLI_main_dir, "labels", i ) for i in
                         os.listdir(os.path.join(MICOLI_main_dir,"labels")) if i.endswith(".txt")]

        micoli_all = micoli_photos + micoli_labels

        for path in micoli_all:
            new_filepath = path
            photo = os.path.basename(path)
            if "-rep_add" in photo:
                new_filename = photo.replace("-rep_add", "-re").replace("-", "-add-", 1)
                new_filepath = os.path.join(os.path.dirname(path), new_filename)
                os.rename(path, new_filepath)
                continue
            elif "rep" in photo:
                new_filename = photo.replace("-rep", "-re")
                new_filepath = os.path.join(os.path.dirname(path), new_filename)
                os.rename(path, new_filepath)
                continue
            elif photo.startswith("add-"):
                new_filename = photo.replace("add-", "").replace("-", "-add-", 1)
                new_filepath = os.path.join(os.path.dirname(path), new_filename)
                os.rename(path, new_filepath)
                continue


    micoli_photos = [i for i in os.listdir(os.path.join(MICOLI_main_dir,"images")) if i.endswith(".jpg")]

    micoli_df = pd.DataFrame(micoli_photos,
                             columns =['filename'])
    micoli_df['img_id'] = range(len(micoli_df))


    #-------------
    # Split data into train-validation
    #-------------
    train_inds, test_inds = next(GroupShuffleSplit(test_size=0.30, random_state = 7)\
                                 .split(micoli_df, groups=micoli_df['img_id']))

    train_micoli = micoli_df.iloc[train_inds]
    valid_micoli = micoli_df.iloc[test_inds]

    # Export the csv files (we'll use these to shuffle data in a second step)
    train_micoli.to_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","train_imgs_micoli.csv"))
    valid_micoli.to_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","valid_imgs_micoli.csv"))

    #-------------
    # Move images to final folder
    #-------------
    # Move training photos and labels from temp folder to data folder
    for filename in train_micoli.filename.unique():
        shutil.copy(os.path.join(MICOLI_main_dir, "images", filename),
                        os.path.join(data_dir, "images", "train", filename))

        filename = filename.replace(".jpg", ".txt")
        shutil.copy(os.path.join(MICOLI_main_dir, "labels", filename),
                    os.path.join(data_dir, "labels", "train", filename))

    # Move validation photos and labels from temp folder to data folder
    for filename in valid_micoli.filename.unique():
        shutil.copy(os.path.join(MICOLI_main_dir, "images", filename),
                    os.path.join(data_dir, "images", "valid", "micoli", filename))

        filename = filename.replace(".jpg", ".txt")
        shutil.copy(os.path.join(MICOLI_main_dir, "labels", filename),
                    os.path.join(data_dir, "labels", "valid", "micoli" , filename))


    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                                            MICOLI split
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    # Split MICOLI photos into 4 parts and save each photo and corresponding
    # label file in the final training folder (has to be the same random
    # train-validation split as normal MICOLI)

    def crop(im, height, width):
        # im = Image.open(infile)
        imgwidth, imgheight = im.size
        rows = int(imgheight / height)
        cols = int(imgwidth / width)
        for i in range(rows):
            for j in range(cols):
                # print (i,j)
                box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                yield im.crop(box)

    def create_split_boxes(row, xmin, xmax, ymin, ymax):
        height1 = (min(0.5, ymax) - ymin) * 2
        height2 = (ymax - max(0.5, ymin)) * 2

        width1 = (min(0.5, xmax) - xmin) * 2
        width2 = (xmax - max(0.5, xmin)) * 2

        row0 = copy.copy(row)
        row0["w"] = width1
        row0["h"] = height1
        row0["xcenter"] = max(0, xmin * 2) + row0["w"] / 2
        row0["ycenter"] = max(0, ymin * 2) + row0["h"] / 2

        row1 = copy.copy(row)
        row1["w"] = width2
        row1["h"] = height1
        row1["xcenter"] = max(0, (xmin - 0.5) * 2) + row1["w"] / 2
        row1["ycenter"] = max(0, ymin * 2) + row1["h"] / 2

        row2 = copy.copy(row)
        row2["w"] = width1
        row2["h"] = height2
        row2["xcenter"] = max(0, xmin * 2) + row2["w"] / 2
        row2["ycenter"] = max(0, (ymin - 0.5) * 2) + row2["h"] / 2

        row3 = copy.copy(row)
        row3["w"] = width2
        row3["h"] = height2
        row3["xcenter"] = max(0, (xmin - 0.5) * 2) + row3["w"] / 2
        row3["ycenter"] = max(0, (ymin - 0.5) * 2) + row3["h"] / 2

        return row0, row1, row2, row3

    imgdirs = [data_dir +'/images/train', data_dir + '/images/valid/micoli']

    # Split micoli pictures into 4 parts and save them in the final folder micoli_split (retain train-validation split)
    for imgdir in imgdirs:
        dir_new = re.sub("micoli", "micoli_split", imgdir)
        img_dir_new = dir_new
        labeldir = re.sub("images","labels",imgdir)
        label_dir_new = re.sub("images", "labels", dir_new)

        filelist = glob.glob(os.path.join(imgdir, '*.jpg'))

        for filenum, infile in enumerate(filelist):
            img_name = os.path.basename(infile)
            if re.match(".*(-re|-up)\.jpg$", img_name):
                im = Image.open(infile)
                img_name = img_name.replace(".jpg", "")
                imgwidth, imgheight = im.size
                height = int(imgheight / 2)
                width = int(imgwidth / 2)

                # text file
                column_names = ["class", "xcenter", "ycenter", "w", "h"]
                text_df = pd.read_csv(os.path.join(labeldir, img_name + '.txt'), sep=' ', header=None,
                                      names=column_names)

                #create df for each quarter
                df_0 = pd.DataFrame(columns=column_names)
                df_1 = pd.DataFrame(columns=column_names)
                df_2 = pd.DataFrame(columns=column_names)
                df_3 = pd.DataFrame(columns=column_names)

                #create split txt files
                for index, row in text_df.iterrows():

                    xmax = row["xcenter"] + row["w"] / 2
                    xmin = row["xcenter"] - row["w"] / 2
                    ymax = row["ycenter"] + row["h"] / 2
                    ymin = row["ycenter"] - row["h"] / 2

                    row0, row1, row2, row3 = create_split_boxes(row, xmin, xmax, ymin, ymax)

                    if ((xmax <= 0.5) & (ymax <= 0.5)):
                        # no overlap, append row only to first quarter
                        df_0 = pd.concat([df_0, pd.DataFrame([row0])], ignore_index=True)
                        # df_0 = df_0.append(row0, ignore_index=True)
                    elif ((xmax > 0.5) & (ymax <= 0.5)):
                        df_1 = pd.concat([df_1, pd.DataFrame([row1])], ignore_index=True)
                        # df_1 = df_1.append(row1, ignore_index=True)
                        if (xmin < 0.5):
                            df_0 = pd.concat([df_0, pd.DataFrame([row0])], ignore_index=True)
                            # df_0 = df_0.append(row0, ignore_index=True)
                    elif ((xmax <= 0.5) & (ymax > 0.5)):
                        df_2 = pd.concat([df_2, pd.DataFrame([row2])], ignore_index=True)
                        # df_2 = df_2.append(row2, ignore_index=True)
                        if (ymin < 0.5):
                            df_0 = pd.concat([df_0, pd.DataFrame([row0])], ignore_index=True)
                            # df_0 = df_0.append(row0, ignore_index=True)
                    elif ((xmax > 0.5) & (ymax > 0.5)):
                        df_3 = pd.concat([df_3, pd.DataFrame([row3])], ignore_index=True)
                        # df_3 = df_3.append(row3, ignore_index=True)
                        if (xmin < 0.5):
                            df_2 = pd.concat([df_2, pd.DataFrame([row2])], ignore_index=True)
                            # df_2 = df_2.append(row2, ignore_index=True)
                        if (ymin < 0.5):
                            df_1 = pd.concat([df_1, pd.DataFrame([row1])], ignore_index=True)
                            # df_1 = df_1.append(row1, ignore_index=True)
                        if ((xmin < 0.5) & (ymin < 0.5)):
                            df_0 = pd.concat([df_0, pd.DataFrame([row0])], ignore_index=True)
                            # df_0 = df_0.append(row0, ignore_index=True)

                #save split text files
                np.savetxt(os.path.join(label_dir_new, img_name + "-split-0.txt"), df_0.values,
                           fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])
                np.savetxt(os.path.join(label_dir_new, img_name + "-split-1.txt"), df_1.values,
                           fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])
                np.savetxt(os.path.join(label_dir_new, img_name + "-split-2.txt"), df_2.values,
                           fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])
                np.savetxt(os.path.join(label_dir_new, img_name + "-split-3.txt"), df_3.values,
                           fmt=["%10.0f", "%10.7f", "%10.7f", "%10.7f", "%10.7f"])

                #crop image and save in image dir
                for k, piece in enumerate(crop(im, height, width), start=0):
                    img = Image.new('RGB', (width, height), 255)
                    # print img
                    img.paste(piece)
                    new_filename = img_name+"-split-"+str(k)+".jpg"
                    img.save(os.path.join(img_dir_new, new_filename), quality=95)


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                MICOLI split overlapping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
# Split MICOLI photos into 4 overlapping parts.
# Uses exactly the same train-validation split as normal MICOLI.
#
# All datasets receive new random filename prefixes in the final shuffle.


#-------------
# Read existing MICOLI train-validation split
#-------------

MICOLI_main_dir = os.path.join(raw_data_dir, "02_micoli")

train_micoli = pd.read_csv(
    os.path.join(
        data_dir,
        "group_splits/01_group_splits_no_shuffle",
        "train_imgs_micoli.csv"
    ),
    sep=','
)

valid_micoli = pd.read_csv(
    os.path.join(
        data_dir,
        "group_splits/01_group_splits_no_shuffle",
        "valid_imgs_micoli.csv"
    ),
    sep=','
)


#-------------
# Settings
#-------------

# 10% overlap of the complete image
overlap = 0.10

# If a bounding box is cut by the border of a tile,
# keep it only if at least 40% of the original box is visible
min_visible = 0.40


#-------------
# Output directories
#-------------

train_img_dir_overlap = os.path.join(data_dir, "images", "train")
train_label_dir_overlap = os.path.join(data_dir, "labels", "train")

valid_img_dir_overlap = os.path.join(
    data_dir,
    "images",
    "valid",
    "micoli_split_overlapping"
)

valid_label_dir_overlap = os.path.join(
    data_dir,
    "labels",
    "valid",
    "micoli_split_overlapping"
)

os.makedirs(valid_img_dir_overlap, exist_ok=True)
os.makedirs(valid_label_dir_overlap, exist_ok=True)


#-------------
# Function to create overlapping split
#-------------

def create_overlap_split(df_micoli, img_dir_new, label_dir_new):

    new_files = []

    for filename in df_micoli.filename.unique():

        # Original MICOLI image
        infile = os.path.join(
            MICOLI_main_dir,
            "images",
            filename
        )

        label_file = os.path.join(
            MICOLI_main_dir,
            "labels",
            filename.replace(".jpg", ".txt")
        )

        im = Image.open(infile)

        img_name = filename.replace(".jpg", "")
        imgwidth, imgheight = im.size


        # Four tiles with 10% overlap:
        #
        # tile 0: upper left
        # tile 1: upper right
        # tile 2: lower left
        # tile 3: lower right

        overlap_half = overlap / 2

        tiles = [
            (0,
             0,
             0.5 + overlap_half,
             0.5 + overlap_half),

            (0.5 - overlap_half,
             0,
             1,
             0.5 + overlap_half),

            (0,
             0.5 - overlap_half,
             0.5 + overlap_half,
             1),

            (0.5 - overlap_half,
             0.5 - overlap_half,
             1,
             1)
        ]


        # Read original label file
        column_names = ["class", "xcenter", "ycenter", "w", "h"]

        if os.path.getsize(label_file) > 0:
            text_df = pd.read_csv(
                label_file,
                sep=' ',
                header=None,
                names=column_names
            )
        else:
            text_df = pd.DataFrame(columns=column_names)


        #-------------
        # Create four tiles
        #-------------

        for k, tile in enumerate(tiles):

            tile_xmin = tile[0]
            tile_ymin = tile[1]
            tile_xmax = tile[2]
            tile_ymax = tile[3]

            # Crop coordinates in pixels
            crop_xmin = int(tile_xmin * imgwidth)
            crop_ymin = int(tile_ymin * imgheight)
            crop_xmax = int(tile_xmax * imgwidth)
            crop_ymax = int(tile_ymax * imgheight)

            piece = im.crop(
                (
                    crop_xmin,
                    crop_ymin,
                    crop_xmax,
                    crop_ymax
                )
            )

            new_filename = (
                img_name
                + "-splitov-"
                + str(k)
                + ".jpg"
            )

            piece.save(
                os.path.join(
                    img_dir_new,
                    new_filename
                ),
                quality=95
            )


            #-------------
            # Create labels for tile
            #-------------

            tile_df = pd.DataFrame(columns=column_names)

            tile_width = tile_xmax - tile_xmin
            tile_height = tile_ymax - tile_ymin


            for index, row in text_df.iterrows():

                # Original bounding box
                xmin = row["xcenter"] - row["w"] / 2
                xmax = row["xcenter"] + row["w"] / 2
                ymin = row["ycenter"] - row["h"] / 2
                ymax = row["ycenter"] + row["h"] / 2

                original_area = (
                    (xmax - xmin)
                    *
                    (ymax - ymin)
                )


                # Part of the bounding box inside this tile
                xmin_new = max(xmin, tile_xmin)
                xmax_new = min(xmax, tile_xmax)
                ymin_new = max(ymin, tile_ymin)
                ymax_new = min(ymax, tile_ymax)

                width_new = xmax_new - xmin_new
                height_new = ymax_new - ymin_new


                # Object is not inside this tile
                if width_new <= 0 or height_new <= 0:
                    continue


                visible_area = width_new * height_new
                visible_fraction = visible_area / original_area


                # Do not save very small pieces of objects
                if visible_fraction < min_visible:
                    continue


                # Convert bounding box to coordinates relative to tile
                xcenter_new = (
                    ((xmin_new + xmax_new) / 2 - tile_xmin)
                    /
                    tile_width
                )

                ycenter_new = (
                    ((ymin_new + ymax_new) / 2 - tile_ymin)
                    /
                    tile_height
                )

                width_new = width_new / tile_width
                height_new = height_new / tile_height


                row_new = copy.copy(row)

                row_new["xcenter"] = xcenter_new
                row_new["ycenter"] = ycenter_new
                row_new["w"] = width_new
                row_new["h"] = height_new

                tile_df = pd.concat(
                    [
                        tile_df,
                        pd.DataFrame([row_new])
                    ],
                    ignore_index=True
                )


            # Save label
            label_name = new_filename.replace(".jpg", ".txt")

            np.savetxt(
                os.path.join(
                    label_dir_new,
                    label_name
                ),
                tile_df.values,
                fmt=[
                    "%10.0f",
                    "%10.7f",
                    "%10.7f",
                    "%10.7f",
                    "%10.7f"
                ]
            )

            new_files.append(new_filename)


    return new_files


def restore_training_files(data_root):
    """Return parked training images and labels before reshuffling all datasets."""
    data_root = Path(data_root)
    parked_root = data_root.parent / "99_temp"
    datasets = [
        "01_TACO", "02_TACO_extended", "03_MICOLI", "04_Plastopol",
        "06_MICOLI_split", "07_MICOLI_split_overlapping",
    ]
    moves = []
    for dataset in datasets:
        for kind in ("images", "labels"):
            for source in (parked_root / dataset / kind / "train").glob("*"):
                if source.is_file() and source.suffix.lower() in (".jpg", ".jpeg", ".txt"):
                    target = data_root / kind / "train" / source.name
                    moves.append((source, target))
    targets = [target for _, target in moves]
    if len(targets) != len(set(targets)) or any(p.exists() for p in targets):
        raise FileExistsError("Duplicate training files in data and 99_temp; check before moving.")
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))


restore_training_files(data_dir)

# Strip old prefixes before regenerating crops, so old files cannot replace new crops.
dirs = [
    'images/train', 'labels/train',
    'images/valid/micoli', 'labels/valid/micoli',
    'images/valid/micoli_split', 'labels/valid/micoli_split',
    'images/valid/micoli_split_overlapping', 'labels/valid/micoli_split_overlapping',
    'images/valid/taco', 'labels/valid/taco',
    'images/valid/taco_extended', 'labels/valid/taco_extended',
    'images/valid/plastopol', 'labels/valid/plastopol'
]
dir_names = [ os.path.join(data_dir,i) for i in dirs]

# Delete/reset random order in file naming
for dir in dir_names:
    files = glob.glob(os.path.join(dir, '**', '*.[jJt][pPx][gGt]'), recursive=True)
    files = [os.path.basename(file) for file in files]
    pattern = "^[0-9]+_.*"
    for file in files:
        if re.match(pattern, file):
            dst = os.path.join(dir, re.sub(r'^[^_]+_', '', file))
            src = os.path.join(dir, file)
            if os.path.exists(dst):
                raise FileExistsError(f"Duplicate image or label: {dst}")
            os.rename(src, dst)


#-------------
# Create training overlapping MICOLI
#-------------

new_overlap_train_files = create_overlap_split(
    train_micoli,
    train_img_dir_overlap,
    train_label_dir_overlap
)


#-------------
# Create validation overlapping MICOLI
#-------------

new_overlap_valid_files = create_overlap_split(
    valid_micoli,
    valid_img_dir_overlap,
    valid_label_dir_overlap
)




"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
Shuffle all data and rename
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

#--------
# Get all training and validation datasets
#--------

train_taco = pd.read_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","train_imgs_TACO.csv"), sep=',')
valid_taco = pd.read_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","valid_imgs_TACO.csv"), sep=',')

train_taco_extended = pd.read_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","train_imgs_TACO_extended.csv"), sep=',')
valid_taco_extended = pd.read_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","valid_imgs_TACO_extended.csv"), sep=',')

train_plastopol = pd.read_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","train_imgs_plastopol.csv"), sep=',')
valid_plastopol = pd.read_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","valid_imgs_plastopol.csv"), sep=',')

train_micoli = pd.read_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","train_imgs_micoli.csv"), sep=',')
valid_micoli = pd.read_csv(os.path.join(data_dir,"group_splits/01_group_splits_no_shuffle","valid_imgs_micoli.csv"), sep=',')

train_micoli_split = train_micoli.loc[train_micoli.index.repeat(4)].reset_index(drop=True)
for i in range(0, 4):
    train_micoli_split.loc[train_micoli_split.index % 4 == i, 'filename'] = train_micoli_split.loc[train_micoli_split.index % 4 == i, 'filename'].str.replace('.jpg', f'-split-{i}.jpg')

valid_micoli_split = valid_micoli.loc[valid_micoli.index.repeat(4)].reset_index(drop=True)
for i in range(0, 4):
    valid_micoli_split.loc[valid_micoli_split.index % 4 == i, 'filename'] = valid_micoli_split.loc[valid_micoli_split.index % 4 == i, 'filename'].str.replace('.jpg', f'-split-{i}.jpg')


train_micoli_split_overlapping = train_micoli.loc[train_micoli.index.repeat(4)].reset_index(drop=True)
for i in range(0, 4):
    train_micoli_split_overlapping.loc[train_micoli_split_overlapping.index % 4 == i,'filename'
    ] = train_micoli_split_overlapping.loc[
        train_micoli_split_overlapping.index % 4 == i,
        'filename'
    ].str.replace('.jpg',f'-splitov-{i}.jpg')


valid_micoli_split_overlapping = valid_micoli.loc[valid_micoli.index.repeat(4)].reset_index(drop=True)
for i in range(0, 4):
    valid_micoli_split_overlapping.loc[valid_micoli_split_overlapping.index % 4 == i,'filename'
    ] = valid_micoli_split_overlapping.loc[
        valid_micoli_split_overlapping.index % 4 == i,
        'filename'
    ].str.replace('.jpg',f'-splitov-{i}.jpg')


# Collect all training images
train = pd.concat([
    train_taco[['filename', 'img_id']],
    train_taco_extended[['filename', 'img_id']],
    train_plastopol[['filename', 'img_id']],
    train_micoli[['filename', 'img_id']],
    train_micoli_split[['filename', 'img_id']],
    train_micoli_split_overlapping[['filename', 'img_id']]
])

#--------
# Give random order and export
#--------

# Random training order
train['random'] = np.random.permutation(len(train))

# Keep validation datasets separate, shuffle separately
valid_taco['random'] =np.random.permutation(len(valid_taco))
valid_taco_extended['random'] =np.random.permutation(len(valid_taco_extended))
valid_plastopol['random'] =np.random.permutation(len(valid_plastopol))
valid_micoli['random'] = np.random.permutation(len(valid_micoli))
valid_micoli_split['random'] = np.random.permutation(len(valid_micoli_split))
valid_micoli_split_overlapping['random'] = np.random.permutation(len(valid_micoli_split_overlapping))

# Export random order
group_split_folder = os.path.join(data_dir,"group_splits","02_final_group_splits_shuffled")
train.to_csv(os.path.join(group_split_folder, "final_train_imgs_all.csv"))
valid_micoli.to_csv(os.path.join(group_split_folder, "final_valid_imgs_micoli.csv"))
valid_micoli_split.to_csv(os.path.join(group_split_folder, "final_valid_imgs_micoli_split.csv"))
valid_micoli_split_overlapping.to_csv(os.path.join(group_split_folder,"final_valid_imgs_micoli_split_overlapping.csv"))
valid_taco.to_csv(os.path.join(group_split_folder, "final_valid_imgs_taco.csv"))
valid_plastopol.to_csv(os.path.join(group_split_folder, "final_valid_imgs_plastopol.csv"))
valid_taco_extended.to_csv(os.path.join(group_split_folder, "final_valid_imgs_taco_extended.csv"))


#--------
# Rename according to random order
#--------

dirs = [
    'images/train', 'labels/train',
    'images/valid/taco', 'labels/valid/taco',
    'images/valid/taco_extended', 'labels/valid/taco_extended',
    'images/valid/plastopol', 'labels/valid/plastopol',
    'images/valid/micoli', 'labels/valid/micoli',
    'images/valid/micoli_split', 'labels/valid/micoli_split',
    'images/valid/micoli_split_overlapping',
    'labels/valid/micoli_split_overlapping'
]

dir_names = [ os.path.join(data_dir,i) for i in dirs]

for dir in dir_names:
    df_temp = train
    if re.search("valid/micoli$", dir):
        df_temp = valid_micoli
    elif re.search("valid/micoli_split_overlapping", dir):
        df_temp = valid_micoli_split_overlapping
    elif re.search("valid/micoli_split$", dir):
        df_temp = valid_micoli_split
    elif re.search("valid/taco$", dir):
        df_temp = valid_taco
    elif re.search("valid/taco_extended", dir):
        df_temp = valid_taco_extended
    elif re.search("valid/plastopol", dir):
        df_temp = valid_plastopol

    for i,row in df_temp.iterrows():
        file = row['filename']
        if re.search("labels", dir):
            file= file.replace("jpg","txt").replace("JPG", "txt")
        src = os.path.join(dir,file)

        os.rename(src,os.path.join(dir,str(row['random']) + "_" + file))






