import cv2
import pandas as pd
import os
import copy
import numpy as np
import shutil
import glob
from pathlib import Path
import statistics


def getFilename(path):
    name = Path(path).stem
    # print(name)
    return name

def check_holo(row):
    category = 'C'
    for col_idx, column in enumerate(row):
        # print(column)
        if col_idx == 0:
            pass
        else:
            if column > 0.0:
                if column > 0.20:
                    # have hologram and sufficient
                    category = 'A'
                    break
                else:
                    category = 'B'
                    pass

            else:
                pass
    # print(category)
    return category

def cleanup_by_category(df_by_category, category, category_sub, src_path, dest_src_path):
    # category = printed/ physical/ onscreen
    # category_sub = A, B, C
    img_src_path = '{}/{}'.format(src_path, 'img')
    dist_src_path = '{}/{}'.format(src_path, 'dist')
    all_images_category = glob.glob(os.path.join(img_src_path, '*'))
    all_dist_category = glob.glob(os.path.join(dist_src_path, '*'))
    # print(all_images_category)
    #cleanup for images outputs
    # print(category)
    for category_img_folder in all_images_category:
        # print(category_img_folder)
        if '.zip' in category_img_folder:
            continue
        category_img_name = getFilename(category_img_folder)
        if category in category_img_name:
            this_src_path = category_img_folder
            break
    # print('This Image Path to Clean : {}'.format(this_src_path))
    onboarding_ID_dest_path_img = '{}/img/{}/{}_{}'.format(dest_src_path, category, category, category_sub)
    # print('Out you go : {}'.format(onboarding_ID_dest_path_img))
    if not os.path.exists(onboarding_ID_dest_path_img):
        os.makedirs(onboarding_ID_dest_path_img)
    for idx, row in df_by_category.iterrows():
        onboarding_ID = row['Onboarding ID']
        onboarding_ID_img_path = this_src_path + '/' + onboarding_ID + '/'

        shutil.copytree(onboarding_ID_img_path, '{}/{}'.format(onboarding_ID_dest_path_img, onboarding_ID))

    #cleanup on distance score
    for distance_score_csv in all_dist_category:
        if category in distance_score_csv:
            this_src_path_csv = distance_score_csv
            break

    # print(this_src_path_csv)
    print(category_sub)
    print(df_by_category)
    cat_obID = df_by_category['Onboarding ID'].tolist()
    # print('Cat obID : {}'.format(cat_obID))
    dist_df = pd.read_csv(r'{}'.format(this_src_path_csv))
    subdf = dist_df[dist_df['Onboarding ID'].isin(cat_obID)]
    subdf = subdf.reset_index(drop=True)
    # print(subdf)
    onboarding_ID_dest_path_dist = '{}/dist/{}/'.format(dest_src_path, category)
    if not os.path.exists(onboarding_ID_dest_path_dist):
        os.makedirs(onboarding_ID_dest_path_dist)
    subdf.to_csv(r'{}/{}_{}.csv'.format(onboarding_ID_dest_path_dist, category, category_sub), index=False)

    # return None


if __name__ == '__main__':
    # pass
    # D:\atiqshariff\xendity\projects\fserver_multiF\ in \img
    PARENT_HOLO_FOLDER = 'D:/atiqshariff/xendity/projects/fserver_multiF/train_val_cropped_benchmarking_rerun/in/holo/'
    PARENT_DIST_FOLDER = 'D:/atiqshariff/xendity/projects/fserver_multiF/train_val_cropped_benchmarking_rerun/in/dist/'
    PARENT_IMG_FOLDER = 'D:/atiqshariff/xendity/projects/fserver_multiF/train_val_cropped_benchmarking_rerun/in/img/'
    PARENT_IN_FOLDER = 'D:/atiqshariff/xendity/projects/fserver_multiF/train_val_cropped_benchmarking_rerun/in/'
    DEST_OUT_FOLDER = 'D:/atiqshariff/xendity/projects/fserver_multiF/train_val_cropped_benchmarking_rerun/out/'
    # DEST_CSV_FOLDER = 'D:/atiqshariff/xendity/projects/results_benchmarking_multiframe_v1/csvs/'
    # DEST_CLEANED_IMG = 'D:/atiqshariff/xendity/projects/results_benchmarking_multiframe_v1/viz_outputs/'
    all_csvs = glob.glob(os.path.join(PARENT_HOLO_FOLDER, '*'))
    # REMINDER : SCORES TO REMOVE THE ONBOARDING IDS ARE BASED ON HOLOGRAM SCORES
    exceptional = []
    for csv in all_csvs:
        print(csv)
        df = pd.read_csv(r'{}'.format(csv))
        category_name = csv.split('.')[0].split('_')[-1]

        print(category_name)
        # iterate each row and check whether have holo or not
        category_A = []
        category_B = []
        category_C = []
        for idx, row in df.iterrows():
            # print(row)
            have_holo_category = check_holo(row)
            if have_holo_category == 'A':
                category_A.append(row)
            elif have_holo_category == 'B':
                category_B.append(row)
            else:
                category_C.append(row)
                # break

        new_dfA = pd.DataFrame(category_A)
        new_dfB = pd.DataFrame(category_B)
        new_dfC = pd.DataFrame(category_C)
        this_category_result = '{}/holo/{}'.format(DEST_OUT_FOLDER, category_name)
        print('Out you go : {}'.format(this_category_result))
        if not os.path.exists(this_category_result):
            os.makedirs(this_category_result)

        new_dfA.to_csv(r'{}/{}_have_hologram.csv'.format(this_category_result, category_name), index = False)
        new_dfB.to_csv(r'{}/{}_insuff_hologram.csv'.format(this_category_result, category_name), index = False)
        new_dfC.to_csv(r'{}/{}_no_hologram.csv'.format(this_category_result, category_name), index = False)
        try:
            cleanup_by_category(new_dfA, category_name, 'have_hologram', PARENT_IN_FOLDER, DEST_OUT_FOLDER)
            cleanup_by_category(new_dfB, category_name, 'insuff_hologram', PARENT_IN_FOLDER, DEST_OUT_FOLDER)
            cleanup_by_category(new_dfC, category_name, 'no_hologram', PARENT_IN_FOLDER, DEST_OUT_FOLDER)
        except:
            exceptional.append(csv)

    print('Cannot be compute: error 404 : no dataframe {}'.format(exceptional))
