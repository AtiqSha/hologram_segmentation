import re
import json
import base64
# import falcon
from PIL import Image
# import examples.FasterRCNN.predict_original as predict_hologram
import io
from datetime import datetime
import os, psutil, cv2
import numpy as np
import pandas as pd
import glob
from pathlib import Path
# from examples.FasterRCNN.load import load_session
# from examples.FasterRCNN.predict_original import *

def cals_difference_score_a1(scores):
    # approach 1 (highest - lowest)
    app1_1 = 0.0
    app1_1 = max(scores) - min(scores)
    # approach 2 (average of all elements in array scores)
    app1_2 = 0.0
    app1_2 = sum(scores) / (len(scores) - 1)
    return app1_1, app1_2


def cals_difference_score_a2(scores):
    diffList2_1 = []
    diffList2_2 = []
#     print('Scores by reg : {}'.format(scores))
    columns = ['R1', 'R2', 'R3', 'R4', 'R5']
    # 2.1 is by getting highest and lowest of that region
    # 2.2 is getting average of region / frame - 1
    df = pd.DataFrame(scores, columns=columns)
#     print('DF\n{}'.format(df))
    for c in df:
        # print(c)
        if c == 'OB_ID':
            continue
        app21 = max(df[c]) - min(df[c])
        app22 = sum(df[c]) / (len(df) - 1)
        diffList2_1.append(app21)
        diffList2_2.append(app22)
#     print('Approach 2.1 : {}'.format(diffList2_1))
#     print('Approach 2.2 : {}'.format(diffList2_2))

    return diffList2_1, diffList2_2


def cals_difference_score_a3(viz_arrays):
    viz_overlaps = []
    holo_percs = []
    start_app3 = datetime.now()
    for idx_v in range(len(viz_arrays) - 1):
        # height, width,
        # print(idx_v)
        black = 0
        white = 0
        viz_ov_byrow = []
        for idx_pixRow in range(len(viz_arrays[idx_v])):
            viz_ov_bycol = []
            for idx_pixCol in range(len(viz_arrays[idx_v][idx_pixRow])):
                # print(idx_pixCol)
                if viz_arrays[idx_v][idx_pixRow][idx_pixCol][0] == viz_arrays[idx_v + 1][idx_pixRow][idx_pixCol][0]:
                    # print('Same')
                    viz_ov_bycol.append(0)
                    black += 1
                else:
                    viz_ov_bycol.append(255)
                    white += 1
                # pass
            viz_ov_byrow.append(viz_ov_bycol)

        ov = np.array(viz_ov_byrow, dtype=np.uint8)
        # n_white_pix = np.sum(ov == 255)
        # all_pix = np.sum(ov == 255) + np.sum(ov == 0)
        # holo_perc = n_white_pix / all_pix
#         print('White : {}/ Black + White : {}'.format(white, black + white))
        holo_perc = white / (black + white)
        holo_percs.append(holo_perc)
        new_image = Image.fromarray(ov)
        # new_image.save('debug/hologram/overlap_frame{}_{}.png'.format(idx_v, holo_perc))
    end_app3 = datetime.now()
    averages = sum(holo_percs) / (len(holo_percs))
#     print('Elapsed for approach 3 : {}'.format(end_app3 - start_app3))

    return averages, holo_percs


def threshold_config(holo_app, dist_app):
    thres = []
    # hologram by whole mykad
    if holo_app == 1:
        # highest - lowest
        if dist_app == 1:
            pass
        # average of all frames
        elif dist_app == 2:
            pass

    # hologram by 5 different regions
    if holo_app == 2:
        # highest - lowest for each regions
        if dist_app == 1:
            thres = [0.05, 0.05, 0.05, 0.05, 0.05]
        # average of all frames for each regions
        elif dist_app == 2:
            thres = [0.05, 0.05, 0.05, 0.05, 0.05]

    # non-overlapping holograms of frame k and k+1 on whole mykad
    if holo_app == 3:
        # get average of all non-overlapping regions
        if dist_app == 1:
            # average of whole region
            thres = [0.15]
        # get scores for each difference of frame: k1 & k2, k2 & k3, k3 & k4
        elif dist_app == 2:
            thres = [0.1, 0.1, 0.1]
    return thres


def check_passable_hologram(holo_app, dist_app, thres, distance_score):
    distance_passable = []
    #  print('Thres {}'.format(thres))
    # print('Distance score : {}'.format(distance_score))
    if holo_app == 1:
        # one index threshold only
        pass

    # if mykad is separated into multiple smaller regions
    elif holo_app == 2:
        for idx, d in enumerate(distance_score):
            if d < thres[idx]:
                distance_passable.append(0)
            else:
                distance_passable.append(1)

    # if distance score is overlap
    else:
        # whole region
        if dist_app == 1:
            if distance_score[0] < thres[0]:
                distance_passable.append(0)
            else:
                distance_passable.append(1)
        else:
            for idx, d in enumerate(distance_score):
#                 print('D : {}, Thres[Idx] : {}'.format(d, thres[idx]))
                if d < thres[idx]:
                    distance_passable.append(0)
                else:
                    distance_passable.append(1)

    return distance_passable

def generate_hologram_regions():
    region_1 = [0.00, 0.00, 0.28, 0.37]
    region_2 = [0.00, 0.37, 0.61, 0.35]
    region_3 = [0.28, 0.00, 0.32, 0.37]
    region_4 = [0.61, 0.00, 0.26, 0.72]
    region_5 = [0.37, 0.73, 0.40, 1.00]
    regions = OrderedDict([('REG1', region_1),
                           ('REG2', region_2),
                           ('REG3', region_3),
                           ('REG4', region_4),
                           ('REG5', region_5)])


    return regions


def getFilename(path):
    name = Path(path).stem
    # print(name)
    return name

def populate_to_csv(df_inlist, outpath):
    df = pd.DataFrame(df_inlist, columns= ['Onboarding ID', 'F1-F2', 'F2-F3', 'F3-F4'])
    # df.to_csv(r'/home/atiq/Desktop/approach3_for_physical2.csv', index=False)
    df.to_csv(r'{}'.format(outpath), index=False)

'use for each onboarding ID'
if __name__ == '__main__':
    pb_model_file = '/home/atiq/Desktop/Xen_Git/hologram_segm_dr/xendocumentverification/tf_files/malaysia_mykad/holo_segmv2/holo_v1.pb'
    sess, input_tensor, output_tensors = load_session(pb_model_file)

    # outpath = ''
    'set config here'
    approach = []

    folderPath = '/home/atiq/Desktop/clean_pr_original/'
    outpath = '/home/atiq/Desktop/clean_pr_output/'
    folders = glob.glob(os.path.join(folderPath, '*'))
    start_time = datetime.now()
    for f in folders:
        print(f)
        images = glob.glob(os.path.join(f, '*'))
        viz_arrays = []
        holo_scores_by_whole = []
        holo_scores_by_region = []
        model = [sess, input_tensor, output_tensors]
        for i in images:
            img = cv2.imread(i)
            viz, viz_array, edge3d, holo_perc_all, holo_perc_by_reg = do_predict_pb(model, img, outpath + i)
            viz_arrays.append(viz_array)
            holo_scores_by_whole.append(holo_perc_all)
            holo_scores_by_region.append(holo_perc_by_reg)

        app1_1, app1_2 = cals_difference_score_a1(holo_scores_by_whole)
        app2_1, app2_2 = cals_difference_score_a2(holo_scores_by_region)
        app3_1, app3_2 = cals_difference_score_a3(viz_arrays)
        distance_score = 0
        thres = threshold_config(holo_app = approach[0], dist_app = approach[1])
        distance_passable = check_passable_hologram(approach[0], approach[1], thres, distance_score)

    end_time = datetime.now()
