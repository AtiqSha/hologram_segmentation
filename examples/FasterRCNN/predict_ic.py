#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import cv2
import tqdm

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from dataset import DatasetRegistry, register_coco, register_ic
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow
from eval import DetectionResult, multithread_predict_dataflow, predict_image_ckpt, predict_image_pb
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from viz import (
    draw_annotation, draw_final_outputs, draw_predictions,
    draw_proposal_recall, draw_final_outputs_blackwhite)
from load import load_session


def do_predict_pb(sess, input_tensor, output_tensors, input_file, output_file):
    print('input fn: ', input_file)
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image_pb(sess, input_tensor, output_tensors, img)
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results)
    else:
        final = draw_final_outputs(img, results)
        
    if results:
        binary = results[0].mask*255
        dilate = cv2.dilate(binary, np.ones((7,7), np.uint8))
        erode = cv2.erode(dilate, np.ones((9,9), np.uint8))
        edge = binary - erode
        idx_r, idx_c = np.where(edge==255)
        idx1 = np.stack((idx_r, idx_c), axis=1)
        edge3d = np.zeros((edge.shape[0], edge.shape[1], 3))
        edge3d[list(idx1.T)] = 255
        viz = np.concatenate((img, final, edge3d), axis=1)
    else:
        viz = img
    cv2.imwrite(output_file, viz)
    logger.info("Inference output for {} written to output.png".format(output_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", nargs='+')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    parser.add_argument('--output-inference', help='Path to save inference results')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    finalize_configs(is_training=False)

    outpath = args.output_inference
    if not os.path.exists(outpath):
        os.makedirs(outpath)            
    files = [f for f in os.listdir(args.predict[0]) if os.path.isfile(os.path.join(args.predict[0], f))]
    imgfiles = [f for f in files if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')]            

    for i,image_file in enumerate(imgfiles): 
        do_predict_pb(sess, input_tensor, output_tensors, os.path.join(args.predict[0], image_file), outpath+image_file)  
