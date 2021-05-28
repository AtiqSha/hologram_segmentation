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
import glob
import copy
from pathlib import Path
import pandas as pd

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from dataset import DatasetRegistry, register_coco, register_hologram
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow
from eval import DetectionResult, multithread_predict_dataflow, predict_image_ckpt
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from viz import (
    draw_annotation, draw_final_outputs, draw_predictions,
    draw_proposal_recall, draw_final_outputs_blackwhite)


def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=SmartInit(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels = dp['image'], dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def do_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()

    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        DatasetRegistry.get(dataset).eval_inference_results(all_results, output)


def do_predict_ckpt(pred_func, input_file, output_filename):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image_ckpt(img, pred_func)
    # print('#####\nResults : {}\n#####'.format(results))
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results)
    else:
        final = draw_final_outputs(img, results)

    holo_perc = 0.00
    if results:
        # print('Results is not None')
        all_r = []
        all_c = []
        for i in results:
            data = i.mask
            binary = data * 255
            dilate = cv2.dilate(binary, np.ones((7, 7), np.uint8))
            erode = cv2.erode(dilate, np.ones((9, 9), np.uint8))
            # edge = binary - dilate
            idx_r, idx_c = np.where(erode == 255)
            all_r.extend(idx_r)
            all_c.extend(idx_c)
        idx1 = np.stack((all_r, all_c), axis=1)
        edge3d = np.zeros((img.shape[0], img.shape[1], 3))
        edge3d[list(idx1.T)] = 255
        # n_white_pix = np.sum(edge3d == 255)
        # w = 0
        # b = 0
        # for pixRow in edge3d:
        #     for pixCol in pixRow:
        #         if pixCol[0] == 0:
        #             b += 1
        #         else:
        #             w += 1
        #
        # holo_cap = 0.36
        # # all_pix = img.shape[0] * img.shape[1]
        # holo_overall = img.shape[0] * img.shape[1] * holo_cap
        # # holo_perc = n_white_pix/all_pix
        # holo_perc = w / holo_overall
        holo_perc = calc_holo_scores(edge3d)
        viz = np.concatenate((img, final, edge3d), axis=1)
    else:
        edge3d = np.zeros((img.shape[0], img.shape[1], 3))
        viz = np.concatenate((img, final), axis=1)
    ann_holo = str(holo_perc)[:6]
    #     cv2.imwrite("{}/{}_{}.jpg".format(outpath, base_name, ann_holo), viz)
    return viz, edge3d, holo_perc

def calc_holo_scores(viz_image):
    w = 0
    b = 0
    for pixRow in viz_image:
        for pixCol in pixRow:
            if pixCol[0] == 0:
                b += 1
            else:
                w += 1

    holo_cap = 0.36
    # all_pix = img.shape[0] * img.shape[1]
    holo_overall = viz_image.shape[0] * viz_image.shape[1] * holo_cap
    # holo_perc = n_white_pix/all_pix
    holo_perc = w / holo_overall

    return holo_perc


def calc_dist_scores_app3(viz_arrays):
    dist_scores = []
    viz_overlaps = []
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
        # print('White : {}/ Black + White : {}'.format(white, black + white))
        holo_perc = white / ((black + white) * 0.36)
        dist_scores.append(holo_perc)
    # averages = sum(holo_percs) / (len(holo_percs))
    return dist_scores



def getFilename(path):
    name = Path(path).stem
    # print(name)
    return name


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-ckpt', help='load a checkpoint model for evaluation.')
    parser.add_argument('--load-pb', help='load a pb model for evaluation.')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", nargs='+')
    parser.add_argument('--benchmark', action='store_true', help="Benchmark the speed of the model + postprocessing")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    parser.add_argument('--output-pb', help='Save a model to .pb')
    parser.add_argument('--output-serving', help='Save a model to serving file')
    parser.add_argument('--output-inference', help='Path to save inference results')
    parser.add_argument('--threshold_benchmark', help='Get threshold of hologram score for multiframes images')
    parser.add_argument('--individual-benchmarking',
                        help='Get result for all individual frames images, current in multiple categories')

    argms = ['--config',
             'DATA.BASEDIR="/home/dataset/batch_3_1" MODE_FPN=True "DATA.VAL=("hologram_val",)"  "DATA.TRAIN=("hologram_train",)"',
             '--load-pb', '',
             #              '--load_ckpt', 'True',
             '--predict', '/home/dataset/test_data_hologram_all/',
             '--individual-benchmarking', 'True',
             '--output-inference', 'all_outputs/ind_bm_m2/']

    # args = parser.parse_args(argms)
    args = parser.parse_args()

    if args.config:
        cfg.update_args(args.config)
    try:
        register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
        register_hologram(cfg.DATA.BASEDIR)
    except:
        print('data is already registered')
        pass

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    # if not tf.test.is_gpu_available():
    #     print('TF GPU_Available {}'.format(tf.test.is_gpu_available()))
    # from tensorflow.python.framework import test_util
    # assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
    #       "Inference requires either GPU support or MKL support!"
    assert args.load_ckpt or args.load_pb
    finalize_configs(is_training=False)

    if args.predict or args.visualize:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    if args.visualize:
        do_visualize(MODEL, args.load_ckpt)
    else:
        if args.load_ckpt:
            predcfg = PredictConfig(
                model=MODEL,
                session_init=SmartInit(args.load_ckpt),
                input_names=MODEL.get_inference_tensor_names()[0],
                output_names=MODEL.get_inference_tensor_names()[1])

    if args.output_pb:
        ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
    elif args.output_serving:
        ModelExporter(predcfg).export_serving(args.output_serving)

    benchmark = args.individual_benchmarking
    if args.load_ckpt:
        pb = False
    else:
        pb = True

    # print('is benchmarking : {}'.format(benchmark))
    multiframe = True
    if args.predict:
        if args.load_ckpt:
            print('predict using ckpt model')
            predictor = OfflinePredictor(predcfg)
            print('done loading OfflinePredictor')
            if not benchmark:
                outpath = args.output_inference
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                # only applicable to checkpoints model
                files = [f for f in os.listdir(args.predict[0]) if os.path.isfile(os.path.join(args.predict[0], f))]
                imgfiles = [f for f in files if
                            f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')]
                #                 predictor = OfflinePredictor(predcfg)
                print('imgfiles {}'.format(imgfiles))
                for image_file in imgfiles:
                    print(image_file)
                    viz_output, holo_score = do_predict_ckpt(predictor, '{}{}'.format(args.predict[0], image_file), '{}/{}'.format(outpath, image_file), outpath)
            else:
                if not multiframe:
                    # not multiframe just to check existence of hologram
                    outpath = args.output_inference
                    print(outpath)
                    print(args.predict)
                    parent_folder = glob.glob(os.path.join(args.predict[0], '*'))
                    for category in parent_folder:
                        print('now in category : {}'.format(category))
                        category_name = getFilename(category)
                        sub_folder = glob.glob(os.path.join(category, '*'))
                        for exp_res in sub_folder:
                            exp_res_name = getFilename(exp_res)
                            if 'no' in exp_res_name:
                                no_hologram = True
                            else:
                                no_hologram = False
                            images = glob.glob(os.path.join(exp_res, '*'))
                            det_outpath = '{}/{}/{}/'.format(outpath, category_name, exp_res_name)
                            if not os.path.exists(det_outpath):
                                os.makedirs(det_outpath)
                            holo_scores_by_exp = []
                            for image in images:
                                print(image)
                                image_name = getFilename(image)
                                viz_output, holo_score = do_predict_ckpt(predictor, image, '{}/{}'.format(outpath, image_name))
                                cv2.imwrite('{}{}.jpg'.format(det_outpath, image_name), viz_output)
                                image_res = [image_name, holo_score]
                                holo_scores_by_exp.append(image_res)
                                # break
                            df = pd.DataFrame(holo_scores_by_exp, columns=['Image Name', 'Holo Scores'])
                            csv_outpath = '{}/{}'.format(outpath, 'results_csv/')
                            if not os.path.exists(csv_outpath):
                                os.makedirs(csv_outpath)
                            df.to_csv(r'{}{}_{}.csv'.format(csv_outpath, category_name, exp_res_name), index=False)
                        # break
                    # break
                else:
                    # multiframe just to check results comparing physical and printed
                    # benchmarking is using 3rd approach - comparing overlapping frames of images
                    outpath = args.output_inference
                    print(outpath)
                    print(args.predict)
                    parent_folder = glob.glob(os.path.join(args.predict[0], '*'))
                    # parent_folder contains - printed & physical
                    for category in parent_folder:
                        print('now in category : {}'.format(category))
                        category_name = getFilename(category)
                        onboarding_IDs = glob.glob(os.path.join(category, '*'))
                        # onboarding ID contains folder of onboarding id (by 4)
                        by_category_dist = []
                        for ob_id in onboarding_IDs:
                            print('Category : {}, of : {}'.format(category_name, ob_id))
                            images = glob.glob(os.path.join(ob_id, '*'))
                            ob_id_name = getFilename(ob_id)
                            det_outpath = '{}/{}/{}/'.format(outpath, category_name, ob_id_name)
                            if not os.path.exists(det_outpath):
                                os.makedirs(det_outpath)
                            all_holo_scores = []
                            viz_outputs = []
                            dist_scores = []
                            # by_dis = []
                            for image in images:
                                image_name = getFilename(image)
                                viz_output, final_edge, holo_score = do_predict_ckpt(predictor, image,
                                                                         '{}/{}'.format(outpath, image_name))
                                cv2.imwrite('{}{}.jpg'.format(det_outpath, image_name), viz_output)
                                all_holo_scores.append(holo_score)
                                viz_outputs.append(final_edge)
                            dist_scores = calc_dist_scores_app3(viz_outputs)
                            temp_dist_scores = copy.deepcopy(dist_scores)
                            temp_dist_scores.insert(0, ob_id_name)
                            by_category_dist.append(temp_dist_scores)
                        csv_outpath = '{}/{}'.format(outpath, 'results_csv/')
                        if not os.path.exists(csv_outpath):
                            os.makedirs(csv_outpath)
                        df = pd.DataFrame(by_category_dist, columns=['Onboarding ID', 'F1-F2', 'F2-F3', 'F3-F4'])
                        df.to_csv(r'{}distance_score_{}.csv'.format(csv_outpath, category_name), index=False)

        elif args.load_pb:
            print('predict using pb model')
            if not benchmark:
                sess, input_tensor, output_tensors = load_session(args.load_pb)
            else:
                outpath = args.output_inference




    #         elif args.threshold_benchmark:
    #             outpath = args.output_inference
    #             if not os.path.exists(outpath):
    #                 os.makedirs(outpath)
    #             parent_file = glob.glob(os.path.join(args.threshold_benchmark, '*'))
    #             data = []
    #             print('Parent File {}'.format(parent_file))
    #             predictor = OfflinePredictor(predcfg)
    #             for file in parent_file:
    #                 img_folder = glob.glob(os.path.join(file, '*'))
    #                 f_name = os.path.basename(file)
    #                 # predictor = OfflinePredictor(predcfg)
    #                 holo_scores = []
    #                 print('Img Folder : {}'.format(img_folder))
    #                 _outpath = outpath + '/' + f_name + '/'
    #                 try:
    #                     for img in img_folder:
    #                         print(img)
    #                         if not os.path.exists(_outpath):
    #                             os.makedirs(_outpath)
    #                         base_name = os.path.basename(img)
    #                         base_name = os.path.splitext(base_name)[0]
    #                         holo_score = do_predict_benchmark(predictor, img, _outpath)
    #                         print('Holo score {}'.format(holo_score))
    #                         holo_scores.append(holo_score)
    #                     b_scores1, b_scores2 = cals_difference_score(holo_scores)
    #                     ind_data = [f_name, b_scores1, b_scores2]
    #                     data.append(ind_data)
    #                     # break
    #                 except:
    #                     pass
    #             df = pd.DataFrame(columns=['Onboarding ID','D_S App1.1','D_S App1.2'])
    #             for d in range(len(data)):
    #                 # print('D {}'.format(data[d]))
    #                 df_length = len(df)
    #                 df.loc[df_length] = data[d]
    #                 # df = df.append(data[d], ignore_index=True)
    #             df.to_csv(r'threshold.csv')
    #             # with open('thresholds.csv', 'w', newline='') as csv_file:
    #                 # writer = csv.writer(file)
    #                 # writer.writerows(data)
    #                 # csv_file.close()

    elif args.evaluate:
        assert args.evaluate.endswith('.json'), args.evaluate
        do_evaluate(predcfg, args.evaluate)
    elif args.benchmark:
        df = get_eval_dataflow(cfg.DATA.VAL[0])
        df.reset_state()
        predictor = OfflinePredictor(predcfg)
        for _, img in enumerate(tqdm.tqdm(df, total=len(df), smoothing=0.5)):
            # This includes post-processing time, which is done on CPU and not optimized
            # To exclude it, modify `predict_image`.
            predict_image(img[0], predictor)

