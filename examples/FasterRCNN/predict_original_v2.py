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

            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

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


def do_predict(pred_func, input_file, output_filename, outpath):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image_ckpt(img, pred_func)
    # print('#####\nResults : {}\n#####'.format(results))
    base_name = os.path.basename(output_filename)
    base_name = os.path.splitext(base_name)[0]
    # for i, mask in enumerate(results):
        # np.save('{}/{}_{}.npy'.format('npy_masks_outputs', base_name, i), results[i].mask)
    # np.savetxt('{}/{}.csv'.format('csv_masks_outputs', base_name), results[0].mask)
    # np.save('{}/{}.npy'.format('npy_masks_outputs', base_name), results[0].mask)
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results)
    else:
        final = draw_final_outputs(img, results)

    holo_perc = 0.00    
    if results:
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
        n_white_pix = np.sum(edge3d == 255)
        all_pix = img.shape[0] * img.shape[1]
        holo_perc = n_white_pix/all_pix
        viz = np.concatenate((img, final, edge3d), axis=1)
    else:
        viz = np.concatenate((img, final), axis=1)
    ann_holo = str(holo_perc)[:6]
    cv2.imwrite("{}/{}_{}.jpg".format(outpath, base_name, ann_holo), viz)
    # logger.info("Inference output for {} written to output.png".format(input_file))
    # tpviz.interactive_imshow(viz)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation.', required=True)
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

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)
    register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    register_hologram(cfg.DATA.BASEDIR)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util
        assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
            "Inference requires either GPU support or MKL support!"
    assert args.load
    finalize_configs(is_training=False)

    if args.predict or args.visualize:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    if args.visualize:
        do_visualize(MODEL, args.load)
    else:
        predcfg = PredictConfig(
            model=MODEL,
            session_init=SmartInit(args.load),
            input_names=MODEL.get_inference_tensor_names()[0],
            output_names=MODEL.get_inference_tensor_names()[1])

        if args.output_pb:
            ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
        elif args.output_serving:
            ModelExporter(predcfg).export_serving(args.output_serving)

        if args.predict:
            outpath = args.output_inference
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                # only applicable to checkpoints model
            files = [f for f in os.listdir(args.predict[0]) if os.path.isfile(os.path.join(args.predict[0], f))]
            imgfiles = [f for f in files if
                        f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png')]
            predictor = OfflinePredictor(predcfg)
            print('Imgfiles {}'.format(imgfiles))
            for image_file in imgfiles:
                print(image_file)
                do_predict(predictor, '{}{}'.format(args.predict[0], image_file), '{}/{}'.format(outpath, image_file), outpath)
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
