import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry


from tensorpack.utils import viz
from tensorpack.utils.palette import PALETTE_RGB

from config import config as cfg
from utils.np_box_ops import area as np_area
from utils.np_box_ops import iou as np_iou
from common import polygons_to_mask

__all__ = ["register_ic"]


class ICDemo(DatasetSplit):
    def __init__(self, base_dir, split):
        assert split in ["train", "val", "train_val"]
        base_dir = os.path.expanduser(base_dir)
        self.imgdir = os.path.join(base_dir, split)
        assert os.path.isdir(self.imgdir), self.imgdir
        
#         annotation_file = [os.path.join(self.imgdir,f) for f in os.listdir(self.imgdir) if os.path.isfile(os.path.join(self.imgdir, f)) if f.endswith('.json')]
        
#         from pycocotools.coco import COCO
#         self.coco = COCO(annotation_file)
#         self.annotation_file = annotation_file
#         logger.info("Instances loaded from {}.".format(annotation_file))        

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        div = det(xdiff, ydiff)
        if div == 0:
           raise Exception('lines do not intersect')
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        return x,y

    def training_roidbs(self):
        files = [f for f in os.listdir(self.imgdir) if os.path.isfile(os.path.join(self.imgdir, f))]
        jsonfiles = [f for f in files if f.endswith('.json')]
        imgfiles = [f for f in files if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]

        ret = []
        for i,fn in enumerate(jsonfiles):
            json_file = os.path.join(self.imgdir, fn)
            with open(json_file) as f:
                obj = json.load(f)

            try:
                fname = [filename for filename in imgfiles if '.'.join(fn.split('.')[:-1]) in filename][0] #image filename
                fname = os.path.join(self.imgdir, fname)

                roidb = {"file_name": fname}

                annos = obj["shapes"]

                if i == 0:
                    print('standard annos for {}: {}'.format(fname,annos))
                    
                lines, poly, box = [], [], []
                
#                 lines.append([annos[0]["points"][0], annos[7]["points"][0]]) # left line
#                 lines.append([annos[1]["points"][0], annos[2]["points"][0]]) # top line
#                 lines.append([annos[3]["points"][0], annos[4]["points"][0]]) # right line
#                 lines.append([annos[6]["points"][0], annos[5]["points"][0]]) # bottom line
                
                if len(annos) == 1:
                    poly = np.asarray(annos[0]["points"])
                else:
                    for i, anno in enumerate(annos):
                        if len(anno["points"])==1:
                            poly.append(np.asarray(anno["points"][0]))
                    poly = np.asarray(poly)
                maxxy = poly.max(axis=0)
                minxy = poly.min(axis=0)

                box.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])            

                N = 1
                roidb["boxes"] = np.asarray(box, dtype=np.float32)
                roidb["segmentation"] = [[poly]]

                roidb["class"] = np.ones((N, ), dtype=np.int32)
                roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
                ret.append(roidb)
            except Exception as e:
                print('img file not found for', fn)
                print('annos: ', annos)

        return ret
    
    def inference_roidbs(self):
        files = [f for f in os.listdir(self.imgdir) if os.path.isfile(os.path.join(self.imgdir, f))]
        jsonfiles = [f for f in files if f.endswith('.json')]
        imgfiles = [f for f in files if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]

        ret = []
        for i,fn in enumerate(jsonfiles):
            json_file = os.path.join(self.imgdir, fn)
            with open(json_file) as f:
                obj = json.load(f)

            try:
                fname = [filename for filename in imgfiles if '.'.join(fn.split('.')[:-1]) in filename][0] #image filename
                fname = os.path.join(self.imgdir, fname)

                roidb = {"file_name": fname, "image_id": i}

                annos = obj["shapes"]

                if i == 0:
                    print('standard annos for {}: {}'.format(fname,annos))
                    
                lines, poly, box = [], [], []
                
#                 lines.append([annos[0]["points"][0], annos[7]["points"][0]]) # left line
#                 lines.append([annos[1]["points"][0], annos[2]["points"][0]]) # top line
#                 lines.append([annos[3]["points"][0], annos[4]["points"][0]]) # right line
#                 lines.append([annos[6]["points"][0], annos[5]["points"][0]]) # bottom line
                
                if len(annos) == 1:
                    poly = np.asarray(annos[0]["points"])
                else:
                    for i, anno in enumerate(annos):
                        if len(anno["points"])==1:
                            poly.append(np.asarray(anno["points"][0]))
                    poly = np.asarray(poly)
                maxxy = poly.max(axis=0)
                minxy = poly.min(axis=0)

                box.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])            

                N = 1
                roidb["boxes"] = np.asarray(box, dtype=np.float32)
                roidb["segmentation"] = [[poly]]

                roidb["class"] = np.ones((N, ), dtype=np.int32)
                roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
                ret.append(roidb)
            except Exception as e:
                print('img file not found for', fn)
                print('annos: ', annos)

        return ret    
    
#     def print_coco_metrics(self, results):
#         """
#         Args:
#             results(list[dict]): results in coco format
#         Returns:
#             dict: the evaluation metrics
#         """
#         from pycocotools.cocoeval import COCOeval
#         ret = {}
#         has_mask = "segmentation" in results[0]  # results will be modified by loadRes

#         cocoDt = self.coco.loadRes(results)
#         cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
#         cocoEval.evaluate()
#         cocoEval.accumulate()
#         cocoEval.summarize()
#         fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
#         for k in range(6):
#             ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

#         if len(results) > 0 and has_mask:
#             cocoEval = COCOeval(self.coco, cocoDt, 'segm')
#             cocoEval.evaluate()
#             cocoEval.accumulate()
#             cocoEval.summarize()
#             for k in range(6):
#                 ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
#         return ret
    
    
#     def eval_inference_results(self, results, output=None):
# #         continuous_id_to_COCO_id = {v: k for k, v in self.COCO_id_to_category_id.items()}
#         for res in results:
# #             # convert to COCO's incontinuous category id
# #             if res['category_id'] in continuous_id_to_COCO_id:
# #                 res['category_id'] = continuous_id_to_COCO_id[res['category_id']]
#             # COCO expects results in xywh format
#             box = res['bbox']
#             box[2] -= box[0]
#             box[3] -= box[1]
#             res['bbox'] = [round(float(x), 3) for x in box]

#         if output is not None:
#             with open(output, 'w') as f:
#                 json.dump(results, f)
                
#         if len(results):
#             # sometimes may crash if the results are empty?
#             return self.print_coco_metrics(results)
#         else:
#             return {}    

def register_ic(basedir):
    for split in ["train", "train_val", "val"]:
        print('split: ', split)
        name = "ic_" + split
        DatasetRegistry.register(name, lambda x=split: ICDemo(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "IC"])
        print(DatasetRegistry._metadata_registry)

if __name__ == '__main__':
    basedir = '~/data/ic'
    roidbs = ICDemo(basedir, "train").training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    from tensorpack.utils.viz import interactive_imshow as imshow
    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
        imshow(vis)
