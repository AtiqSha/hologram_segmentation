import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry
import sys
# sys.path.append("examples/FasterRCNN")
__all__ = ["register_hologram"]


class HologramDemo(DatasetSplit):
    def __init__(self, base_dir, split):
        assert split in ["train", "val"]
        base_dir = os.path.expanduser(base_dir)
        self.imgdir = os.path.join(base_dir, split)
        assert os.path.isdir(self.imgdir), self.imgdir

    def training_roidbs(self):
        # json_file = os.path.join(self.imgdir, "via_region_data.json")
        # with open(json_file) as f:
        #     obj = json.load(f)

        files = [f for f in os.listdir(self.imgdir) if os.path.isfile(os.path.join(self.imgdir, f))]
        jsonfiles = [f for f in files if f.endswith('.json')]
        imgfiles = [f for f in files if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')]
        print('LEN JSON FILE {}'.format(len(jsonfiles)))
        ret = []
        for i,fn in enumerate(jsonfiles):

            try:
                json_file = os.path.join(self.imgdir, fn)
                with open(json_file) as f:
                    obj = json.load(f)

                fname = [filename for filename in imgfiles if '.'.join(fn.split('.')[:-1]) in filename][0]  # image filename
                fname = os.path.join(self.imgdir, fname)
                print('Fname {}'.format(fname))
                roidb = {"file_name": fname}
                # print('OBJ {}'.format(obj))
                annos = obj["shapes"]
                # print('Annos {}'.format(annos))
                lines, poly, box = [], [], []
                polies = []
                print(len(annos))
                for a, ann in enumerate(annos):
                    # pass
                    print('Enumerate process')
                    # print(ann)
                    points = ann['points']
                    # print(points)
                    px = []
                    py = []
                    for p in points:
                        px.append(p[0])
                        py.append(p[1])
                    poly = np.stack((px,py), axis = 1)
                    poly = np.asarray(poly)
                    polies.append([poly])

                    maxxy = poly.max(axis=0)
                    minxy = poly.min(axis=0)
                    box.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])

                    N = len(annos)
                    roidb["boxes"] = np.asarray(box, dtype=np.float32)
                    roidb["segmentation"] = polies

                    roidb["class"] = np.ones((N, ), dtype=np.int32)
                    roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
                print('Boxes {}/ Segmentation {}/ Class {}/ Is Crowd {}'.format(len(roidb["boxes"]),
                                                                            len(roidb["segmentation"]),
                                                                            len(roidb["class"]),
                                                                            len(roidb["is_crowd"])))

                ret.append(roidb)
            
            except:
                pass

        # print(ret)
        # ft = open("hologram_datav3.txt", "w")
        # ft.write(str(ret))
        # ft.close()
            # if len(annos) == 1:
            #     poly = np.asarray(annos[0]["points"])
            # else:
            # annos_points = annos["points"]
            # print('ANNOS POINTS {}'.format(annos_points))
            # for a in annos_points:
            #     print('A {}'.format(a))
            #     px.append(a[0])
            #     py.append(a[1])
                # for i, anno in enumerate(annos):
                #     # print('{}'.format(anno["points"]))
                #     if len(anno["points"]) == 1:
                #         poly.append(np.asarray(anno["points"]))
                #     else:
                #         ann_points = anno["points"]
                #         # print('ANN POINTS {}'.format(ann_points))
                #         px = []
                #         py = []
                #         for a in ann_points:
                #             # print('A {}'.format(a))
                #             px.append(a[0])
                #             py.append(a[1])
                #         poly = np.stack((px, py), axis=1)
                #         # print('Poly {}'.format(poly))
                #         # print('{} / {}'.format(px, py))
                #
                #         poly = np.asarray(poly)
                #         polies.append(poly)
                #         # print('Poly {}'.format(poly))
                #
                #         maxxy = poly.max(axis=0)
                #         minxy = poly.min(axis=0)
                #
                #         box.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])
                #
                #         N = len(annos)
                #         roidb["boxes"] = np.asarray(box, dtype=np.float32)
                #         roidb["segmentation"] = polies
                #
                #         roidb["class"] = np.ones((N, ), dtype=np.int32)
                #         roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
                #         ret.append(roidb)
            # break

        # print(ret)

        # ft = open("hologram_datav2.txt", "w")
        # ft.write(str(ret))
        # ft.close()
            # except Exception as e:
            #     print('img file not found for', fn)
            #     print('annos: ', annos)

        # for _, v in obj.items():
        #     fname = v["filename"]
        #     fname = os.path.join(self.imgdir, fname)
        #
        #     roidb = {"file_name": fname}
        #
        #     annos = v["regions"]
        #
        #     boxes = []
        #     segs = []
        #     for _, anno in annos.items():
        #         assert not anno["region_attributes"]
        #         anno = anno["shape_attributes"]
        #         px = anno["all_points_x"]
        #         py = anno["all_points_y"]
        #         poly = np.stack((px, py), axis=1) + 0.5
        #         maxxy = poly.max(axis=0)
        #         minxy = poly.min(axis=0)
        #
        #         boxes.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])
        #         segs.append([poly])
        #     N = len(annos)
        #     roidb["boxes"] = np.asarray(boxes, dtype=np.float32)
        #     roidb["segmentation"] = segs
        #     roidb["class"] = np.ones((N, ), dtype=np.int32)
        #     roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
        #     ret.append(roidb)
        #     print(ret)
        #     break
        return ret


def register_hologram(basedir):
    for split in ["train", "val"]:
        name = "hologram_" + split
        DatasetRegistry.register(name, lambda x=split: HologramDemo(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["HG", "hologram"])


if __name__ == '__main__':
    # basedir = '~/data/balloon'
    # basedir = '/media/atiq/T7/atiqshariff/xendity/dataset/labelled/hologram_segmentation_by_batch_labelled/'
    basedir = '/home/atiq/Desktop/hologram_segmentation_by_batch_labelled'
    roidbs = HologramDemo(basedir, "train").training_roidbs()
    # print(roidbs)
    print("#images:", len(roidbs))
    # sys.path.append("examples/FasterRCNN")

    # from viz import draw_annotation
    # from tensorpack.utils.viz import interactive_imshow as imshow
    # import cv2
    # for r in roidbs:
    #     im = cv2.imread(r["file_name"])
    #     # print(im)
    #     print(r["class"])
    #     vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
    #     imshow(vis)
    #     break
