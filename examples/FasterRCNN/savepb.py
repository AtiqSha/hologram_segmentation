import argparse

from config import config as cfg
from dataset import DatasetRegistry, register_coco, register_ic, register_hologram
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel

from tensorpack.predict import PredictConfig
from tensorpack.tfutils import SmartInit
from tensorpack.tfutils.export import ModelExporter

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py", nargs='+')	
	parser.add_argument('--load', help='load a model for evaluation.', required=True)
	parser.add_argument('--output-pb', help='Save a model to .pb')

	args = parser.parse_args()
	if args.config:
		cfg.update_args(args.config)
	register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
	register_hologram(cfg.DATA.BASEDIR)	

	cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

	MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

	predcfg = PredictConfig(
		model=MODEL,
		session_init=SmartInit(args.load),
		input_names=MODEL.get_inference_tensor_names()[0],
		output_names=MODEL.get_inference_tensor_names()[1])

	ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
