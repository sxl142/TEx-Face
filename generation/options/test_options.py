from argparse import ArgumentParser

class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--output_dir', type=str, default='./outputs', help='Path to experiment output directory')
		self.parser.add_argument('--test_batch_size', default=6, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--checkpoint_path', default='./checkpoints/model.pt', type=str, help='Path to model checkpoint')
		self.parser.add_argument('--eg3d_checkpoint_path', default='../inversion/pretrained/ffhqrebalanced512-128.pth', type=str, help='Path to model checkpoint')
		self.parser.add_argument('--device', default='cuda', type=str, help='Path to model checkpoint')
		#
		self.parser.add_argument('--captions', default='This woman has no bangs.', type=str, help='Path to model checkpoint')
		self.parser.add_argument('--exp_path', default='./exp_data/exp5.npy', type=str, help='Path to model checkpoint')
		self.parser.add_argument('--cond_drop_prob_txt', default=0.5, type=float)
		self.parser.add_argument('--cond_drop_prob', default=0.5, type=float)
		self.parser.add_argument('--cond_drop_prob_exp', default=0.5, type=float)
		self.parser.add_argument('--head', default=8, type=int)
		self.parser.add_argument('--input_dim', default=512, type=int)
		self.parser.add_argument('--output_dim', default=512, type=int)
		self.parser.add_argument('--en_layers', default=4, type=int)
		self.parser.add_argument('--embed_dim_en', default=512, type=int)
		self.parser.add_argument('--drop_rate', default=0.1, type=float)
		self.parser.add_argument('--use_rotary', default=True, action="store_true")
		self.parser.add_argument('--use_ema', default=False, action="store_true")
		self.parser.add_argument('--cond_scale', default=7.0, type=float)


	def parse(self):
		opts = self.parser.parse_args()
		return opts