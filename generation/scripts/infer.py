import os
from argparse import Namespace
import sys
sys.path.append(".")
sys.path.append("..")
import torchvision
import torch
from options.test_options import TestOptions
import importlib
from release.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
import sys
sys.path.append('./eg3d')
from eg3d.training.triplane import TriPlaneGenerator
from eg3d.camera_utils import LookAtPoseSampler

import torch
import numpy as np
import random
import os
from model import scd

def set_random_seed(seed=None):
	"""set random seeds for pytorch, random, and numpy.random
	"""
	if seed is not None:
		os.environ['PYTHONHASHSEED'] = str(seed)
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

def load_eg3d(path, device='cuda'):
	# load eg3d
	init_args = ()
	init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
					'channel_max': 512, 'fused_modconv_default': 'inference_only',
					'rendering_kwargs': {'depth_resolution': 48, 'depth_resolution_importance': 48, 'ray_start': 2.25,
					'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
					'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
					'disparity_space_sampling': False, 'clamp_mode': 'softplus',
					'superresolution_module': 'eg3d.training.superresolution.SuperresolutionHybrid8XDC',
					'c_gen_conditioning_zero': False, 'c_scale': 1.0,
					'superresolution_noise_mode': 'none', 'density_reg': 0.25,
					'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
					'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
					'sr_kwargs': {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': 'inference_only'},
					'conv_clamp': None, 'c_dim': 25, 'img_resolution': 512, 'img_channels': 3}


	eg3d = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(device)
	ckpt = torch.load(path, map_location='cpu')
	eg3d.load_state_dict(ckpt['G_ema'], strict=False)
	eg3d.neural_rendering_resolution = 128
	return eg3d

def run(test_opts):
	device = test_opts.device
	os.makedirs(test_opts.output_dir, exist_ok=True)
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	net = scd.Model(test_opts).to(device)
	# net.load_state_dict(ckpt['ema'])
	if  test_opts.use_ema:
		net.load_state_dict(ckpt['ema'])
	else:
		net.load_state_dict(ckpt['state_dict'])
	net.eval()
	
	diffusion = GaussianDiffusion1D(
			net,
			seq_length = 14,
			timesteps = 1000,
			objective = 'pred_v',
			sampling_timesteps=50
	).to(device)

	
	eg3d = load_eg3d(test_opts.eg3d_checkpoint_path, device)
	

	# camera conditions
	camera_lookat_point = torch.tensor(eg3d.rendering_kwargs['avg_camera_pivot'], device=device)
	cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=eg3d.rendering_kwargs['avg_camera_radius'], device=device)
	focal_length = 4.2647
	intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
	c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
	c = c.repeat(test_opts.test_batch_size, 1)
	
	# captions
	if test_opts.captions is not None:
		cap = test_opts.captions
	else:
		cap = None
	# expression codes	
	if test_opts.exp_path is not None:
		exp = torch.tensor(np.load(test_opts.exp_path)).float().unsqueeze(0).repeat(test_opts.test_batch_size, 1).to(device)
		print(exp.shape)
	else:
		exp = None
	
	with torch.no_grad():
		if cap is None:
			w = diffusion.sample(img=None,batch_size=test_opts.test_batch_size, cond=None, cond_scale=test_opts.cond_scale, pos=None, exp=exp)
		else:
			w = diffusion.sample(img=None,batch_size=test_opts.test_batch_size, cond=[cap] * test_opts.test_batch_size, cond_scale=test_opts.cond_scale, pos=None, exp=exp)
		
		x = eg3d.synthesis(ws=w, c=c, noise_mode='const')['image']
		vis_img = x
		torchvision.utils.save_image(vis_img.detach().cpu(), os.path.join(test_opts.output_dir, 'output.png'),
									normalize=True, scale_each=True, range=(-1, 1), nrow=4)

if __name__ == '__main__':

	set_random_seed(999)

	test_opts = TestOptions().parse()
	
	run(test_opts)
