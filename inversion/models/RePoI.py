import torch
from torch import nn
from models.encoders import network
from configs.paths_config import model_paths
import sys
sys.path.append('./eg3d/training')
sys.path.append('./eg3d')
from eg3d.training.triplane import TriPlaneGenerator

class Model(nn.Module):

	def __init__(self, opts=None):
		super(Model, self).__init__()
		self.set_opts(opts)
		self.backbone = network.GradualStyleEncoder(50, 'ir_se')
		self.pose_mapping = network.pos_mapping()
		# Load weights if needed
		init_args = ()
		init_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 32768,
		'channel_max': 512, 'fused_modconv_default': False,
		'rendering_kwargs': {'depth_resolution': 48, 'depth_resolution_importance': 48, 'ray_start': 2.25,
		'ray_end': 3.3, 'box_warp': 1, 'avg_camera_radius': 2.7,
		'avg_camera_pivot': [0, 0, 0.2], 'image_resolution': 512,
		'disparity_space_sampling': False, 'clamp_mode': 'softplus',
		'superresolution_module': 'eg3d.training.superresolution.SuperresolutionHybrid8XDC',
		'c_gen_conditioning_zero': False, 'gpc_reg_prob': 0.8, 'c_scale': 1.0,
		'superresolution_noise_mode': 'none', 'density_reg': 0.25,
		'density_reg_p_dist': 0.004, 'reg_type': 'l1', 'decoder_lr_mul': 1.0,
		'sr_antialias': True}, 'num_fp16_res': 0, 'sr_num_fp16_res': 4,
		'sr_kwargs': {'channel_base': 32768, 'channel_max': 512, 'fused_modconv_default': False},
		'conv_clamp': None, 'c_dim': 25, 'img_resolution': 512, 'img_channels': 3}
		print("Reloading Modules!")

		self.decoder = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to('cuda')
		self.load_weights()

		self.residual_encoder = network.Res_encoder(n_channels=6) #Ec
		self.pose_mapping_residual = network.pos_mapping()

		d_model = 512
		self.w_plus_query = nn.Parameter(torch.randn(1, 14, d_model))


		self.cam_proj = nn.Sequential(
			nn.Linear(25, d_model),
			nn.ReLU(),
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Linear(d_model, d_model),
			nn.Dropout(0.1)
		)

		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

	def load_weights(self):
	
		print('Loading encoders weights from irse50!')
		encoder_ckpt = torch.load(model_paths['ir_se50'], map_location='cpu')
		self.backbone.load_state_dict(encoder_ckpt, strict=False)
		print('Loading decoder weights from pretrained!')
		ckpt = torch.load(model_paths['eg3d_ffhq'], map_location='cpu')
		self.decoder.load_state_dict(ckpt['G_ema'], strict=False)
		self.decoder.neural_rendering_resolution = 128
		
		self.__load_latent_avg()

	def forward(self, x, cam, image_mode='image'):
		b = x.shape[0]
		query = self.w_plus_query.expand(b, 14, 512)
		cam_emb = self.cam_proj(cam)
	
		p3, p2, p1 = self.backbone(x)
		codes = self.pose_mapping(p3,p2,p1,query,cam_emb)
		coarse_img_rec = self.decoder.synthesis(ws=codes, c=cam, cache_backbone=False, use_cached_backbone=False,noise_mode='const')
		coarse_img_rec = self.face_pool(coarse_img_rec[image_mode])
		

		p3_res, p2_res, p1_res = self.residual_encoder(torch.cat((x, coarse_img_rec), dim=1))
		codes_res = self.pose_mapping_residual(p3_res, p2_res, p1_res,codes,cam_emb)
		codes = codes + codes_res
		refine_img_rec = self.decoder.synthesis(ws=codes, c=cam, cache_backbone=False, use_cached_backbone=False,noise_mode='const')
		refine_img_rec = self.face_pool(refine_img_rec[image_mode])
		
		return codes, coarse_img_rec, refine_img_rec
		
	
	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, path='./eg3d/eg3d_w_avg.pt', repeat=None):
		self.latent_avg = torch.load(path, map_location='cpu').unsqueeze(0).repeat(self.decoder.backbone.mapping.num_ws, 1).unsqueeze(0)