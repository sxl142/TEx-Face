import torch
import argparse
import importlib

def setup_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    
    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)
    
    model_module = importlib.import_module('.%s' % opts.model_name, 'models')
    net = model_module.Model(opts)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    net = net.to(device)
    return net, opts

