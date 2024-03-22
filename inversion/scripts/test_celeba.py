import argparse
import torch
import sys
import os
sys.path.append(".")
sys.path.append("..")
from criteria import id_loss
from criteria.lpips.lpips import LPIPS
import torch.nn as nn
from datasets.test_celeba_dataset import InferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.train_utils import set_random_seed
import torchvision
from models.RePoI import Model

def flip_yaw(pose_matrix):
    flipped = pose_matrix.clone()
    flipped[:, 0, 1] *= -1
    flipped[:, 0, 2] *= -1
    flipped[:, 0, 3] *= -1
    flipped[:, 1, 0] *= -1
    flipped[:, 2, 0] *= -1
    return flipped

def cal_mirror_c(camera):
    pose, intrinsics = camera[:, :16].reshape(-1, 4, 4), camera[:, 16:].reshape(-1, 3, 3)
    flipped_pose = flip_yaw(pose)
    mirror_camera = torch.cat([flipped_pose.view(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    return mirror_camera

def main(args):

    set_random_seed(666)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    net = Model()
    net.load_state_dict(ckpt['state_dict'])
    net = net.to('cuda')
    net.eval()

    test_dataset = InferenceDataset(root=args.test_dir)
    data_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False)
    
    evaluate_metric(net, data_loader, args)





def evaluate_metric(net, data_loader, args):

    lpips = LPIPS(net_type='alex').to('cuda').eval()
    mse = nn.MSELoss().to('cuda').eval()
    id_ = id_loss.IDLoss().to('cuda').eval()
    
    lp_sum = []
    m_sum = []
    id_sum = []

    out_path = args.save_dir
    save_fig = True
    if save_fig and not os.path.exists(out_path):
        os.makedirs(out_path)

    with torch.no_grad():
        for batch in tqdm(data_loader):
            img, cam, idx = batch
            img, cam = img.to('cuda').float(), cam.to('cuda')
            
            _, _, img_rec = net.forward(img, cam)
            lp = lpips(img_rec, img)
            m = mse(img_rec, img)
            loss_id = id_(img_rec, img)
            
            lp_sum.append(lp)
            m_sum.append(m)
            id_sum.append(loss_id)

            if save_fig:
                vis_x = torch.cat((img, img_rec), dim=-1).detach().cpu()
                torchvision.utils.save_image(vis_x, os.path.join(out_path, idx[0]),
									 normalize=True, scale_each=True, range=(-1, 1))
                
    print(f'metric: lpips {sum(lp_sum) / len(lp_sum)} mse {sum(m_sum) / len(m_sum)} id {sum(id_sum) / len(id_sum)}')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--test_dir", type=str, default="./HFGI3D-main/test_celeba_img_dataset",
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default="./e4e_experiment/visualization",
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)
