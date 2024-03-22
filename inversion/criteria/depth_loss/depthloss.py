import torch
from criteria.depth_loss.segmodel import BiSeNet
from torchvision.transforms import transforms
import torch.nn.functional as F

class DepthLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cuda'
        self.model_path = "pretrained/faceparsing_model.pth"
        self.init_model()

    def init_model(self):
        n_classes = 19
        net = BiSeNet(n_classes=n_classes).to(self.device)
        net.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.net = net
        self.net.eval()
        self.to_tensor = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        for k, v in self.net.named_parameters():
            v.requires_grad = False

    def forward(self, x, dep):
        with torch.set_grad_enabled(False):
            mask = self.net(self.to_tensor((x + 1) / 2))
        mask = mask.argmax(1)
        mask[mask != 0] = 1
        mask = mask.unsqueeze(1).float()
        H, W = dep.shape[-2:]
        mask = F.interpolate(mask, (H, W), mode='nearest')
        dep_ = dep * (1-mask)
        avg_depth = torch.ones_like(dep) * 2.7729 * (1-mask)
        depth_loss = F.mse_loss(dep_, avg_depth) 
        return depth_loss
        
        

