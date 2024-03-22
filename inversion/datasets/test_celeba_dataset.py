from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import json
import numpy as np
import os 
import torchvision.transforms as transforms

class InferenceDataset(Dataset):

	def __init__(self, root):
		
		self.test_paths = sorted(data_utils.make_dataset(root))

		self.transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		
	def __len__(self):
		return len(self.test_paths)

	def __getitem__(self, index):
		img_path = self.test_paths[index]
		img_name = os.path.basename(img_path)
		cam = np.load(img_path.replace('png', 'npy')).astype(np.float32)
		
		img = Image.open(img_path).convert('RGB')
		if self.transform:
			img = self.transform(img)
			
		return img, cam, img_name
