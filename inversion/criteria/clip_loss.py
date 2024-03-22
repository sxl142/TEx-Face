import torch
import clip
import torch.nn.functional as F

class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=256 // 32)

    def v2t(self, image, text): 
        image = self.avg_pool(self.upsample(image))
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        similarity = 1 - (F.cosine_similarity(image_features, text_features)).mean()
        # print('t', similarity)

        return similarity
    
    def v2v(self, image1, image2): 
        image1 = self.avg_pool(self.upsample(image1))
        image2 = self.avg_pool(self.upsample(image2))
        image_features1 = self.model.encode_image(image1)
        image_features2 = self.model.encode_image(image2)
        # print(image_features1.shape,  image_features2.shape)
        similarity = 1 - (F.cosine_similarity(image_features1, image_features2)).mean()
        # print(similarity)
        return similarity
    

    