import torch
import torch.nn as nn
import clip

class BaseEmbedding(nn.Module):
    def get_loss(self):
        return None

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            super().train()
        return self

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()

class CLIPImageEmbedding(BaseEmbedding):
    def __init__(self, 
                 clip_name='ViT-B/32',
                 normalize=True,
        ):
        super().__init__()
        self.normalize = normalize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(clip_name, device=self.device, jit=False)

        self.visual = model.visual
        
        self.trainable = False
        # self._set_trainable()

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def finetune(self):
        self.trainable = True
        for pn, p in self.named_parameters():
            p.requires_grad = True
        self.train(mode=True)
    
    def encode_image(self, image):
        emb = self.visual(image.type(self.dtype))
        if self.normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

class CLIPTextEmbedding(BaseEmbedding):
    def __init__(self, 
                 clip_name='ViT-B/32',
                 normalize=True,
                 pick_last_embedding=False,
        ):
        super().__init__()
        self.clip_name = clip_name
        self.normalize = normalize
        self.pick_last_embedding = pick_last_embedding

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        
        self.trainable = False
        self._set_trainable()

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype

    def encode_text(self, text):
        # print(text)
        
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        if self.pick_last_embedding:
            x = x[torch.arange(x.shape[0]), index.argmax(dim=-1)] @ self.text_projection
        
        if self.normalize:
            x = x / x.norm(dim=-1, keepdim=True)

        return x
