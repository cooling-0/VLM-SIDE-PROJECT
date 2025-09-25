import torch
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict
from dataclasses import dataclass
from PIL import Image
from ..prototype_gen import build_negative_prototype

# -----------------------------------------
# 8) 텍스트 임베딩 추출
# -----------------------------------------
class TextEmbedder:
    def __init__(self, model_name : str = "openai/clip-vit-base-patch32", device : str = 'cpu'):
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.proc(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats)
        return torch.cat(out, dim=0)
    

@dataclass
class Weights:
    cat: float = 0.5
    color: float = 0.2
    fit: float = 0.15
    detail: float = 0.15
    material: float = 0.0  # 필요 시 활성화

@torch.no_grad()
def score_images(
    NEG_NEIGHBORS : Dict,
    image_paths: List[str],
    prototypes: Dict[str, torch.Tensor],
    embedder: TextEmbedder,
    beta: float = 0.2,
    temperature: float = 0.7,
    zscore_per_image: bool = True
):
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = embedder.proc(images=imgs, return_tensors="pt").to(embedder.device)
    img_feats = embedder.model.get_image_features(**inputs)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # (B, d)

    cat_names = list(prototypes.keys())
    P = torch.cat([prototypes[c] for c in cat_names], dim=0)  # (K, d)

    neg_map = {c: build_negative_prototype(c, prototypes, NEG_NEIGHBORS) for c in cat_names}

    pos_scores = img_feats @ P.T
    neg_scores = torch.zeros_like(pos_scores)
    for j, c in enumerate(cat_names):
        nproto = neg_map[c]
        if nproto is not None:
            neg_scores[:, j] = (img_feats @ nproto.T).squeeze(-1)

    scores = pos_scores - beta * neg_scores

    if zscore_per_image:
        mu = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True).clamp_min(1e-6)
        scores = (scores - mu) / std

    logits = scores / max(1e-6, temperature)
    probs = torch.softmax(logits, dim=-1)  # (B, K)
    return probs, cat_names