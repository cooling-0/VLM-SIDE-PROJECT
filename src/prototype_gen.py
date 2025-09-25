from prompt_ensmble_utils import *
import torch
from models.TextEmb import TextEmbedder, Weights
from typing import Tuple, Dict, Optional


# 프로토타입 생성: (a) 앙상블 평균, (b) 속성 가중 평균
def build_prototype_for_category(category: str, embedder: TextEmbedder, n_samples:int=30) -> torch.Tensor:
    # (a) 템플릿×속성 자동 확장
    WEIGHTS = Weights
    prompts = sample_prompts_for_category(category, n=n_samples)
    emb_a = embedder.embed_texts(prompts).mean(dim=0, keepdim=True)
    # (b) 속성 가중 평균
    phrases = make_attribute_phrases(category)
    cat_e = embedder.embed_texts(phrases["category"]).mean(0, keepdim=True)
    color_e = embedder.embed_texts(phrases["color"]).mean(0, keepdim=True) if phrases["color"] else 0
    fit_e = embedder.embed_texts(phrases["fit"]).mean(0, keepdim=True) if phrases["fit"] else 0
    detail_e = embedder.embed_texts(phrases["detail"]).mean(0, keepdim=True) if phrases["detail"] else 0
    material_e = embedder.embed_texts(phrases["material"]).mean(0, keepdim=True) if phrases["material"] else 0

    weighted = (
        WEIGHTS.cat * cat_e +
        (WEIGHTS.color * color_e if isinstance(color_e, torch.Tensor) else 0) +
        (WEIGHTS.fit * fit_e if isinstance(fit_e, torch.Tensor) else 0) +
        (WEIGHTS.detail * detail_e if isinstance(detail_e, torch.Tensor) else 0) +
        (WEIGHTS.material * material_e if isinstance(material_e, torch.Tensor) else 0)
    )

    proto = (emb_a + weighted) / 2.0
    proto = proto / proto.norm(dim=-1, keepdim=True)
    return proto  # shape: (1, d)


# 전체 카테고리 프로토타입 사전생성
def build_all_prototypes(categories: Dict[str, List[str]], n_samples:int=30) -> Tuple[Dict[str, torch.Tensor], TextEmbedder]:
    embedder = TextEmbedder()
    prototypes = {}
    for cat in categories.keys():
        prototypes[cat] = build_prototype_for_category(cat, embedder, n_samples=n_samples)
    return prototypes, embedder


# 음성(negative) 프로토타입 (양성처럼 동일 방식으로 생성)
def build_negative_prototype(category: str, prototypes: Dict[str, torch.Tensor], NEG_NEIGHBORS : dict) -> Optional[torch.Tensor]:
    neg_cands = NEG_NEIGHBORS.get(category, [])
    if not neg_cands:
        return None
    negs = [prototypes[c] for c in neg_cands if c in prototypes]
    if not negs:
        return None
    neg = torch.stack([t.squeeze(0) for t in negs], dim=0).mean(dim=0, keepdim=True)
    neg = neg / neg.norm(dim=-1, keepdim=True)
    return neg