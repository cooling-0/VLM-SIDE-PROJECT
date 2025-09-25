import random
from types import List, Dict


def get_attr_lexicon(category: str, ATTR_LEXICON_GLOBAL, ATTR_LEXICON_PER_CATEGORY):
    base = ATTR_LEXICON_GLOBAL
    over = ATTR_LEXICON_PER_CATEGORY.get(category, {})
    merged = {k: over.get(k, base.get(k, [])) for k in ["color", "material", "fit", "detail"]}
    return merged


def sample_prompts_for_category(category: str, n: int = 30, TEMPLATES : list = None) -> List[str]:
    attrs = get_attr_lexicon(category)
    bag = set()
    for _ in range(n * 2):  # 여유롭게 뽑고 중복 제거
        t = random.choice(TEMPLATES)
        s = t.format(
            color=random.choice(attrs["color"]) if attrs["color"] else "",
            material=random.choice(attrs["material"]) if attrs["material"] else "",
            fit=random.choice(attrs["fit"]) if attrs["fit"] else "",
            detail=random.choice(attrs["detail"]) if attrs["detail"] else "",
            category=category.replace("_", " ")
        ).replace("  ", " ").strip()
        bag.add(s)
        if len(bag) >= n:
            break
    return list(bag)


def make_attribute_phrases(category: str) -> Dict[str, List[str]]:
    attrs = get_attr_lexicon(category)
    phrases = {
        "color":  [f"{c} {category.replace('_',' ')}" for c in attrs["color"]],
        "fit":    [f"{f} {category.replace('_',' ')}" for f in attrs["fit"]],
        "detail": [f"{category.replace('_',' ')} {d}" for d in attrs["detail"]],
        "material":[f"{m} {category.replace('_',' ')}" for m in attrs["material"]],
    }
    # 간단한 기본 카테고리 문장
    phrases["category"] = [f"a photo of a {category.replace('_',' ')}", f"a {category.replace('_',' ')}"]
    return phrases