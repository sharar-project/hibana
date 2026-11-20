import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from hibana.model.flashnext.vision import target_size, MIN_PIXELS, MAX_PIXELS
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize
import random
random.seed(3)
bad = 0
for _ in range(2000):
    w = random.randint(16, 4000); h = random.randint(16, 4000)
    if max(w, h) / min(w, h) > 200:
        continue
    rh, rw = smart_resize(h, w, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    mw, mh = target_size(w, h)
    if (rh, rw) != (mh, mw):
        bad += 1
        if bad < 5: print("MISMATCH", w, h, "hf:", (rw, rh), "mine:", (mw, mh))
print("smart_resize mismatches:", bad, "of ~2000")
