# Text → mood classifier (5 moods)

Fine-tuned `distilroberta` → classifies English text into 5 moods:
`excited, fear, tense, sad, relax`. Test macro-F1 ≈ 0.80.

## Use

```bash
pip install transformers torch
unzip text_mood_clf.zip -d text_mood_clf
```

```python
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("text_mood_clf")
model = AutoModelForSequenceClassification.from_pretrained("text_mood_clf").eval()
LABELS = ["excited", "fear", "tense", "sad", "relax"]

@torch.no_grad()
def predict(texts):
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=64, padding=True)
    p = F.softmax(model(**enc).logits, -1)
    return [(LABELS[i], float(p[r, i])) for r, i in enumerate(p.argmax(-1))]

print(predict(["a calm rainy evening with tea", "best concert of my life!"]))
# [('relax', 0.89), ('excited', 0.93)]
```

Labels come from `model.config.id2label` too — no need to hardcode.
