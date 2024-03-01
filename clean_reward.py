from typing import Any, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np

class GiberishPenalty(object):
    
    def __init__(self, device: str) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457", device=device)
        self.device = device
        
        print("Giberish reward device: ", self.device)
    
    def __call__(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                return_tensors="pt").to(self.device)
        probs = F.softmax(self.model(**inputs).logits, dim=1)
        # clean_probs = probs[:, 0]
        dirty_probs = probs[:, 1:].sum(dim=1)
        return -dirty_probs.detach().cpu().numpy()
    
    
if __name__ == "__main__":
    texts = [
        "A-z-i-R-E-R-G-A-R-A-",
        "Choose a-B-G-E-R-C-A-L-A-F",
        "I love ML!",
    ]
    
    model = CleanReward("cuda")
    print(model(texts))
    
        