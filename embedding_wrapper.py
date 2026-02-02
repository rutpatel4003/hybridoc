import torch
import torch.nn.functional as F
from typing import List
from transformers import AutoModel, AutoTokenizer

class GemmaEmbeddings:
    """
    Optimized implementation for google/embedding-gemma-300m.
    """
    
    def __init__(
        self,
        model_name: str = "google/embedding-gemma-300m",
        device: str = "cuda",
        batch_size: int = 4,   
        max_length: int = 2048 
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        
        print(f"Loading EmbeddingGemma: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="flash_attention_2"
        )
        self.model.to(self.device)
        self.model.eval()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents using the recommended instruction prefix.
        Format: 'title: <title> | text: <content>' (title is optional)
        """
        # using "none" as title per official Google recommendation for generic RAG
        prefixed = [f"title: none | text: {t}" for t in texts]
        return self._embed_batch(prefixed)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed queries with the mandatory retrieval prefix.
        Format: 'task: search result | query: <content>'
        """
        prefixed = f"task: search result | query: {text}"
        return self._embed_batch([prefixed])[0]
    
    @torch.no_grad()
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # safety 1: clean empty strings before tokenizing
            batch = [t if t.strip() else "empty_chunk_placeholder" for t in batch]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # mean pooling logic
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            sum_embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
            mask_sum = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_embeddings / mask_sum
            # safety 2: zero out any remaining NaNs or Infs before normalization
            mean_pooled = torch.nan_to_num(mean_pooled, nan=0.0, posinf=0.0, neginf=0.0)
            # normalize for cosine similarity
            normalized = F.normalize(mean_pooled, p=2, dim=1)
            embeddings.extend(normalized.cpu().float().numpy().tolist())
            
        return embeddings