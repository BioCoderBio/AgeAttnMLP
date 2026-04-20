import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def extract_esm2_meanpool(csv_path="data/new_peptides.csv", 
                         model_name="facebook/esm2_t33_650M_UR50D",
                         output_path="features/esm2_meanpool.npy"):
    os.makedirs("features", exist_ok=True)
    
    df = pd.read_csv(csv_path)
    sequences = df['sequence'].tolist()
    labels = df['label'].astype(int).values
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda().eval()
    
    feats = []
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Extracting ESM2 features"):
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1022).to('cuda')
            outputs = model(**inputs)
            # mean pooling (without <cls> and <sep>)
            emb = outputs.last_hidden_state[:, 1:-1].mean(dim=1).cpu().numpy()
            feats.append(emb[0])
    
    feats = np.stack(feats)  # (N, 1280)
    np.save(output_path, feats)
    np.save("features/labels.npy", labels)
    print(f"Features saved to: {output_path}, shape={feats.shape}")
    
if __name__ == "__main__":
    extract_esm2_meanpool()