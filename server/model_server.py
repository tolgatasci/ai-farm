from flask import Flask, jsonify, request
import os

app = Flask(__name__)

MODEL_REPO = {
    "sonnet": {
        "1.0": """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

class Config:
    def __init__(self, vocab_size=1000, d_model=64, nhead=4, num_layers=2):  # Parametreleri küçülttük
        self.vocab_size = vocab_size        # Vocabulary boyutunu küçülttük
        self.d_model = d_model             # Model boyutunu küçülttük
        self.nhead = nhead                 # Head sayısını azalttık
        self.num_layers = num_layers       # Layer sayısını azalttık
        self.dropout = 0.1
        self.dim_feedforward = 256         # Feedforward boyutunu küçülttük

class SonnetModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True  # Batch first parametresini ekledik
            )
            for _ in range(config.num_layers)
        ])
        self.fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)

def create_model(vocab_size=1000, **kwargs):  # Varsayılan vocab_size'ı küçülttük
    config = Config(vocab_size=vocab_size, **kwargs)
    model = SonnetModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

def train_step(model, optimizer, batch):
    model.train()
    data, target = batch
    
    # Veri boyutlarını küçült
    if data.size(1) > 64:  # Sequence length'i sınırla
        data = data[:, :64]
        target = target[:, :64]

    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
    loss.backward()
    optimizer.step()

    return loss.item()

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for data, target in dataloader:
            # Veri boyutlarını küçült
            if data.size(1) > 64:  # Sequence length'i sınırla
                data = data[:, :64]
                target = target[:, :64]
                
            output = model(data)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss
"""
    }
}


@app.route('/models/<name>/<version>', methods=['GET'])
def get_model_code(name, version):
    if name in MODEL_REPO and version in MODEL_REPO[name]:
        return jsonify({
            'name': name,
            'version': version,
            'code': MODEL_REPO[name][version]
        })
    return jsonify({'error': 'Model not found'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)