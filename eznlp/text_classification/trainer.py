# -*- coding: utf-8 -*-
from typing import List
import torch

from ..training.trainer import Trainer
from .dataset import TextClassificationDataset


class TextClassificationTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        
    def forward_batch(self, batch):
        losses, hidden = self.model(batch, return_hidden=True)
        loss = losses.mean()
        
        batch_labels_pred = self.model.decode(batch, hidden)
        batch_labels_gold = [self.model.decoder.idx2label[label_id] for label_id in batch.label_ids.cpu().tolist()]
        return loss, (batch_labels_gold, batch_labels_pred)
        
    
    def evaluate(self, *set_y: List[List[list]]):
        (y_gold, y_pred), = set_y
        # Use accuracy
        return (sum(yp == yg for yp, yg in zip(y_gold, y_pred)) / len(y_gold), )
        
    
    def predict(self, dataset: TextClassificationDataset, batch_size=32):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        self.model.eval()
        set_labels = []
        with torch.no_grad():
            for batch in dataloader:
                batch.to(self.device)
                set_labels.extend(self.model.decode(batch))
        return set_labels
    
    