# -*- coding: utf-8 -*-
import torch

from ..training.trainer import Trainer
from .dataset import SpanClassificationDataset
from ..sequence_tagging.metric import precision_recall_f1_report


class SpanClassificationTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        
    def forward_batch(self, batch):
        losses, hidden = self.model(batch, return_hidden=True)
        loss = losses.mean()
        
        batch_chunks_pred = self.model.decode(batch, hidden)
        batch_chunks_gold = [spans_obj.chunks for spans_obj in batch.spans_objs]
        return loss, batch_chunks_gold, batch_chunks_pred
        
    
    def evaluate(self, y_gold: list, y_pred: list):
        # Use micro-F1, according to https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']
    
    def predict_chunks(self, dataset: SpanClassificationDataset, batch_size=32):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        self.model.eval()
        set_chunks = []
        with torch.no_grad():
            for batch in dataloader:
                batch.to(self.device)
                set_chunks.extend(self.model.decode(batch))
        return set_chunks
    
    