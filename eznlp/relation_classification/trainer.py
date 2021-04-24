# -*- coding: utf-8 -*-
from typing import List
import torch

from ..training.trainer import Trainer
from .dataset import RelationClassificationDataset
from ..metrics import precision_recall_f1_report


class RelationClassificationTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        
    def forward_batch(self, batch):
        losses, hidden = self.model(batch, return_hidden=True)
        loss = losses.mean()
        
        batch_relations_pred = self.model.decode(batch, hidden)
        batch_relations_gold = [span_pairs_obj.relations for span_pairs_obj in batch.span_pairs_objs]
        return loss, (batch_relations_gold, batch_relations_pred)
        
    
    def evaluate(self, *set_y: List[List[list]]):
        (y_gold, y_pred), = set_y
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return (ave_scores['micro']['f1'], )
    
    
    def predict(self, dataset: RelationClassificationDataset, batch_size=32):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        self.model.eval()
        set_relations = []
        with torch.no_grad():
            for batch in dataloader:
                batch.to(self.device)
                set_relations.extend(self.model.decode(batch))
        return set_relations
    
    