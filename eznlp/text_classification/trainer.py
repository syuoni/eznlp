# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch

from ..data import Batch
from ..training import Trainer
from .dataset import TextClassificationDataset


class TextClassificationTrainer(Trainer):
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer=None, 
                 scheduler=None, 
                 device=None, 
                 grad_clip=1.0):
        super().__init__(model, optimizer=optimizer, scheduler=scheduler, device=device, grad_clip=grad_clip)
        
        
    def forward_batch(self, batch):
        losses, hidden = self.model(batch, return_hidden=True)
        loss = losses.mean()
        
        acc = self._batch_accuracy(batch, hidden)
        return loss, acc
        
    def predict_labels(self, dataset: TextClassificationDataset, batch_size=32):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        self.model.eval()
        labels = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch.to(self.device)
                labels.extend(self.model.decode(batch))
        return labels
    
    def _batch_accuracy(self, batch: Batch, hidden: torch.Tensor):
        pred_labels = self.model.decode(batch, hidden)
        gold_labels = [self.model.decoder.idx2label[label_id] for label_id in batch.label_id.cpu().tolist()]
        
        return sum(pl==gl for pl, gl in zip(pred_labels, gold_labels)) / len(gold_labels)
    
    