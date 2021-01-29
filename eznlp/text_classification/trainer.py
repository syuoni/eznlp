# -*- coding: utf-8 -*-
import tqdm
import torch

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
        
        batch_labels_pred = self.model.decode(batch, hidden)
        batch_labels_gold = [self.model.decoder.idx2label[label_id] for label_id in batch.label_id.cpu().tolist()]
        return loss, batch_labels_gold, batch_labels_pred
        
    
    def evaluate(self, y_gold: list, y_pred: list):
        # Use accuracy
        return sum(yp == yg for yp, yg in zip(y_gold, y_pred)) / len(y_gold)
        
    
    def predict_labels(self, dataset: TextClassificationDataset, batch_size=32):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        self.model.eval()
        set_labels = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                batch.to(self.device)
                set_labels.extend(self.model.decode(batch))
        return set_labels
    
    