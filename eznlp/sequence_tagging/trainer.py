# -*- coding: utf-8 -*-
import tqdm
import torch

from ..data import Batch
from ..training import Trainer
from .dataset import SequenceTaggingDataset
from .metric import precision_recall_f1_report


class SequenceTaggingTrainer(Trainer):
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
        
        # acc = self._batch_accuracy(batch, hidden)
        f1 = self._batch_f1(batch, hidden)
        return loss, f1
        
    
    def predict_tags(self, dataset: SequenceTaggingDataset, batch_size=32):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        self.model.eval()
        paths = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                batch.to(self.device)
                paths.extend(self.model.decode(batch))
        return paths
    
    
    def predict_chunks(self, dataset: SequenceTaggingDataset, batch_size=32):
        paths = self.predict_tags(dataset, batch_size=batch_size)
        chunks = [self.model.decoder.translator.tags2chunks(path) for path in paths]
        return chunks
    
    
    def _batch_accuracy(self, batch: Batch, hidden: torch.Tensor):
        pred_paths = self.model.decode(batch, hidden)
        gold_paths = [tags_obj.tags for tags_obj in batch.tags_objs]
        
        pred_paths = [t for path in pred_paths for t in path]
        gold_paths = [t for path in gold_paths for t in path]
        return sum(pt==gt for pt, gt in zip(pred_paths, gold_paths)) / len(gold_paths)
    
    
    def _batch_f1(self, batch: Batch, hidden: torch.Tensor):
        pred_paths = self.model.decode(batch, hidden)
        chunks_pred_data = [self.model.decoder.translator.tags2chunks(path) for path in pred_paths]
        
        chunks_gold_data = [tags_obj.chunks for tags_obj in batch.tags_objs]
        scores, ave_scores = precision_recall_f1_report(chunks_gold_data, chunks_pred_data)
        # According to https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        return ave_scores['micro']['f1']
    
    