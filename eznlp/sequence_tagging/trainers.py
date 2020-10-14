# -*- coding: utf-8 -*-
import torch.nn as nn

from ..trainers import Trainer
from .metrics import precision_recall_f1_report


class SequenceTaggingTrainer(Trainer):
    def __init__(self, model: nn.Module, optimizer=None, scheduler=None, 
                 device=None, grad_clip=1.0):
        super().__init__(model, optimizer=optimizer, scheduler=scheduler, 
                         device=device, grad_clip=grad_clip)
        
    def forward_batch(self, batch):
        batch = batch.to(self.device)
        losses, hidden = self.model(batch, return_hidden=True)
        loss = losses.mean()
        # acc = calc_acc(self.model, batch, hidden)
        f1 = calc_f1(self.model, batch, hidden)
        return loss, f1
        
        
        
def calc_acc(model, batch, hidden):
    pred_paths = model.decode(batch, hidden)
    # Here fetch the gold tags, instead of the modeling tags
    gold_paths = model.decoder.tag_helper.fetch_batch_tags(batch.tags_objs)
    
    pred_paths = [t for path in pred_paths for t in path]
    gold_paths = [t for path in gold_paths for t in path]
    return sum(pt==gt for pt, gt in zip(pred_paths, gold_paths)) / len(gold_paths)


def calc_f1(model, batch, hidden):
    pred_paths = model.decode(batch, hidden)
    # Here fetch the gold tags, instead of the modeling tags
    gold_paths = model.decoder.tag_helper.fetch_batch_tags(batch.tags_objs)
    
    chunks_pred_data = [model.decoder.tag_helper.translator.tags2chunks(path) for path in pred_paths]
    chunks_gold_data = [model.decoder.tag_helper.translator.tags2chunks(path) for path in gold_paths]
    scores, ave_scores = precision_recall_f1_report(chunks_gold_data, chunks_pred_data)
    # According to https://www.clips.uantwerpen.be/conll2000/chunking/output.html
    return ave_scores['micro']['f1']

