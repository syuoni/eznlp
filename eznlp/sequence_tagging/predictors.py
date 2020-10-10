# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .data_utils import tags2entities


def _align_data(dataset: Dataset, gold_entities_data: list):
    df = pd.DataFrame(dataset.data)
    gold_entities_df = pd.DataFrame(gold_entities_data)
    aligned_df = pd.merge(gold_entities_df, df, on='raw_idx', how='left')
    return aligned_df, gold_entities_df


def _build_entities_data(aligned_df: pd.DataFrame, gold_entities_df: pd.DataFrame, entities_list: list):
    aligned_df['entities_to_build'] = entities_list
    entities_df = pd.merge(gold_entities_df, aligned_df.groupby('raw_idx')['entities_to_build'].sum(), 
                           left_on='raw_idx', right_index=True, how='left')
    
    entities_data = []
    for raw_text, entities_to_build in zip(entities_df['text'].tolist(), 
                                           entities_df['entities_to_build'].tolist()):
        if len(entities_to_build) == 0:
            entities_to_build = Predictor.default_entities
        entities_data.append({'text': raw_text, 
                              'entities': entities_to_build})
    return entities_data


class Predictor(object):
    default_entities = [{'entity': 'entity', 
                         'type': 'type', 
                         'start': 1, 
                         'end': 2}]
    
    def __init__(self, tagger: nn.Module, device):
        self.tagger = tagger
        self.device = device
        

    def predict_tags(self, dataset: Dataset, batch_size=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        self.tagger.eval()
        paths = []
        with torch.no_grad():
            with tqdm(total=len(dataloader)) as t:
                for batch in dataloader:
                    batch.to(self.device)
                    paths.extend(self.tagger.decode(batch))
                    t.update(1)
        return paths
    
    
    def predict_entities_data(self, dataset: Dataset, gold_entities_data: list, **kwargs):
        """
        Parameters
        ----------
        entities_data : list of dicts
            [{'text': raw_text, 
              'entities': [{'entity': entity_text, 
                            'type': entity_type, 
                            'start': entity_start, 
                            'end': entity_end}]}, ...]
        """
        df, gold_entities_df = _align_data(dataset, gold_entities_data)
        
        entities_pred_list = []
        for tokens, raw_text, tags_pred in zip(df['tokens'].tolist(), 
                                               df['text'].tolist(),
                                               self.predict_tags(dataset, **kwargs)):
            entities_pred = tags2entities(raw_text, tokens, tags_pred, 
                                          labeling=self.tagger.decoder.tag_helper.labeling)
            entities_pred_list.append(entities_pred)
            
        return _build_entities_data(df, gold_entities_df, entities_pred_list)
    

    def retrieve_entities_data(self, dataset: Dataset, gold_entities_data: list):
        """
        Parameters
        ----------
        entities_data : list of dicts
            [{'text': raw_text, 
              'entities': [{'entity': entity_text, 
                            'type': entity_type, 
                            'start': entity_start, 
                            'end': entity_end}]}, ...]
        """
        df, gold_entities_df = _align_data(dataset, gold_entities_data)
        
        entities_retr_list = []
        for tokens, raw_text, tags_gold in zip(df['tokens'].tolist(), 
                                               df['text'].tolist(),
                                               df['tags'].tolist()):
            entities_retr = tags2entities(raw_text, tokens, tags_gold, 
                                          labeling=self.tagger.decoder.tag_helper.labeling)
            entities_retr_list.append(entities_retr)
                
        return _build_entities_data(df, gold_entities_df, entities_retr_list)
    
    