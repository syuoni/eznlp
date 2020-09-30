# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .data_utils import tags2entities


def _align_data(dataset: Dataset, raw_data: list):
    data_df = pd.DataFrame(dataset.data)
    raw_data_df = pd.DataFrame(raw_data)
    aligned_df = pd.merge(raw_data_df, data_df, on='raw_idx', how='left')
    return aligned_df, raw_data_df


def _build_covid19_data(aligned_df: pd.DataFrame, raw_data_df: pd.DataFrame, entities_list: list):
    aligned_df['entities_to_build'] = entities_list
    raw_data_df = pd.merge(raw_data_df, aligned_df.groupby('raw_idx')['entities_to_build'].sum(), 
                           left_on='raw_idx', right_index=True, how='left')
    
    covid19_data = []
    for raw_text, entities_to_build in zip(raw_data_df['text'].tolist(), 
                                           raw_data_df['entities_to_build'].tolist()):
        if len(entities_to_build) == 0:
            entities_to_build = Predictor.default_entities
        covid19_data.append({'text': raw_text, 
                             'entities': entities_to_build})
    return covid19_data


class Predictor(object):
    default_entities = [{'entity': 'entity', 
                         'type': 'type', 
                         'start': 1, 
                         'end': 2}]
    
    def __init__(self, tagger: nn.Module):
        self.tagger = tagger

    def predict_tags(self, dataset: Dataset):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate)
        for batch in dataloader:
            for path in self.tagger.decode(batch):
                yield path
    
    def predict_covid19_data(self, dataset: Dataset, raw_data: list):
        df, raw_data_df = _align_data(dataset, raw_data)
        
        entities_pred_list = []
        with tqdm(total=len(dataset)) as t:
            for tokens, raw_text, tags_pred in zip(df['tokens'].tolist(), 
                                                   df['text'].tolist(),
                                                   self.predict_tags(dataset)):
                entities_pred = tags2entities(raw_text, tokens, tags_pred, 
                                              labeling=self.tagger.decoder.tag_helper.labeling)
                entities_pred_list.append(entities_pred)
                t.update(1)
                
        return _build_covid19_data(df, raw_data_df, entities_pred_list)

    def retrieve_covid19_data(self, dataset: Dataset, raw_data: list):
        df, raw_data_df = _align_data(dataset, raw_data)
        
        entities_retr_list = []
        for tokens, raw_text, tags_gold in zip(df['tokens'].tolist(), 
                                               df['text'].tolist(),
                                               df['tags'].tolist()):
            entities_retr = tags2entities(raw_text, tokens, tags_gold, 
                                          labeling=self.tagger.decoder.tag_helper.labeling)
            entities_retr_list.append(entities_retr)
                
        return _build_covid19_data(df, raw_data_df, entities_retr_list)
    
    