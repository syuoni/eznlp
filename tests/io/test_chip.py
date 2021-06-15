# -*- coding: utf-8 -*-
import pytest
import jieba

from eznlp.io import ChipIO
from eznlp.utils.chunk import detect_nested, filter_clashed_by_priority

class TestChipIO(object):
    """
    
    References
    ----------
    [1] http://www.cips-chip.org.cn/2020/eval1
    """
    def test_chip2020_task1(self):
        io = ChipIO(tokenize_callback='char', sep='|||', encoding='utf-8')
        train_data, train_errors, train_mismatches = io.read("data/chip2020/task1/train_data.txt", return_errors=True)
        dev_data,   dev_errors,   dev_mismatches   = io.read("data/chip2020/task1/val_data.txt", return_errors=True)
        
        assert len(train_data) == 15_000
        assert sum(len(ex['chunks']) for ex in train_data) == 61_796
        assert len(train_errors) == 0
        assert len(train_mismatches) == 0
        assert len(dev_data) == 5_000
        assert sum(len(ex['chunks']) for ex in dev_data) == 20_300
        assert len(dev_errors) == 0
        assert len(dev_mismatches) == 0
        
        assert any(detect_nested(ex['chunks']) for data in [train_data, dev_data] for ex in data)
        # TODO: 
        # assert all(filter_clashed_by_priority(ex['chunks'], allow_nested=True) == ex['chunks'] for data in [train_data, dev_data] for ex in data)
