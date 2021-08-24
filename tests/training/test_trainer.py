# -*- coding: utf-8 -*-
import pytest
import random
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, SequenceTaggingDecoderConfig, ExtractorConfig
from eznlp.training import Trainer


@pytest.mark.parametrize("use_amp", [False, True])
def test_train_steps(use_amp, conll2003_demo, device):
    if use_amp and device.type.startswith('cpu'):
        pytest.skip("test requires cuda, while current session runs on cpu")
    
    config = ExtractorConfig('sequence_tagging')
    dataset = Dataset(conll2003_demo, config)
    dataset.build_vocabs_and_dims()
    model = config.instantiate().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = Trainer(model, optimizer=optimizer, use_amp=use_amp, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate)
    trainer.train_steps(train_loader=dataloader, 
                        dev_loader=dataloader, 
                        num_epochs=4, 
                        disp_every_steps=1, 
                        eval_every_steps=2)



def test_gradient_accumulation(conll2003_demo, device):
    # Note: set dropout rate as 0 for consistency
    config = ExtractorConfig(intermediate2=EncoderConfig(in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0), 
                             decoder=SequenceTaggingDecoderConfig(in_drop_rates=(0.0, 0.0, 0.0)))
    dataset = Dataset(conll2003_demo[:8], config)
    dataset.build_vocabs_and_dims()
    
    seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    model1 = config.instantiate().to(device)
    torch.manual_seed(seed)
    model2 = config.instantiate().to(device)
    assert all((p1 - p2).abs().max().item() < 1e-4 for p1, p2 in zip(model1.parameters(), model2.parameters()))
    params_backup = [p1.data.clone() for p1 in model1.parameters()]
    
    
    optimizer1 = torch.optim.AdamW(model1.parameters())
    trainer1 = Trainer(model1, optimizer=optimizer1, device=device)
    dataloader1 = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.collate)
    trainer1.train_steps(train_loader=dataloader1, num_epochs=1)
    
    optimizer2 = torch.optim.AdamW(model2.parameters())
    trainer2 = Trainer(model2, optimizer=optimizer2, num_grad_acc_steps=2, device=device)
    dataloader2 = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate)
    trainer2.train_steps(train_loader=dataloader2, num_epochs=1)
    
    assert trainer1.num_steps / trainer1.num_grad_acc_steps == trainer2.num_steps / trainer2.num_grad_acc_steps
    assert all((p1 - p2).abs().max().item() < 1e-4 for p1, p2 in zip(model1.parameters(), model2.parameters()))
    assert all((p1 - pb).abs().max().item() > 1e-4 for p1, pb in zip(model1.parameters(), params_backup))
