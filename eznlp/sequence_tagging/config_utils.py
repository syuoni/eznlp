# -*- coding: utf-8 -*-

def _default_encoder_config(arch, demo=False):
    if arch.lower() in ('lstm', 'gru'):
        return {'arch': arch, 
                'hid_dim': 128,
                'n_layers': 2, 
                'dropout': 0.5}
    elif arch.lower() == 'cnn':
        return {'arch': arch, 
                'hid_dim': 128,
                'kernel_size': 3,
                'n_layers': 3, 
                'dropout': 0.25}
    elif arch.lower() == 'transformer':
        return {'arch': arch, 
                'hid_dim': 128,
                'nhead': 8,
                'pf_dim': 256, 
                'n_layers': 3,
                'dropout': 0.1}
    else:
        raise ValueError(f"Invalid encoder architecture {arch}")


def _default_decoder_config(arch, demo=False):
    return {'arch': arch, 
            'dropout': 0.5}


class ConfigHelper(object):
    @staticmethod
    def load_default_config(config, enc_arch=None, dec_arch=None, bert=None, demo=False):
        config['tok'].update({'emb_dim': 100})
        config['max_len'] = 300
        
        config['char'].update({'emb_dim': 25, 
                               'out_dim': 32, 
                               'kernel_size': 3, 
                               'dropout': 0.5})
        for f, enum_config in config['enum'].items():
            enum_config['emb_dim'] = 15 if enum_config['voc_dim'] < 100 else 30
        for f, val_config in config['val'].items():
            val_config['emb_dim'] = 20
            
        assert (enc_arch is not None) or (bert is not None)
        if enc_arch is not None:
            config['enc'] = _default_encoder_config(enc_arch, demo=demo)
        if bert is not None:
            config['bert'] = {'hid_dim': bert.config.hidden_size}
            
        assert (dec_arch is not None)
        config['dec'] = _default_decoder_config(dec_arch, demo=demo)
        config['shortcut'] = False
        
        return ConfigHelper.update_full_dims(config)


    def update_full_dims(config):
        tok_emb_dim = config['tok']['emb_dim'] if 'tok' in config else 0
        char_out_dim = config['char']['out_dim'] if 'char' in config else 0
        enum_full_dim = sum(enum_config['emb_dim'] for f, enum_config in config['enum'].items()) if 'enum' in config else 0
        val_full_dim = sum(val_config['emb_dim'] for f, val_config in config['val'].items()) if 'val' in config else 0
        config['emb_full_dim'] = tok_emb_dim + char_out_dim + enum_full_dim + val_full_dim
        
        enc_hid_dim = config['enc']['hid_dim'] if 'enc' in config else 0
        bert_hid_dim = config['bert']['hid_dim'] if 'bert' in config else 0
        config['hid_full_dim'] = enc_hid_dim + bert_hid_dim
        if config['shortcut']:
            config['hid_full_dim'] += config['emb_full_dim']
        return config
