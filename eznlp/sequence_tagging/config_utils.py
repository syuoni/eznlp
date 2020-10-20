# -*- coding: utf-8 -*-

def _default_encoder_config(arch):
    if arch.lower() == 'shortcut':
        return {'arch': arch}
    elif arch.lower() in ('lstm', 'gru'):
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


def _default_decoder_config(arch):
    return {'arch': arch, 
            'dropout': 0.5}


class ConfigHelper(object):
    @staticmethod
    def load_default_config(config, enc_arches=None, dec_arch=None, ptm=None):
        emb_config = config['emb']
        emb_config['tok'].update({'emb_dim': 100, 
                                  'max_len': 300, 
                                  'use_pos_emb': False})
        emb_config['char'].update({'arch': 'CNN', 
                                   'emb_dim': 25, 
                                   'out_dim': 32, 
                                   'kernel_size': 3, 
                                   'dropout': 0.5})
        if 'enum' in emb_config:
            for f, enum_config in emb_config['enum'].items():
                enum_config['emb_dim'] = 15 if enum_config['voc_dim'] < 100 else 30
        if 'val' in emb_config:
            for f, val_config in emb_config['val'].items():
                val_config['emb_dim'] = 20
            
            
        assert (enc_arches is not None) or (ptm is not None)
        if enc_arches is not None:
            config['enc'] = [_default_encoder_config(enc_arch) for enc_arch in enc_arches]
        if ptm is not None:
            config['pt_enc'] = {'hid_dim': ptm.config.hidden_size, 
                                'arch': 'PTM'}
            
            
        assert (dec_arch is not None)
        config['dec'] = _default_decoder_config(dec_arch)
        
        return ConfigHelper.update_dims(config)
    

    @staticmethod
    def update_dims(config):
        if 'emb' in config:
            emb_config = config['emb']
            tok_emb_dim = emb_config['tok']['emb_dim'] if 'tok' in emb_config else 0
            char_out_dim = emb_config['char']['out_dim'] if 'char' in emb_config else 0
            enum_full_dim = sum(enum_config['emb_dim'] for f, enum_config in emb_config['enum'].items()) if 'enum' in emb_config else 0
            val_full_dim = sum(val_config['emb_dim'] for f, val_config in emb_config['val'].items()) if 'val' in emb_config else 0
            emb_dim = tok_emb_dim + char_out_dim + enum_full_dim + val_full_dim
            config['emb']['emb_dim'] = emb_dim
            
            if 'enc' in config:
                for enc_config in config['enc']:
                    enc_config['emb_dim'] = emb_dim
                    if enc_config['arch'].lower() == 'shortcut':
                        enc_config['hid_dim'] = emb_dim
                        
        enc_hid_dim = sum(enc_config['hid_dim'] for enc_config in config['enc']) if 'enc' in config else 0
        pt_enc_hid_dim = config['pt_enc']['hid_dim'] if 'pt_enc' in config else 0
        config['dec']['hid_full_dim'] = enc_hid_dim + pt_enc_hid_dim
        return config
    

    @staticmethod
    def model_name(config, tag_helper):
        name_elements = []
        if 'enc' in config:
            name_elements.extend([enc_config['arch'] for enc_config in config['enc']])
            
        if 'pt_enc' in config:
            name_elements.append(config['pt_enc']['arch'])
            
        name_elements.append(config['dec']['arch'])
        
        if tag_helper.cascade_mode.lower() != 'none':
            name_elements.append(tag_helper.cascade_mode)
        
        return '-'.join(name_elements)
        
        
