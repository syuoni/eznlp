# -*- coding: utf-8 -*-
import glob
import re
import argparse
import logging
import pandas as pd


dict_re = re.compile("\{[^\{\}]+\}")
micro_f1_re = re.compile("(?<=Micro F1-score: )\d+\.\d+(?=%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='conll2003', 
                        help="dataset name")
    args = parser.parse_args()
    
    
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    logging_fns = glob.glob(f"cache/{args.dataset}-*/training.log")
    exp_results = []
    for logging_fn in logging_fns:
        with open(logging_fn) as f:
            log_text = f.read()
            
        try:
            exp_res = dict_re.search(log_text).group()
            exp_res = eval(exp_res)
            dev_f1, test_f1 = micro_f1_re.findall(log_text)
            exp_res['dev_f1'] = float(dev_f1)
            exp_res['test_f1'] = float(test_f1)
        except:
            logger.warning(f"Failed to parse {logging_fn}")
        else:
            exp_results.append(exp_res)
            
            
    df = pd.DataFrame(exp_results)
    df.to_excel(f"cache/{args.dataset}-collected.xlsx", index=False)
    