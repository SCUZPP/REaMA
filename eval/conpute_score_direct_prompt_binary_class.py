import json 
import re
import argparse

import json
def run_simple_tsv_eval(in_gold_tsv_file, labels):
    tp = 0.
    fp = 0.
    fn = 0.
    with open(in_gold_tsv_file, "r") as fin:
        for i, line in enumerate(fin):
            data = json.loads(line)
            pred_label = data["pre"]
            if pred_label in labels:
                pred_label = labels[pred_label]
            _gold = data["label"]
            if _gold in labels:
                _gold = labels[_gold]

            if not pred_label.startswith('None'):
                pred_label = 'Association'
            if not _gold.startswith('None'):
                _gold = 'Association'
                
            if pred_label.startswith('None'):
                pred_label = 'None'
            if _gold.startswith('None'):
                _gold = 'None'           

            if pred_label == _gold:
                if not pred_label.startswith('None'):
                    tp += 1
            else:
                if not pred_label.startswith('None'):
                    if not _gold.startswith('None'):
                        fp += 1
                        fn += 1
                    else:
                        fp += 1
                else:
                    if not _gold.startswith('None'):
                        fn += 1
                    
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0.
        recall = tp/ (tp+fn) if (tp+fn) > 0 else 0.
        f = (2*prec*recall) / (prec+recall) if (prec+recall) > 0 else 0.
        #print(tp, fp, fn)
        print(prec, recall, f)


        
def compute(args):

    labels = {"1": "Association", "0": "None", "false": "None", "False": "None"}

    in_gold_tsv_file = args.data_dir
    run_simple_tsv_eval(in_gold_tsv_file, labels)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../results/gad/llama2_13b/predictions.jsonl")
    args = parser.parse_args()
    compute(args)
    