import json 
import re
import argparse


def remove_non_alpha_from_ends(text):
    # 正则表达式匹配字符串开始和结束的非字母字符
    # ^[^a-zA-Z]+ 匹配开头的非字母字符，[^a-zA-Z]+$ 匹配结束的非字母字符
    pattern = r'^[^a-zA-Z]+|[^a-zA-Z]+$'
    # 使用空字符串替换匹配到的非字母字符
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text  


def extract_ans(pre):
    ans_model = pre.lower().split('\n')
    #print('ans_model', ans_model)
    ans = []
    for li, al in enumerate(ans_model):
        ans.append(al)
        if('\ntherefore, the relation between ' in al):
            break
    residual = list(ans_model[li + 1:])
    pred_str = '\n'.join(ans)
    #print('pred_str', pred_str)
    pattern = pred_str.split('\ntherefore, the relation between ')[-1].split("is ")[-1]
    pattern = remove_non_alpha_from_ends(pattern)
    #print('pattern', pattern)
    return pattern
    
    
def run_simple_tsv_eval(in_gold_tsv_file, labels):
    tp = 0.
    fp = 0.
    fn = 0.
    with open(in_gold_tsv_file, "r") as fin:
        for i, line in enumerate(fin):
            #if i != 3:
            #    continue
            data = json.loads(line)
            
            pred_label = extract_ans(data["pre"]).lower()
            
            if pred_label in labels:
                pred_label = labels[pred_label]
            _gold = data["label"].lower()
            if _gold in labels:
                _gold = labels[_gold]

            if not pred_label.startswith('none'):
                pred_label = 'association'
            if not _gold.startswith('none'):
                _gold = 'association'
                
            if pred_label.startswith('none'):
                pred_label = 'none'
            if _gold.startswith('none'):
                _gold = 'none'           

            if pred_label == _gold:
                #print("pred_label", i, pred_label, _gold, line)
                if not pred_label.startswith('none'):
                    #print("pred_label", i, pred_label, _gold, line)
                    tp += 1
            else:
                #print("pred_label", i, pred_label, _gold)
                if not pred_label.startswith('none'):
                    if not _gold.startswith('none'):
                        fp += 1
                        fn += 1
                    else:
                        fp += 1
                else:
                    if not _gold.startswith('none'):
                        fn += 1
                    
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0.
        recall = tp/ (tp+fn) if (tp+fn) > 0 else 0.
        f = (2*prec*recall) / (prec+recall) if (prec+recall) > 0 else 0.
        #print(tp, fp, fn)
        print(prec, recall, f)

        
def run_simple_tsv_eval_mlt(in_gold_tsv_file, labels):
    tp = 0.
    fp = 0.
    fn = 0.
    with open(in_gold_tsv_file, "r") as fin:
        for i, line in enumerate(fin):
            #if i > 100:
            #    break
            data = json.loads(line)
            pred_label = extract_ans(data["pre"])
            #print("pred_label", pred_label)
            if pred_label in labels:
                pred_label = labels[pred_label] 
                
            _gold = data["label"]
            if _gold in labels:
                _gold = labels[_gold]  
                #print(i, _gold, pred_label)
                #break
            _gold = _gold.lower()
            pred_label = pred_label.lower()
            #if not pred_label.startswith('None'):
            #    pred_label = 'Association'
            #if not _gold.startswith('None'):
            #    _gold = 'Association'
                
            #if pred_label.startswith('None'):
            #    pred_label = 'None'
            #if _gold.startswith('None'):
            #    _gold = 'None'           
            #print(i, pred_label, _gold)
            if pred_label == _gold:
                #print(i, pred_label, _gold)
                if not pred_label.startswith('none'):
                    tp += 1
                #if tp > 10:
                #    break
            else:
                #print("pred_label", i, pred_label, _gold)
                if not pred_label.startswith('none'):
                    if not _gold.startswith('none'):
                        fp += 1
                        fn += 1
                    else:
                        fp += 1
                else:
                    if not _gold.startswith('none'):
                        fn += 1
                    
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0.
        recall = tp/ (tp+fn) if (tp+fn) > 0 else 0.
        f = (2*prec*recall) / (prec+recall) if (prec+recall) > 0 else 0.
        #print(tp, fp, fn)
        print(prec, recall, f)

        
def compute_multi_class(args):

    labels = {"false": "None", "False": "None", "CPR:3": "Activation", "CPR:4": "Inhibition", 
              "CPR:5": "Agonist", "CPR:6": "Antagonist", "CPR:9": "Substrate",
              "None-CID": "None", "None-DDI": "None", "None-BC7": "None", 
              "None-dis": "None", "None-nary": "None", "None-Chem": "None", 
              "None-HPRD50": "None", "None-AIMED": "None", "0": "None", "1": "Association", 
              "2": "Bind", "3": "Comparison", "4": "Conversion", "5": "Cotreatment", 
              "6": "Drug_Interaction", "7": "Negative_Correlation", "8": "Positive_Correlation"}

    in_gold_tsv_file = args.data_dir
    run_simple_tsv_eval_mlt(in_gold_tsv_file, labels)
    
    
    
def compute_binary_class(args):

    labels = {"1": "association", "0": "none", "false": "none", "none-nary": "none"}
    in_gold_tsv_file = args.data_dir
    run_simple_tsv_eval(in_gold_tsv_file, labels)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../results/chemprot/llama2_13b/predictions.jsonl")
    args = parser.parse_args()
    compute_multi_class(args)
    