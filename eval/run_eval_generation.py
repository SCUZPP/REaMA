import argparse
import os
import re
import json
import random
import evaluate
from eval.utils import generate_completions, load_hf_lm_and_tokenizer, query_openai_chat_model
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

#exact_match = evaluate.load("exact_match", module_type='metric')
#assert exact_match.module_type == 'metric', "wrong exact_match module loaded, expected to be 'metric', get '{exact_match.module_type}'"
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def find_entity(pattern, text):
    #pattern = r"@GeneOrGeneProductSrc$(.*?)@GeneOrGeneProductSrc$"
    
    # 使用正则表达式搜索文本
    match = re.search(pattern, text)
    content_between = None
    # 检查是否有匹配项，并提取内容
    if match:
        content_between = match.group(1).strip()
        #print("Found content:", content_between)
    #print('\n\ntext', content_between)
    return content_between
    
    
def extract_answer_number(completion):
    text = completion.split('\n#### ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None
    
def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            #answer = example["label"].lower()
            
            test_data.append(example)
        

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
        

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    options = "\nA. " + "None" + "\nB. " + "Association" + "\nC. " + "Bind" + "\nD. " + "Comparison" + "\nE. " + "Conversion" + "\nF. " + "Cotreatment" + "\nG. " + "Drug_Interaction" + "\nH. " + "Negative_Correlation" + "\nI. " + "Positive_Correlation" 
    global GSM_EXAMPLARS
    if args.n_shot:
        if len(GSM_EXAMPLARS) > args.n_shot:
            GSM_EXAMPLARS = random.sample(GSM_EXAMPLARS, args.n_shot)
        demonstrations = []
        for example in GSM_EXAMPLARS:
            if args.no_cot:
                demonstrations.append(
                    "Input: " + example["content"] + options + "\n" + "Answer: " + example["label"]
                )
            else:
                demonstrations.append(
                    "Input: " + example["content"] + options + "\n" + "Answer: " + example["label"]
                )
        prompt_prefix = "The following is a text annoted with two entities from the biomedical literature. Please select the one from A, B, C, D, E, F, G, H, and I that best fits in the blank space to correctly describe the relationship between the two entities.\n\n" + "\n\n".join(demonstrations) + "\n\n"

    else:
        prompt_prefix = "The following is a text annotated with two entities from the biomedical literature. Please select the one from 'None, Association, Bind, Comparison, Conversion, Cotreatment, Drug_Interaction, Negative_Correlation, Positive_Correlation' that correctly describe the relationship between the two entities.\n\n"
                
    prompt_prefix = "The following is a text annotated with two entities from the biomedical literature. Please select the one from 'None, Association, Bind, Comparison, Conversion, Cotreatment, Drug_Interaction, Negative_Correlation, Positive_Correlation' that correctly describe the relationship between the two entities.\n\n"
    prompts = []
    count = 0
    for index, example in enumerate(test_data):

        type1 = example["entity_name1"]
        type2 = example["entity_name2"]
        
        cloze = "\nThe relation between " + type1 + " and " + type2 + " is"
        if args.use_chat_format:
            prompt = "<|user|>\n" + prompt_prefix + "Input: " + example["content"].strip() + cloze + "\nAnswer:" + "\n<|assistant|>\n" 
        else:
            prompt = prompt_prefix + "Input: " + example["content"].strip() + "\nAnswer:"
        prompts.append(prompt)
    #print('prompts', prompts)


    tp_size = args.eval_batch_size   
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop = ["\nInput", "USER:", "USER", "ASSISTANT:", "ASSISTANT"])
    model_name = args.model_name_or_path
    llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.8, tensor_parallel_size=tp_size)
    outputs = llm.generate(prompts, sampling_params)
    generated_text = [output.outputs[0].text for output in outputs]
    outputs = generated_text[:]
    
        
    print("Calculating accuracy...")
    targets = [example["label"] for example in test_data]
    #gsm8k_ins = [example["question"] for example in test_data]
    print(f'predictions: {len(outputs)}')
    print(f'targets: {len(targets)}')
    #print(predictions)
    #print(targets)
    predictions_res = []
    result = []
    #em_score = exact_match._compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    for idx, (prompt, output) in enumerate(zip(test_data, outputs)):
        prompt["pre"] = output
        predictions_res.append(prompt)
            
    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions_res:
            fout.write(json.dumps(prediction) + "\n") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mgsm")
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str, default="results/mgsm")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="if specified, we will load the tokenizer from here.")
    parser.add_argument("--openai_engine", type=str, default=None, help="if specified, we will use the OpenAI API to generate the predictions.")
    parser.add_argument("--n_shot", type=int, default=0, help="max number of examples to use for demonstration.")
    parser.add_argument("--no_cot", action="store_true", help="If given, we're evaluating a model without chain-of-thought.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--use_chat_format", action="store_true", help="If given, the prompt will be encoded as a chat format with the roles in prompt.")
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
