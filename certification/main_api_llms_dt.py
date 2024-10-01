import sys 
sys.path.append('.')
import utils_api as utils
import pandas as pd
import time
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
import argparse 
import numpy as np
import common
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run certification on decoding trust dataset')
parser.add_argument('expt_name', type=str, help='name of the experiment')
parser.add_argument('expt_mode', type=str, help='mode: common jb or unknown jb') # no soft jb for now
parser.add_argument('model_name', type=str, help='name of the model to be used for certification')
args = parser.parse_args()
    
# hyperparameters 
batch_size = 10 # 2*batch_size is the number of completions per batch
cp_alpha = 0.05 # 95% confidence interval
ATTEMPTS = 50

lbs = []
ubs = []
tokenizer, impossible_tokens = utils.load_tokenizer(model_name=args.model_name)

print('*'*100)
with open(f'results/{args.expt_name}.txt', 'r') as fp:
    present_output = fp.read()
    
df_dt = pd.read_csv('data/decoding_trust_templates.csv')
    
for ts in tqdm(range(df_dt.shape[0])):
    c = df_dt.iloc[ts]['stereotype_template']
    
    print('Starting certification for:', c)
    
    _biased_total = 0
    attempts_total = 0
    t1 = time.time()
    completions_black = []
    completions_white = []
    sents = []
    _bias_scores = []
    
    if f'Average bounds for {c}:' in present_output:
        bounds = present_output.split(f'Average bounds for {c}: ',1)[1].split('\n')[0]
        lbs.append(float(bounds.split(',')[0][1:]))
        ubs.append(float(bounds.split(',')[1][:-1]))
        continue
    
    obtained_sents = []
    obtained_comp_man = []
    obtained_comp_woman = []
    
    pointer = 0 
    while True:
        if pointer+batch_size > len(obtained_sents):
            fairness_prompts = utils.make_prefix_prompts_dt(text=c, batch_size=batch_size, tokenizer=tokenizer, model_name=args.model_name, impossible_tokens=impossible_tokens, mode=args.expt_mode, prefix_length=100, mag=0.04)
            comp = utils.query_model(fairness_prompts, model_name=args.model_name)
        else:
            fairness_prompts = []
            comp = []
            for i in range(batch_size):
                fairness_prompts.extend([obtained_sents[pointer+i], obtained_sents[pointer+i]])
                comp.extend([obtained_comp_man[pointer+i], obtained_comp_woman[pointer+i]])
            pointer += batch_size    
        
        for i in range(batch_size):
            completions_black.append(comp[2*i])
            completions_white.append(comp[2*i+1])
            sents.append(fairness_prompts[2*i])
            _my_bias = common.get_agreement([comp[2*i], comp[2*i+1]])
            _bias_scores.append(_my_bias)
            _biased = _my_bias
            _biased_total += int(_biased)
        attempts_total += batch_size
        _unbiased_total = attempts_total - _biased_total
        print('attempts:', attempts_total, 'biased:', _biased_total)
        if attempts_total < ATTEMPTS:
            continue
        verif_prob_unbiased = proportion_confint(_unbiased_total, attempts_total, alpha=cp_alpha, method="beta")
        print("probability bounds for unbiased: ", verif_prob_unbiased)
        if attempts_total >= ATTEMPTS:
            print(f"time taken for {c}:", time.time() - t1)
            lbs.append(verif_prob_unbiased[0])
            ubs.append(verif_prob_unbiased[1])
            break


    df = pd.DataFrame({
        'partial sentence': sents,
        'completion_black':completions_black, 
        'completion_white':completions_white, 
        'Bias score':_bias_scores,})
    
    if ts == 0:
        df.to_csv(f'results/{args.expt_name}.csv', index=False)
    else:
        df.to_csv(f'results/{args.expt_name}.csv', index=False, mode='a') # save results to csv file    
    print('-'*100)
print(f'Average bounds for {c}: ({np.mean(lbs)},{np.mean(ubs)})')
print('*'*100)