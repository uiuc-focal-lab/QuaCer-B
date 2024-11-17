import sys
sys.path.append('.')
import utils 
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
import argparse 
import common
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Run certification on BOLD dataset')
parser.add_argument('expt_name', type=str, help='name of the experiment')
parser.add_argument('expt_mode', type=str, help='mode: common jb or unknown jb or soft jb')
parser.add_argument('model_name', type=str, help='name of the model to be used for certification')
parser.add_argument('--topk', type=int, default=10, help='temperature for sampling')
args = parser.parse_args()

if args.expt_mode == 'common jb' or args.expt_mode == 'unknown jb':
    prompt_fun = utils.make_prefix_prompts
elif args.expt_mode == 'soft jb':
    prompt_fun = utils.make_prefix_soft_prompts
else:
    raise ValueError('Invalid expt_mode')

# hyperparameters 
batch_size = 10 # 2*batch_size is the number of completions per batch
cp_alpha = 0.05 # 95% confidence interval
ATTEMPTS = 50

lbs = []
ubs = []
tokenizer, model = utils.load_model(model_name=args.model_name)

print('*'*100)
with open(f'results/{args.expt_name}.txt', 'r') as fp:
    present_output = fp.read()

with open('data/ablation_test_set.txt', 'r') as fp:
    test_set = fp.read().split('*'*100+'\n')
    test_set = [t.split('\n') for t in test_set if t != '']

iu = -1
for ts in tqdm(test_set):
    iu += 1
    c = ts[0]
    my_pool = ts[1]
    my_p = ts[2]
    
    if f'Average bounds for {c}, {my_pool} and {my_p}:' in present_output:
        bounds = present_output.split(f'Average bounds for {c}, {my_pool} and {my_p}: ',1)[1].split('\n')[0].strip()
        lbs.append(float(bounds.split(',')[0][1:]))
        ubs.append(float(bounds.split(',')[1][:-1]))
        continue
    
    # useful for extending to reading some pre-obtained completions
    obtained_sents = []
    obtained_comp_man = []
    obtained_comp_woman = []
        
    regard_biased_total = 0
    attempts_total = 0
    t1 = time.time()
    completions_man = []
    completions_woman = []
    sents = []
    regard_bias_scores = []
    profs = []
    profs_broad = []
    
    pointer = 0 
    while True:
        if pointer+batch_size > len(obtained_sents):
            fairness_prompts = prompt_fun(my_p, my_pool, batch_size, tokenizer, model, args.model_name, mode=args.expt_mode, prefix_length=100, mag=0.04)
            if args.topk == 1:
                do_sample=False
            else:
                do_sample=True
            comp = utils.query_model(fairness_prompts, model, tokenizer, args.model_name, mode=args.expt_mode, top_k=args.topk, do_sample=do_sample)
        else:
            fairness_prompts = []
            comp = []
            for i in range(batch_size):
                fairness_prompts.extend([obtained_sents[pointer+i], obtained_sents[pointer+i]])
                comp.extend([obtained_comp_man[pointer+i], obtained_comp_woman[pointer+i]])
            pointer += batch_size
            
        for i in range(batch_size):
            completions_man.append(comp[2*i])
            completions_woman.append(comp[2*i+1])
            if args.expt_mode == 'soft jb':
                sents.append(my_p)
            else:
                sents.append(fairness_prompts[2*i])
            profs_broad.append(c)
            profs.append(my_pool)
            regard_my_bias = common.get_bias_score_regard([comp[2*i], comp[2*i+1], my_p])
            regard_bias_scores.append(regard_my_bias)
            regard_biased_total += int(regard_my_bias)
        attempts_total += batch_size
        print('attempts:', attempts_total, 'regard biased:', regard_biased_total)
        regard_unbiased_total = attempts_total - regard_biased_total
        if attempts_total < ATTEMPTS:
            continue
        verif_prob_unbiased = proportion_confint(regard_unbiased_total, attempts_total, alpha=cp_alpha, method="beta")
        print("probability bounds for unbiased: ", verif_prob_unbiased)
        if attempts_total >= ATTEMPTS:
            print(f"Average bounds for {c}, {my_pool} and {my_p}: ", verif_prob_unbiased)
            print(f"time taken for {c}, {my_pool} and {my_p}:", time.time() - t1)
            lbs.append(verif_prob_unbiased[0])
            ubs.append(verif_prob_unbiased[1])
            break


    df = pd.DataFrame({
        'profession types':profs_broad, 
        'profession':profs, 
        'partial sentence': sents,
        'completion_man':completions_man, 
        'completion_woman':completions_woman, 
        'regard bias score':regard_bias_scores})
    
    if iu == 0:
        df.to_csv(f'results/{args.expt_name}.csv', index=False)
    else:
        df.to_csv(f'results/{args.expt_name}.csv', index=False, mode='a') # save results to csv file    
    print('-'*100)
print(f'Average bounds: ({np.mean(lbs)},{np.mean(ubs)})')
print('*'*100)