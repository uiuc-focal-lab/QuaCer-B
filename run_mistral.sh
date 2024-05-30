echo "starting with expt on Mistral-7b-instruct-v0.2 unk jb"
python3 certification/main_hf_llms_bold.py 'unknown_jb_mistral_7b_bold' 'unknown jb' --model_name 'mistralai/Mistral-7B-Instruct-v0.2' >> results/unknown_jb_mistral_7b_bold.txt

echo "finished"