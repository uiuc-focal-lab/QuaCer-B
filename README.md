# Quantitative Certification of Bias in Large Language Models

## Introduction
QCB is a framework for quantitatively certifying the bias in large language models (LLMs).

## Setup
To setup QCB, please set up a conda environment with the following command:
```conda env create -f environment.yml```

Then, activate the environment with the following command:
```conda activate qcb```

Add the API keys for closed-source models in the file ```api_keys.py```.

## Certifying supported models
We currently support the following models. We are working towards extending to models. 
- Open-source models (from Huggingface):
  - Vicuna
  - Llama-2
  - Mistral
- Closed-source models (with API access):
  - Gemini-Pro
  - GPT

To certify an open-source model for the BOLD dataset, run the following command, by replacing the placeholders with the appropriate values:
```python certification/main_hf_llms_bold.py <expt_name> <expt_mode> <model_name>```
The arguments to the above command are described next:
- ```<expt_name>```: Name of the experiment (to name the result files appropriately)
- ```<expt_mode>```: Indicates the prefix distribution wrt which certification is to be done. Possible values are: ```common jb``` or ```unknown jb``` or ```soft jb``` (```soft jb``` is only for the open-source models)
- ```<model_name>```: Name of the model to be certified. Use the official names of the models, as given in the Huggingface model hub (for open-source models) or the websites of the API models to query them. 

The settings of the certification experiments can be modified by varying the Python script invoked for certification. 
The scripts are named as ```main_<type of LLM>_<dataset>```. The ```<type of LLM>``` for open-source, Huggingface models is ```hf_llms``` and for closed-source models is ```api_llms```. The ```<dataset>``` can be ```bold``` for the BOLD dataset or ```dt``` for the Decoding Trust's stereotype dataset. The arguments to the certification scripts remain the same as described above.
