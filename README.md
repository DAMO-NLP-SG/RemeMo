# RemeMo

This repo contains the resources used in our paper: [Once Upon a *Time* in *Graph*: Relative-Time Pretraining for Complex Temporal Reasoning](https://arxiv.org/abs/2310.14709) (*EMNLP 2023*).

Checkpoints hosted on 🤗 HuggingFace: 
- Pre-trained RemeMo 👉 [[rememo-base](https://huggingface.co/DAMO-NLP-SG/rememo-base)] [[rememo-large](https://huggingface.co/DAMO-NLP-SG/rememo-large)]
- Fine-tuned time expression extractor 👉 [[roberta-time_identification](https://huggingface.co/DAMO-NLP-SG/roberta-time_identification)]

## Table of Contents

- [Overview](https://github.com/DAMO-NLP-SG/RemeMo#overview)
- [Requirements](https://github.com/DAMO-NLP-SG/RemeMo#requirements)
- [Usage](https://github.com/DAMO-NLP-SG/RemeMo#usage)
- [Repo Structure](https://github.com/DAMO-NLP-SG/RemeMo#repo-structure)
- [Checkpoints](https://github.com/DAMO-NLP-SG/RemeMo#checkpoints)
- [Citation](https://github.com/DAMO-NLP-SG/RemeMo#citation)
- [Acknowledgements](https://github.com/DAMO-NLP-SG/RemeMo#acknowledgments)

## Overview
![rememo_example](https://github.com/DAMO-NLP-SG/RemeMo/assets/18526640/6d1af421-11f7-4ded-9cbd-342316bd5c43)

- **What Is RemeMo?** RemeMo is an improved T5-based language model, which gains better complex temporal reasoning abilities through pre-training using a novel time-relation-prediction (TRC) objective. 
As shown in the figure above, the time relation between any pair of facts is adopted as the TRC pre-training label. The complex temporal dependencies among all facts are thus modeled within a fully-connected directed graph.

- **When to Use RemeMo?** RemeMo is recommended to be used as a replacement for T5 (or other seq2seq models) in downstream tasks that require complex temporal reasoning, e.g., temporal question answering.

## Environment Setup

```
conda create -n rememo python=3.8
conda activate rememo

git clone https://github.com/DAMO-NLP-SG/RemeMo.git
cd RemeMo
pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html
pip install -r requirements.txt
```

## Usage

### Fine-tune

1. Customize the configurations in `src/odqa_t5/run_finetuning.sh` ;
2. Run 
    ```
    mkdir src/odqa_t5/log
    sh src/odqa_t5/run_finetuning.sh
    ```

### Time-identification

1. To run inference:
    
    see `src/time_expression/inference_time_identification.sh` .
    
2. To train a new model:
    
    see `src/time_expression/train_time_identification_model.sh` .
    

### Pre-train

1. Customize  the configurations in `src/pretran_t5/run_pretrain.sh` :
    - `NSP_MODE`: choices include { `mlm` (T5+LM), `mlm_trelation` (RemeMo) }.
    - See `data/pretrain` for examples of the pre-training data. Prepare your own pre-training data following the same format.
    - Modify other arguments if needed.
2. Run
    ```
    mkdir src/pretrain_t5/log
    sh src/pretrain_t5/run_pretrain.sh
    ```

## Repo Structure

- Code:
    - `src/time_expression`: time-identification;
    - `src/preprocess`: analysis of the pre-processing pipeline;
    - `src/pretrain_t5`: pre-training code;
    - `src/odqa_t5`: fine-tuning code;
- Data:
    - `data/time_expression/ner_task`: obtained by pre-processing TimeBank & adopted for training the roberta-time_identification model;
    - `data/pretrain`: examples of the pre-training data;
    - `data/finetune`: temporal question answering data;

- Checkpoints:
    - `model_checkpoints/time_expression`: to store the roberta-time_identification model checkpoint;
    - `model_checkpoints/rememo_ckpt`: to store RemeMo-{base/large} model checkpoints;

## Checkpoints

| Model | Size (# parameters) | 🤗 Link |
|----------|----------|----------|
| rememo-base| ~250M | [DAMO-NLP-SG/rememo-base](https://huggingface.co/DAMO-NLP-SG/rememo-base) |
| rememo-large| ~800M | [DAMO-NLP-SG/rememo-large](https://huggingface.co/DAMO-NLP-SG/rememo-large) |


## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@inproceedings{yang-etal-2023-once,
    title = "Once Upon a $\textit{Time}$ in $\textit{Graph}$: Relative-Time Pretraining for Complex Temporal Reasoning",
    author = "Yang, Sen  and
      Li, Xin and
      Bing, Lidong and
      Lam, Wai",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
}
```

## Acknowledgments

This project uses the code from:
- [HuggingFace Transformers](https://github.com/huggingface/transformers/)
- [timex-normaliser](https://github.com/filannim/timex-normaliser)
