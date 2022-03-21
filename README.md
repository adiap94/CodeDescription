# Source Code Summarization with Pretrained CodeBert Model and Adversarial Data Augmentation 

This repo provides the code for reproducing the experiments on  python language from [CodeSearchNet](https://arxiv.org/abs/1909.09436) dataset for code document generation tasks with and without data augmentation.
We used the pretrained model - CodeBERT from microsoft that was first published on 2020 in this [paper](https://arxiv.org/abs/2002.08155). 
## Dependency

The required packages for this project you can find in requiremnts.txt file.


## Data Preprocess

We used the clean CodeSearchNet that is provided in the official CodeBert repository. 
You can download the data from this [website](https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h).
In addition we udapted  the code from the following [paper](https://arxiv.org/abs/2002.03043) 
by Goutham Ramakrishnan and Jordan Henkel.

the amount of examples for train,valid and test datasets in the original and augmented datasets.

| PL         | Training |  Dev   |  Test  |
| :--------- | :------: | :----: | :----: |
| Clean     | 251,820  | 13,914 | 14,918 |
| Rename Parameters        |   |  |  |
| Rename Local Variables         |   |   |   |
| Rename Fields       |   |   |  |
| Add Dead Code |   |   |   |



## Data Download

You can download dataset from the [website](https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h). Or use the following command.

```shell
pip install gdown
mkdir data data/code2nl
cd data/code2nl
gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h
unzip Cleaned_CodeSearchNet.zip
rm Cleaned_CodeSearchNet.zip
cd ../..
```

## Data Augmentation

In order to apply augmentaion on the dataset, you need to run the following script after providing the path to the data and path to the saved model after training. 
```shell
python preprocess_ADV_data.py
```


## Fine-Tune

We fine-tuned the model on 4*P40 GPUs. 

```shell
cd code2nl

lang=php #programming language
lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../data/code2nl/CodeSearchNet
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
eval_steps=1000 #400 for ruby, 600 for javascript, 1000 for others
train_steps=50000 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps 
```



## Inference and Evaluation

After fine-tuning, inference and evaluation are as follows:

```shell
lang=php #programming language
beam_size=10
batch_size=128
source_length=256
target_length=128
output_dir=model/$lang
data_dir=../data/code2nl/CodeSearchNet
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
```

The results of BLUE-4 score on the original CodeBERT with clean python dataset from CodeSearchNet, and pretained CodeBERT models on different Adversarial augmentations on the same dataset.

py CodeSearchNet are shown in this Table:


## Results

| Model       |   BLEU-4    |
| ----------- | :-------: |
| CodeBERT      |      |  
| CodeBERT and Rename Parameters  |      |  
| CodeBERT and Rename Local Variables     |     |     
| CodeBERT and Rename Fields   |   |
| CodeBERT and Add Dead Code   | |              

