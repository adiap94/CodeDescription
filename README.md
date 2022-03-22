# Source Code Summarization with Pretrained CodeBert Model and Adversarial Data Augmentation 

This repo provides the code for reproducing the experiments on python language from [CodeSearchNet](https://arxiv.org/abs/1909.09436) dataset for code document generation tasks with and without data augmentation.
We used the pretrained model - CodeBERT from microsoft that was first published on 2020 in this [paper](https://arxiv.org/abs/2002.08155).
This repo is based on both [CodeBERT code](https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/code2nl) as architecture, BLEU code and main run script, and the [AVERLOC code](https://github.com/jjhenkel/averloc) as data augmentation script following the [paper](https://arxiv.org/abs/2002.03043) by Goutham Ramakrishnan and Jordan Henkel.

The full research is described in [our paper](Source Code Summarization with Pre-trained CodeBERT Model and Adversarial Data Augmentation.pdf) .
## Dependencies


We ran our project with python version 3.8 , torch=1.11.0, transformers=2.5.0
The full required packages for this project are listed in [requirements.txt](requirements.txt) file.

## Data Preprocess

### Data Preprocess - Download and unzip the "clean" data.
We used the clean CodeSearchNet that is provided in the official CodeBert repository. 


1. You can download dataset from the [website](https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h)
2. Unzip the zip files in the data directory of your choice.
3. Remove the zipped files

### Preprocess - apply data augmentation.

In order to apply augmentaion on the dataset, you need to run the following script ```preprocess_ADV_data.py```  after changing the path to the "clean" data python folder ```(path_to_data)```  and path to the trained model ```(path_to_model)```. 
After the run is complete, you get 4 different data augmentation sets.

## Data Distribution

the amount of examples for train,valid and test datasets in the original and augmented datasets.

| PL         | Training |  Dev   |  Test  |
| :--------- | :------: | :----: | :----: |
| Clean     | 251,820  | 13,914 | 14,918 |
| Add Dead Code |  171,471 | 9,507   |  9,985 |
| Rename Local Variables         | 135,097  |  7,432 | 7,751  |
| Rename Parameters        | 143,466  | 8,154 | 8,356 |
| Rename Fields       |  74,043 |  4,024 |  4,466 |



## Train

We trained the model on a single GPU. 

```shell
# change directory to your local code directory
cd /code-directory

lr=5e-5
batch_size=10
beam_size=10
source_length=256
target_length=128
output_dir =/project/results/python # change to your output directory of results 
train_file = /project/CodeSearchNet/python/train.jsonl # change to the training jsonl 
dev_file= /project/CodeSearchNet/python/valid.jsonl # change to your valid jsonl
eval_steps=1000
train_steps=50000 
pretrained_model=microsoft/codebert-base

python original_code.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps --define_gpu "0" 
```

if you would like to train the data augmented based models then you need to add the path to the train and dev augmented files.

For example:

```shell
python original_code.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --train_filename /project/CodeSearchNet/python/train.jsonl,/project/CodeSearchNet/python/adv/adv_20220318-215953/data_jsonl/train_new/test_transforms.RenameLocalVariables.jsonl --dev_filename /project/CodeSearchNet/python/valid.jsonl,/project/CodeSearchNet/python/adv/adv_20220318-215953/data_jsonl/valid_new/test_transforms.RenameLocalVariables.jsonl --output_dir /project/results/python/ --max_source_length 256 --max_target_length 128 --beam_size 10 --train_batch_size 10 --eval_batch_size 10 --learning_rate 5e-5 --train_steps 50000 --eval_steps 1000 --define_gpu '0' 
 
```

## Test

After training is complete, you can :

```shell
beam_size=10
batch_size=128
source_length=256
target_length=128
output_dir=model/$lang
test_file=/project/CodeSearchNet/python/test.jsonl
test_model=/project/results/python//checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python original_code.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
```

For example:

```shell
python original_code.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path /project/results/python/20220319-170147/checkpoint-best-bleu/pytorch_model.bin --test_filename /project/CodeSearchNet/python/test.jsonl --output_dir /project/results/python/20220319-170147 --max_source_length 256 --max_target_length 128 --beam_size 10 --eval_batch_size 10 --define_gpu "0" 
 
```
When the test is complete you get a file name BleuTestScore.csv in the output directory with the BLEU-4 score.  


## Results

The results of BLUE-4 score on the original CodeBERT with clean python dataset from CodeSearchNet, and pretained CodeBERT models on different Adversarial augmentations on the same dataset.

py CodeSearchNet are shown in this Table:


| Model       |   BLEU-4    |
| ----------- | :-------: |
| CodeBERT original     |  19.06    |  
| CodeBERT restored     |   18.13   | 
| CodeBERT and Add Dead Code   | 12.30 | 
| CodeBERT and Rename Local Variables     |   19.16  |    
| CodeBERT and Rename Parameters  |   19.18   |
| CodeBERT and Rename Fields   | 19.01  |
            
## Authors

Adi Apotheker Tovi, a MSc student for Biomedical Engineering ,The Technion, Israel

Yevgenia Shteynman, a MSc student for Biomedical Engineering ,The Technion, Israel