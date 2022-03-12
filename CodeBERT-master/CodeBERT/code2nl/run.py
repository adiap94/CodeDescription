# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import time
import os
import sys
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
import pandas as pd
from itertools import cycle
import torch.nn as nn
from utiles import save_params
from model import Seq2Seq
from tqdm import tqdm, trange
from data_loader import CodeDataset , read_examples
from training import train_class
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
DEBUG_MODE = False



def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--debug_mode", action='store_true',
                        help="debug_mode for fast sanity check pre full run")
    parser.add_argument("--delete_token", action='store_true',
                        help="delete_token in training")
    parser.add_argument("--test_while_training", action='store_true',
                        help="Whether to run test while training, whenever model is approved.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--define_gpu", default='0', type=str,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    #debug mode
    if args.debug_mode:
        global DEBUG_MODE
        DEBUG_MODE = True


    # Setup CUDA, GPU & distributed training
    if args.define_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.define_gpu
        # device = torch.device("cuda:0")
        # args.n_gpu = 1
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        # model.load_state_dict(torch.load(args.load_model_path))
        model = torch.load(args.load_model_path)
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)




    if args.do_train:
        if DEBUG_MODE:
            args.output_dir = os.path.join(args.output_dir,"debug", time_str)
        else:
            args.output_dir = os.path.join(args.output_dir,time_str)

        print("out dir is: "+ args.output_dir)
        os.makedirs(args.output_dir,exist_ok=True)
        os.makedirs(os.path.join(args.output_dir,"Models"),exist_ok=True)
        save_params(args=args, out_dir=args.output_dir)
        # csvLoggerFile_path = os.path.join(args.output_dir, "history.csv")

        # Prepare training data loader
        print("loading training data")
        train_dataset = CodeDataset(args=args,tokenizer=tokenizer,split = "train")
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,num_workers=5)
        num_train_optimization_steps =  args.train_steps

        # Start training
        logger.info("***** Info training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size // len(train_dataset))

        # create eval dataloader
        print("loading validation data")
        eval_dataset = CodeDataset(args=args,tokenizer=tokenizer,split = "dev")
        eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,num_workers=5)

        logger.info("\n***** Info evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        train_object=train_class(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=eval_loader,
         epoch_num=num_train_optimization_steps , out_dir=args.output_dir, device=device,scheduler=scheduler, tokenizer = tokenizer
                                 ,path_to_dev = args.dev_filename)
        train_object.train()

    if args.do_test:
        test(args, tokenizer, model, device)


def test(args,tokenizer,model,device):
    # make sure results are saved inside run folder
    args.output_dir = os.path.dirname(os.path.dirname(args.load_model_path))

    files = []
    # if args.dev_filename is not None:
    #     files.append(args.dev_filename)
    if args.test_filename is not None:
        files.append(args.test_filename)

    for idx, file in enumerate(files):
        logger.info("Test file: {}".format(file))
        # create eval dataloader
        print("loading testing data")
        eval_dataset = CodeDataset(args=args,tokenizer=tokenizer,split = "test")
        eval_loader = DataLoader(eval_dataset, shuffle=False,num_workers=5)

        model.eval()
        p = []
        for val_data in eval_loader:

            source_ids_val, source_mask_val, target_ids_val, target_mask_val = (
                val_data["source_ids"].to(device),
                val_data["source_mask"].to(device),
                val_data["target_ids"].to(device),
                val_data["target_mask"].to(device),
                )
            model.eval()
            # p = []

            with torch.no_grad():
                preds = model(source_ids=source_ids_val, source_mask=source_mask_val)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
        eval_examples = read_examples(file, debug_mode=DEBUG_MODE)
        predictions = []
        with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(
                os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
            for ref, gold in zip(p, eval_examples):
                predictions.append(str(gold.idx) + '\t' + ref)
                f.write(str(gold.idx) + '\t' + ref + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target + '\n')

        (goldMap, predictionMap) = bleu.computeMaps(predictions,
                                                    os.path.join(args.output_dir, "test_{}.gold".format(idx)))
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
        logger.info("  " + "*" * 20)


def writeCSVLoggerFile(csvLoggerFile_path,epoch_log):
    df = pd.DataFrame([epoch_log])
    with open(csvLoggerFile_path, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
if __name__ == "__main__":
    main()


