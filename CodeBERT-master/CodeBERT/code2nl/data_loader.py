import pandas as pd
import random
import shelve
import torch
import json
from torch.utils.data import Dataset
import logging
import numpy as np



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename,debug_mode=False):
    """Read examples from filename."""
    # num = sum(1 for line in open(filename))
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if debug_mode:
                if idx == 20:
                    break
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features



class CodeDataset(Dataset):

    def __init__(self,args,tokenizer,split):

        self.args = args
        self.tokenizer = tokenizer
        self.debugBool=args.debug_mode
        self.token_delete_bool = True
        self.split=split

        self._load_entire_data()


    def __len__(self):
        return len(self.all_target_ids)

    def __getitem__(self, idx):
        print(idx)
        #
        source_ids = self.all_source_ids[idx]
        source_mask = self.all_source_mask[idx]
        target_ids = self.all_target_ids[idx]
        target_mask = self.all_target_mask[idx]

        if self.token_delete_bool and self.split=="train":
            target_ids = self.delete_random_token(target_ids)
            
        return {"source_ids":source_ids ,"source_mask": source_mask,"target_ids": target_ids ,"target_mask": target_mask}

    def delete_random_token(self,tokens_vec):
        idx = random.choice(np.where(tokens_vec!=1)[0])
        tokens_vec[idx]=1
        return tokens_vec

    def _load_entire_data(self):
        if self.split=="train":
            filename = self.args.train_filename
        elif self.split=="dev":
            filename = self.args.dev_filename
        elif  self.split=="test":
            filename = self.args.test_filename
        else:
            raise Exception("NEED TO SELECT DATA SET")
        # load data
        examples = read_examples(filename,debug_mode=self.debugBool)
        features = convert_examples_to_features(examples, self.tokenizer,self.args,stage=self.split)
        self.all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        self.all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        self.all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        self.all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)


if __name__ == "__main__":
    pass