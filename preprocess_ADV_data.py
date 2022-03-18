import os
import io
import sys
import ast
import json
import gzip
import copy
import tqdm
import astor
import random
import itertools
import multiprocessing
import re
import resource
import hashlib
import time
from tqdm import tqdm
from yapf.yapflib.yapf_api import FormatCode
from os import listdir
from os.path import isfile, join
import transform_code
import transformed_to_json
import json_to_tsv
import pre_mapping
import create_replace_mapping
import replace_tokens

if __name__ == "__main__":
    splits = ["test","train","valid"]
    path_to_data = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/'
    #Preproccesing and check it can add transformation to the code string to code
    #dir_with_codes_transforms = transform_code.main(path_to_data)
    dir_with_codes_transforms = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220318-215953/'
    #Save the transformes codes to jason file
    #transformed_to_json.main(dir_with_codes_transforms)
    #Creating TSV table with all each transformation with masks
    #json_to_tsv.main(dir_with_codes_transforms)
    #Creating Premaping
    dict_of_Adv_tsvpath = pre_mapping.main(dir_with_codes_transforms)
    print('saved the following files in pre mapping:')
    print(dict_of_Adv_tsvpath)
    path_to_model = "/tcmldrive/project/results/python/20220314-163735/checkpoint-best-bleu/pytorch_model.bin"
    mapping_path = {}
    for split in splits:

        mapping_path[split] = os.path.join(dir_with_codes_transforms, split + "_mapping")
        print('create mapping in ' + mapping_path[split])
        sys.argv = ['create_replace_mapping.py', '--data_path', dict_of_Adv_tsvpath[split],
                    '--save_path', mapping_path[split], '--batch_size', '30', '--num_replacements', '1']
        mapping_path[split] = create_replace_mapping.main()
        print('finished creating mapping in ' + mapping_path[split])
        sys.argv = ['replace_tokens.py', '--source_data_path', dict_of_Adv_tsvpath[split],
                    '--dest_data_path', os.path.join(dir_with_codes_transforms, 'final_' + split + '.tsv'), '--mapping_json', mapping_path[split]]
        replace_tokens.main()






