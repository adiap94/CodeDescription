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

if __name__ == "__main__":
    path_to_data = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/'
    #Preproccesing and check it can add transformation to the code string to code
    dir_with_codes_transforms = transform_code.main(path_to_data)
    #Save the transformes codes to jason file
    transformed_to_json.main(dir_with_codes_transforms)
    #Creating TSV table with all each transformation with masks
    json_to_tsv.main(dir_with_codes_transforms)
    #Creating Premaping
    pre_mapping.main(dir_with_codes_transforms)
