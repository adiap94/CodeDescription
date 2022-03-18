import re
import os
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def read_source_jsonl(source_path):
    df = pd.DataFrame()
    with open(source_path, 'r') as f:
        sample_file = f.readlines()

    for f in sample_file:
        d={}
        js = json.loads(f)
        d["func_name"] = js["func_name"]
        d["code"] = ' '.join(js['code_tokens'])
        d["docstring"] = js["docstring"]
        d["docstring_tokens"] = js["docstring_tokens"]
        # df = pd.concat([df,d], ignore_index=True)
        df = df.append(d, ignore_index=True)
    return df

def alignAugmentData2Source(tsv_path,source_path,Identity_dir,out_dir=None):
    # source
    df_source = read_source_jsonl(source_path)

    # augmentation
    df = pd.read_csv(tsv_path, sep='\t')
    augmentation_type_list = df.columns.drop(["filename","src","tgt"])

    # init
    d_aug = {}
    for aug in augmentation_type_list:
        d_aug[aug] = []

    # alignment
    for index,row in df.iterrows():

        with open(os.path.join(Identity_dir, row.filename), 'r') as f:
            lines = f.readlines()
            first_line = lines[0].split("(")[0]
        source_info = df_source[df_source.code.str.contains(first_line, regex=False)]
        if len(source_info)== 0:
            continue
        if len(source_info)> 1:
            print("need to debug")
        source_info = source_info.reset_index().iloc[0]

        for aug_type in augmentation_type_list:
            d = {}
            code_str = first_line +" "+ row[aug_type]
            code_tokens=code_str.split(" ")

            d["func_name"] = source_info["func_name"]
            d["code"] = code_str
            d['code_tokens'] = code_tokens
            d["docstring"] = source_info["docstring"]
            d["docstring_tokens"] = source_info["docstring_tokens"]

            # save to jsonl
            d_aug[aug].append(d)

    for key, value in d_aug.items():
        out_file = os.path.join("out_dir",key+".jsonl")
        with open(out_file, 'w') as outfile:
            json.dump(value, outfile)


    pass


if __name__ == "__main__":
    file_path ="/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220317-230340/final_test.tsv"
    source_path = "/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/test.jsonl"
    dir ="/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220317-230340/transforms.Identity/test/"
    out_file = "/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/tmp.jsonl"
    alignAugmentData2Source(file_path,source_path)
    pass