import re
import pandas as pd
import json

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
        df = df.append(d, ignore_index=True)
    return df

def alignAugmentData2Source(tsv_path,source_path,out_dir=None):
    # source
    df_source = read_source_jsonl(source_path)

    # augmentation
    df = pd.read_csv(tsv_path, sep='\t')
    augmentation_type_list = df.columns.drop(["index","src","tgt"])

    # alignment
    for index,row in df.iterrows():
        tft = row.tgt
        source_info = df_source[df_source["func_name"]==tft]
        for aug_type in augmentation_type_list:
            d = {}
            code_str = row[aug_type]
            # TODO - THERE IS STILL PROBLEM WITH [ OR ] IN REGEX
            code_tokens = re.findall(r"[\w']+|[.,!?;[{}()/]", code_str)

            d["func_name"] = source_info["func_name"]
            d["code"] = code_str
            d['code_tokens'] = code_tokens
            d["docstring"] = source_info["docstring"]
            d["docstring_tokens"] = source_info["docstring_tokens"]

            # save to jsonl
            #TODO

    pass


if __name__ == "__main__":
    file_path ="/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220317-143412/adv_test.tsv"
    source_path = "/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/train.jsonl"
    alignAugmentData2Source(file_path)
    pass