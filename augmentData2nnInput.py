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

def alignAugmentData2Source(tsv_path,source_path,out_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, mode=0o777)

    # source
    print("about to load source json")
    df_source = read_source_jsonl(source_path)
    print("finish loading source json")

    # augmentation
    df = pd.read_csv(tsv_path, sep='\t')
    df["number"] = df["filename"].apply(lambda x: int(x.split(".")[0]))
    augmentation_type_list = df.columns.drop(["filename","src","tgt","number"])

    # init
    d_aug = {}
    for aug in augmentation_type_list:
        d_aug[aug] = []
    counter = 0
    # alignment
    for index,row in df.iterrows():
        print("| python_id_number is : " +  str(row.number))
        source_info = df_source.loc[row.number]
        if not row.tgt.split(" ")[0].lower() in source_info.func_name.lower():
            print("there is problem in index " + str(row.number))


        for aug_type in augmentation_type_list:
            d = {}
            code_str = "def "+ str(source_info.func_name) +" "+ str(row[aug_type])
            code_tokens=code_str.split(" ")

            d["func_name"] = source_info.func_name
            d["code"] = code_str
            d['code_tokens'] = code_tokens
            d["docstring"] = source_info.docstring
            d["docstring_tokens"] = source_info.docstring_tokens

            # save to jsonl
            d_aug[aug_type].append(d)

    for key, value in d_aug.items():
        out_file = os.path.join(out_dir,"test_"+key+".jsonl")
        # with open(out_file, 'w') as outfile:
        #     json.dump(value, outfile)
        with open(out_file, 'w') as outfile:
            for entry in value:
                json.dump(entry, outfile)
                outfile.write('\n')




if __name__ == "__main__":
    file_path ="/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220317-230340/final_test.tsv"
    source_path = "/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/test.jsonl"
    out_dir = "/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220317-230340/data_jsonl/test"
    alignAugmentData2Source(file_path,source_path,out_dir)
    pass