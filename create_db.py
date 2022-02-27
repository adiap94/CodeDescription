import os
import json
import time
import pandas as pd

def main(source_path,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    save_path = os.path.join(out_dir,"db.csv")

    files = [os.path.join(root, f) for root, dirs, files in os.walk(source_path) for
             f in files if os.path.splitext(os.path.join(root, f))[1].lower() == '.jsonl']
    df = pd.DataFrame()

    for file in files:
        with open(file, 'r') as f:
            sample_file = f.readlines()

        for f in sample_file:
            metadata = json.loads(f)
            d={
                'source_path':file,
                'set': metadata['partition'],
                'language': metadata['language'],
                'code':metadata['code'],
                'code_tokens':metadata['code_tokens'],
                'func_name':metadata['func_name'],
                'path':metadata['path'],
                'docstring':metadata['docstring'],
                'docstring_tokens':metadata['docstring_tokens'],
               }
            # df = df.append(d, ignore_index=True)
            # df = pd.Series(d).to_frame().T
            # df.to_csv(save_path, mode='a',index=False, header=not os.path.exists(save_path))

    # timestamp = time.strftime("%Y%m%d-%H%M%S")


    print("done create db ")

if __name__ == '__main__':
    source_path = '/home/student/project/resources/data'
    out_dir = '/home/student/project/db'
    main(source_path,out_dir)
    pass