import re
import os
import gzip
import json
import tqdm
import os.path
import multiprocessing


def camel_case_split(identifier):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
        identifier
    )
    return [m.group(0) for m in matches]


def subtokens(in_list):
    good_list = []
    for tok in in_list:
        for subtok in tok.replace('_', ' ').split(' '):
            if subtok.strip() != '':
                good_list.extend(camel_case_split(subtok))

    return good_list


def clean_name(in_list):
    return subtokens(in_list)


def normalize_subtoken(subtoken):
    normalized = re.sub(
        r'[^\x00-\x7f]', r'',  # Get rid of non-ascii
        re.sub(
            r'["\',`]', r'',  # Get rid of quotes and comma
            re.sub(
                r'\s+', r'',  # Get rid of spaces
                subtoken.lower()
                    .replace('\\\n', '')
                    .replace('\\\t', '')
                    .replace('\\\r', '')
            )
        )
    )

    return normalized.strip()


def process(item):
    src = list(filter(None, [
        normalize_subtoken(subtok) for subtok in subtokens(item[2])
    ]))
    tgt = list(filter(None, [
        normalize_subtoken(subtok) for subtok in clean_name(item[3])
    ]))

    return (
        len(src) > 0 and len(tgt) > 0,
        item[0],
        item[1],
        ' '.join(src),
        ' '.join(tgt)
    )


def main(data_path):
    print("Loading inputs...")

    has_baselines = False

    TRANSFORMS = ['transforms.Identity', 'transforms.RenameParameters','transforms.RenameLocalVariables', 'transforms.RenameFields', 'transforms.AddDeadCode']
    pool = multiprocessing.Pool()
    for t_name in TRANSFORMS:
        loc = os.path.join(data_path, t_name)
        tasks = []

        for split in ["test","train","valid"]:
            if not os.path.isfile(os.path.join(loc + '/masked_'+split+'.jsonl.gz')):
                continue
            if split == 'baseline':
                has_baselines = True
            for line in gzip.open(os.path.join(loc + '/masked_'+split+'.jsonl.gz')):
                as_json = json.loads(line)
                from_file = as_json['from_file'] if 'from_file' in as_json else '{}.java'.format(as_json['sha256_hash'])
                tasks.append((split, from_file, as_json['source_tokens'], as_json['target_tokens']))


        print("  + Inputs loaded")

        out_map = {
            'train': open(os.path.join(loc , 'masked_token_train.tsv'), 'w'),
            'valid': open(os.path.join(loc , 'masked_token_valid.tsv'), 'w'),
            'test': open(os.path.join(loc, 'masked_token_test.tsv'), 'w'),
        }
        # out_map = {
        #     'test': open(os.path.join(loc , 'masked_token_test.tsv'), 'w'),
        # }


        if has_baselines:
            print("  + Has baselines file")
            out_map['baseline'] = open(os.path.join(loc + 'baseline.tsv'))
            out_map['baseline'].write('from_file\tsrc\ttgt\n')

        print("  + Output files opened")

        out_map['test'].write('from_file\tsrc\ttgt\n')
        out_map['train'].write('from_file\tsrc\ttgt\n')
        out_map['valid'].write('from_file\tsrc\ttgt\n')

        print("  - Processing in parallel...")
        iterator = tqdm.tqdm(
            pool.imap_unordered(process, tasks, 3000),
            desc="    - Tokenizing",
            total=len(tasks)
        )
        for good, split, from_file, src, tgt in iterator:
            if not good:  # Don't let length == 0 stuff slip through
                continue
            out_map[split].write(
                '{}\t{}\t{}\n'.format(from_file, src, tgt)
            )
        print("    + Tokenizing complete")
        print("  + Done extracting tokens")

if __name__ == "__main__":
    data_path = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220317-230340/'
    main(data_path)