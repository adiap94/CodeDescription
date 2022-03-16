import re
import os
import ast
import sys
import json
import gzip
import resource
import hashlib
import multiprocessing


from tqdm import tqdm

from os import listdir
from os.path import isfile, join
from docopt import docopt
from tree_sitter import Language

from language_data import LANGUAGE_METADATA
from process import DataProcessor

from fissix import pygram, pytree
from fissix.pgen2 import driver, token


PY_REJECT_REGEX = re.compile('\) ->')
BANNED_PY_SHAS = [
    '6c4c00718d4ad438aeda74b1f11aa9b4a386abea598b54c813daad38b32432b5',
    '1cb0a93c92bef56e64b37da0979da59a35ab3ea3d4dd5fcc7327efae0c091122',
    '3aab95329c4ac0f2ed4ea30debc94eeb48367df83aa18edef896cf50744d4b73',
    '5b1f872804478e3a48ea364c5ea771bf1fef52dad6cb4b9d535e11aea9b587e4'
]


def subtokenize(identifier):
    RE_WORDS = re.compile(r'''
        # Find words in a string. Order matters!
        [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
        [A-Z]?[a-z]+ |  # Capitalized words / all lower case
        [A-Z]+ |  # All upper case
        \d+ | # Numbers
        .+
    ''', re.VERBOSE)

    return [subtok.strip().lower() for subtok in RE_WORDS.findall(identifier) if not subtok == '_']


def remove_func_name(name, tokens):
    index = 0
    while index < len(tokens) - 1:
        if tokens[index] == name and tokens[index + 1] == "(":
            return tokens[:index + 1], tokens[index + 1:]
        index += 1
    assert False, "Unable to remove function name"


def process(target):
    DataProcessor.PARSER.set_language(Language(os.getcwd() + '/build/py-tree-sitter-languages.so', 'python'))
    processor = DataProcessor(
        language='python',
        language_parser=LANGUAGE_METADATA['python']['language_parser']
    )

    results = []


    try:
        parser = driver.Driver(pygram.python_grammar, convert=pytree.convert)
        parser.parse_string(target['the_code'].strip() + '\n')
        ast.parse(target['the_code'])
    except Exception:
        print('Failed to validate: ' + target['from_file'])
        return False, []

    functions = processor.process_blob(target['the_code'])

    for function in functions:
        sha256 = hashlib.sha256(
            function["function"].strip().encode('utf-8')
        ).hexdigest()
        if PY_REJECT_REGEX.search(function["function"]):
            continue
        if sha256 in BANNED_PY_SHAS:
            # print("  - Skipped '{}'".format(sha256))
            continue  # Spoon transformer chokes on these, so exclude

        tokens_pre, tokens_post = ([], [])

        try:
            tokens_pre, tokens_post = remove_func_name(
                function["identifier"].split('.')[-1],
                function["function_tokens"]
            )
        except:
            continue

        results.append({
            "language": function["language"],
            "identifier": function["identifier"].split('.')[-1],
            "target_tokens": subtokenize(function["identifier"].split('.')[-1]),
            "source_tokens": tokens_post,
            "elided_tokens": tokens_pre,
            "source_code": function["function"] if function["language"] != "java" else (
                    'class WRAPPER {\n' + function["function"] + '\n}\n'
            ),
            "sha256_hash": sha256,
            "split": target['split'],
            "from_file": target['from_file']
        })

    return True, results


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
    sys.setrecursionlimit(10**6)

    pool = multiprocessing.Pool()
    targets = []
    outMap = {}
    Language.build_library('build/py-tree-sitter-languages.so', ['tree-sitter-python'])
    data_path = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220315-161632/'
    splits = ['test', 'train', 'valid']
    TRANSFORMS = ['transforms.Identity','transforms.RenameParameters','transforms.RenameLocalVariables', 'transforms.RenameFields', 'transforms.AddDeadCode']
    for t_name in TRANSFORMS:

        for split in splits:
            location = data_path + '{}/{}'.format(t_name, split)

            outMap[location] = gzip.open(
                data_path +t_name + '/masked_' + split + '.jsonl.gz',
                'wb'
            )

            onlyfiles = [f for f in listdir(location.strip()) if isfile(join(location.strip(), f))]
            for the_file in onlyfiles:
                with open(join(location.strip(), the_file), 'r') as fhandle:
                    targets.append({
                        'the_code': fhandle.read(),
                        'language': 'python',
                        'split': location,
                        'from_file': the_file
                    })

        results = pool.imap_unordered(process, targets, 2000)

        accepts = 0
        total = 0
        func_count = 0
        mismatches = 0
        for status, functions in tqdm(results, total=len(targets), desc="  + Normalizing"):
            total += 1
            if status:
                accepts += 1
            for result in functions:
                func_count += 1
                outMap[result['split']].write(
                    (json.dumps(result) + '\n').encode()
                )

        print("    - Parse success rate {:.2%}% ".format(float(accepts) / float(total)), file=sys.stderr)
        print("    - Rejected {} files for parse failure".format(total - accepts), file=sys.stderr)
        print("    - Rejected {} files for regex mismatch".format(mismatches), file=sys.stderr)
        print("    + Finished. {} functions extraced".format(func_count), file=sys.stderr)