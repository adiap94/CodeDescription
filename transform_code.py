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

from tqdm import tqdm

from os import listdir
from os.path import isfile, join

PY_2_BUILTINS = [
    'bytearray',
    'IndexError',
    'all',
    'help',
    'vars',
    'SyntaxError',
    'unicode',
    'UnicodeDecodeError',
    'memoryview',
    'isinstance',
    'copyright',
    'NameError',
    'BytesWarning',
    'dict',
    'input',
    'oct',
    'bin',
    'SystemExit',
    'StandardError',
    'format',
    'repr',
    'sorted',
    'False',
    'RuntimeWarning',
    'list',
    'iter',
    'reload',
    'Warning',
    '__package__',
    'round',
    'dir',
    'cmp',
    'set',
    'bytes',
    'reduce',
    'intern',
    'issubclass',
    'Ellipsis',
    'EOFError',
    'locals',
    'BufferError',
    'slice',
    'FloatingPointError',
    'sum',
    'getattr',
    'abs',
    'exit',
    'print',
    'True',
    'FutureWarning',
    'ImportWarning',
    'None',
    'hash',
    'ReferenceError',
    'len',
    'credits',
    'frozenset',
    '__name__',
    'ord',
    'super',
    '_',
    'TypeError',
    'license',
    'KeyboardInterrupt',
    'UserWarning',
    'filter',
    'range',
    'staticmethod',
    'SystemError',
    'BaseException',
    'pow',
    'RuntimeError',
    'float',
    'MemoryError',
    'StopIteration',
    'globals',
    'divmod',
    'enumerate',
    'apply',
    'LookupError',
    'open',
    'quit',
    'basestring',
    'UnicodeError',
    'zip',
    'hex',
    'long',
    'next',
    'ImportError',
    'chr',
    'xrange',
    'type',
    '__doc__',
    'Exception',
    'tuple',
    'UnicodeTranslateError',
    'reversed',
    'UnicodeEncodeError',
    'IOError',
    'hasattr',
    'delattr',
    'setattr',
    'raw_input',
    'SyntaxWarning',
    'compile',
    'ArithmeticError',
    'str',
    'property',
    'GeneratorExit',
    'int',
    '__import__',
    'KeyError',
    'coerce',
    'PendingDeprecationWarning',
    'file',
    'EnvironmentError',
    'unichr',
    'id',
    'OSError',
    'DeprecationWarning',
    'min',
    'UnicodeWarning',
    'execfile',
    'any',
    'complex',
    'bool',
    'ValueError',
    'NotImplemented',
    'map',
    'buffer',
    'max',
    'object',
    'TabError',
    'callable',
    'ZeroDivisionError',
    'eval',
    '__debug__',
    'IndentationError',
    'AssertionError',
    'classmethod',
    'UnboundLocalError',
    'NotImplementedError',
    'AttributeError',
    'OverflowError'
]


def t_rename_fields(the_ast, uid=1):
    changed = False

    # Going to need parent info
    for node in ast.walk(the_ast):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    candidates = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and node.id == 'self':
            if isinstance(node.parent, ast.Attribute):
                if isinstance(node.parent.parent, ast.Call) and node.parent.parent.func == node.parent:
                    continue
                if node.parent.attr not in [c.attr for c in candidates]:
                    candidates.append(node.parent)

    if len(candidates) == 0:
        return False, the_ast

    selected = random.choice(candidates)

    to_rename = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and node.id == 'self':
            if isinstance(node.parent, ast.Attribute) and node.parent.attr == selected.attr:
                if isinstance(node.parent.parent, ast.Call) and node.parent.parent.func == node.parent:
                    continue
                to_rename.append(node.parent)

    for node in to_rename:
        changed = True
        node.attr = 'REPLACEME' + str(uid)

    return changed, the_ast


def t_rename_parameters(the_ast, uid=1):
    changed = False
    candidates = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.arg):
            if node.arg != 'self' and node.arg not in [c.arg for c in candidates]:
                # print(node.arg, node.lineno)
                candidates.append(node)

    if len(candidates) == 0:
        return False, the_ast

    selected = random.choice(candidates)
    parameter_defs = {}
    parameter_defs[selected.arg] = selected

    to_rename = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and node.id in parameter_defs:
            to_rename.append(node)
        elif isinstance(node, ast.arg) and node.arg in parameter_defs:
            to_rename.append(node)

    for node in to_rename:
        changed = True
        if hasattr(node, 'arg'):
            node.arg = 'REPLACEME' + str(uid)
        else:
            node.id = 'REPLACEME' + str(uid)

    return changed, the_ast


def t_rename_local_variables(the_ast, uid=1):
    changed = False
    candidates = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id not in [c.id for c in candidates]:
                # print(node.id, node.lineno)
                candidates.append(node)

    if len(candidates) == 0:
        return False, the_ast

    selected = random.choice(candidates)
    local_var_defs = {}
    local_var_defs[selected.id] = selected

    to_rename = []
    for node in ast.walk(the_ast):
        if isinstance(node, ast.Name) and node.id in local_var_defs:
            to_rename.append(node)

    for node in to_rename:
        changed = True
        node.id = 'REPLACEME' + str(uid)

    return changed, the_ast


class t_seq(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, the_ast):
        did_change = False
        cur_ast = the_ast
        for i, t in enumerate(self.transforms):
            changed, cur_ast = t(cur_ast, i + 1)
            if changed:
                did_change = True
        return did_change, cur_ast


def t_identity(the_ast):
    return True, the_ast


def process(item):
    (split, the_hash, og_code) = item

    transforms = [
        (
            'transforms.RenameParameters',
            t_rename_parameters
        ),
        (
          'transforms.Identity',
          t_identity
        )
    ]

    results = []
    for t_name, t_func in transforms:
        try:
            changed, result = t_func(
                ast.parse(og_code)
            )
            results.append((changed, split, t_name, the_hash, astor.to_source(result)))
        except Exception as ex:
            import traceback
            traceback.print_exc()
            results.append((False, split, t_name, the_hash, og_code))
    return results


if __name__ == "__main__":
    print("Starting transform:")
    pool = multiprocessing.Pool(1)
    data_path = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/'

    tasks = []

    print("  + Loading tasks...")
    splits = ['test', 'train','valid']
    for split in splits:
        for line in open(data_path + '{}.jsonl'.format(split)):
            as_json = json.loads(line)
            the_code = as_json['code']
            tasks.append((split, as_json['sha'], the_code))

    print("    + Loaded {} transform tasks".format(len(tasks)))
    results = pool.imap_unordered(process, tasks, 3000)

    print("  + Transforming in parallel...")
    names_covered = []
    for changed, split, t_name, the_hash, code in itertools.chain.from_iterable(tqdm(results, desc="    + Progress", total=len(tasks))):
        if not changed:
            continue

        if (t_name + split) not in names_covered:
            names_covered.append(t_name + split)
            os.makedirs(data_path + 'adv/{}/{}'.format(t_name, split), exist_ok=True)

        with open(data_path + 'adv/{}/{}/'.format(t_name, split) + '{}.py'.format(the_hash), 'w') as fout:
            fout.write('{}\n'.format(code))

    print("  + Transforms complete!")
