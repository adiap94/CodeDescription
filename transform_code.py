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


def t_unroll_whiles(the_ast, uid=1):
    if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
        return False, the_ast

    class UnrollWhiles(ast.NodeTransformer):
        def __init__(self, selection):
            self.selection = selection
            self.count = 0
            self.done = False
            super().__init__()

        def visit_While(self, node):
            if self.done:
                return node
            if self.count != self.selection:
                self.count += 1
                return node

            self.done = True
            return ast.While(
                test=node.test,
                body=node.body + [node, ast.Break()],
                orelse=[]
            )

    changed = False
    count = 0

    for node in ast.walk(the_ast):
        if isinstance(node, ast.While):
            changed = True
            count += 1

    if count == 0:
        return False, the_ast

    return changed, UnrollWhiles(random.randint(0, count - 1)).visit(the_ast)


def t_wrap_try_catch(the_ast, uid=1):
    if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
        return False, the_ast

    temp = ast.Try(
        body=the_ast.body[0].body,
        handlers=[ast.ExceptHandler(
            type=ast.Name(id='Exception', ctx=ast.Load()),
            name='REPLACME' + str(uid),
            body=[ast.Raise()]
        )],
        orelse=[],
        finalbody=[]
    )

    the_ast.body[0].body = [temp]

    return True, the_ast


def t_add_dead_code(the_ast, uid=1):
    if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
        return False, the_ast

    if bool(random.getrandbits(1)):
        the_ast.body[0].body.insert(
            0,
            ast.If(
                test=ast.Name(id="False", ctx=ast.Load()),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id="REPLACME" + str(uid), ctx=ast.Store())],
                        value=ast.Num(n=1)
                    )
                ],
                orelse=[]
            )
        )
    else:
        the_ast.body[0].body.append(
            ast.If(
                test=ast.Name(id="False", ctx=ast.Load()),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id="REPLACME" + str(uid), ctx=ast.Store())],
                        value=ast.Num(n=1)
                    )
                ],
                orelse=[]
            )
        )

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
    transforms.append(('transforms.RenameLocalVariables',  t_rename_local_variables))
    transforms.append(('transforms.RenameFields', t_rename_fields))
    transforms.append(('transforms.AddDeadCode', t_add_dead_code))

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


def remove_comment(code_string):
    code_string_original = code_string
    try:
        while '"""' in code_string:
            idx_start = code_string.index('"""')
            #last_index_doen_line = code_string[:idx_start].rindex('\n')
            idx_end = code_string[idx_start + 1:].index('"""')
            idx_end = idx_end + 2
            # if code_string[idx_start + 1 + idx_end + 1] == '\n':
            #     idx_end +=1
            # code_string = code_string[:last_index_doen_line+1] + code_string[idx_start + 1 + idx_end + 2:]
            code_string = code_string[:idx_start] + code_string[idx_start + 1 + idx_end + 1:]
    except:
        return code_string_original

    code_string_original = code_string
    try:
        while ' #' in code_string or '\n#'  in code_string:
            idx_start = code_string.index('#')
            idx_end = code_string[idx_start + 1:].index('\n')
            #last_index_doen_line = code_string[:idx_start].rindex('\n')
            #code_string = code_string[:last_index_doen_line+1] + code_string[idx_start + 1 + idx_end + 1 + 1:]
            code_string = code_string[:idx_start] + code_string[idx_start + 1 + idx_end + 1 + 1:]
        code_string = code_string+"\n\n"
        return code_string
    except:
        return code_string_original


if __name__ == "__main__":
    time_str = time.strftime("%Y%m%d-%H%M%S")
    print("Starting transform:")

    data_path = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/'

    tasks = []
    pool = multiprocessing.Pool(1)

    print("  + Loading tasks...")
    splits =['test', 'train', 'valid']
    for split in splits:
        for idx, line in enumerate(open(data_path + '{}.jsonl'.format(split))):
            # line = line.strip()
            as_json = json.loads(line)
            code = as_json['code']
            code = remove_comment(code_string=code)
            tasks.append((split, as_json['sha'], code))

    results = pool.imap_unordered(process, tasks, 4000)
    print("    + Loaded {} transform tasks".format(len(tasks)))


    print("  + Transforming in parallel...")
    names_covered = []
    for idx, single_result in enumerate(
            tqdm(results, desc="    + Progress", total=len(tasks))):
        for changed, split, t_name, the_hash, code in single_result:
            if not changed:
                continue
            out_dir_path = os.path.join(data_path, "adv", 'adv_' + time_str, t_name, split)
            if (t_name + split) not in names_covered:
                names_covered.append(t_name + split)
                os.makedirs(out_dir_path, exist_ok=True)
                os.chmod(out_dir_path, mode=0o777)

            file_path = os.path.join(out_dir_path, str(idx) + ".py")
            with open(file_path, 'w') as fout:
                fout.write('{}\n'.format(code))

    print("  + Transforms complete!")
