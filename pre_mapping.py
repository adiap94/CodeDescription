import os
import re
import csv
import sys
import tqdm


def handle_replacement_tokens(line):
    new_line = line
    uniques = set()
    for match in re.compile('replaceme\d+').findall(line):
        uniques.add(match.strip())

    for match in uniques:
        replaced = match.replace("replaceme", "@R_") + '@'
        new_line = new_line.replace(match, replaced)
    return new_line


def main(data_path):
    csv.field_size_limit(sys.maxsize)

    #data_path = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220317-230340'


    TRANSFORMS = ['transforms.RenameParameters','transforms.RenameLocalVariables', 'transforms.RenameFields', 'transforms.AddDeadCode']

    splits = [ 'test','train', 'valid']
    #splits = ['test']
    for split in splits:
        print("Loading identity transform...")
        ID_MAP = {}
        with open(os.path.join(data_path,'transforms.Identity',"masked_token_"+split+".tsv"), 'r') as identity_tsv:
            reader = csv.reader(
                (x.replace('\0', '') for x in identity_tsv),
                delimiter='\t', quoting=csv.QUOTE_NONE
            )
            next(reader, None)
            for line in reader:
                ID_MAP[line[0]] = (line[1], line[2])
        print("  + Loaded {} samples".format(len(ID_MAP)))
        print("Loading transformed samples...")
        TRANSFORMED = {}
        for transform_name in TRANSFORMS:
            TRANSFORMED[transform_name] = {}
            with open(os.path.join(data_path , transform_name,"masked_token_"+split+".tsv"), 'r') as current_tsv:
                reader = csv.reader(
                    (x.replace('\0', '') for x in current_tsv),
                    delimiter='\t', quoting=csv.QUOTE_NONE
                )
                next(reader, None)
                for line in reader:
                    TRANSFORMED[transform_name][line[0]] = handle_replacement_tokens(line[1])
            print("  + Loaded {} samples from '{}'".format(
                len(TRANSFORMED[transform_name]), transform_name
            ))

        print("Writing adv. {}ing samples...".format(split))
        with open(os.path.join(data_path,"adv_"+split+".tsv"), "w") as out_f:
            out_f.write('index\tfilename\tsrc\ttgt\t{}\n'.format(
                '\t'.join([
                    '{}'.format(i) for i in TRANSFORMS
                ])
            ))

            index = 0
            for key in tqdm.tqdm(ID_MAP.keys(), desc="  + Progress"):
                row = [ID_MAP[key][0], ID_MAP[key][1]]
                for transform_name in TRANSFORMS:
                    if key in TRANSFORMED[transform_name]:
                        row.append(TRANSFORMED[transform_name][key])
                    else:
                        row.append(ID_MAP[key][0])
                out_f.write('{}\t{}\t{}\n'.format(index, key,'\t'.join(row)))
                index += 1
        print("  + Adversarial {}ing file generation complete!".format(split))


if __name__ == "__main__":
    data_path = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220317-230340/'
    main(data_path)