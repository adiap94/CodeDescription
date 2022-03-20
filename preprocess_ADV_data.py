import os
import sys
import transform_code
import transformed_to_json
import json_to_tsv
import pre_mapping
import create_replace_mapping
import replace_tokens
import augmentData2nnInput

def main(path_to_data,path_to_model):
    splits = ["test","train","valid"]
    # path_to_data = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/'
    #Preproccesing and check it can add transformation to the code string to code
    print("Running transform_code script "  )
    dir_with_codes_transforms = transform_code.main(path_to_data)
    print("Finished transform_code script run ")
    #dir_with_codes_transforms = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/adv/adv_20220318-215953/'
    #Save the transformes codes to jason file
    print("Running transformed_to_json script ")
    transformed_to_json.main(dir_with_codes_transforms)
    print("Finished transformed_to_json script run")
    #Creating TSV table with all each transformation with masks
    print("Running json_to_tsv script ")
    json_to_tsv.main(dir_with_codes_transforms)
    print("Finished json_to_tsv script run")
    #Creating Premaping
    print("Running pre_mapping script ")
    dict_of_Adv_tsvpath = pre_mapping.main(dir_with_codes_transforms)
    print('saved the following files in pre mapping:')
    print(dict_of_Adv_tsvpath)
    print("Finished pre_mapping script run ")
    # path_to_model = "/tcmldrive/project/results/python/20220314-163735/checkpoint-best-bleu/pytorch_model.bin"
    mapping_path = {}
    for split in splits:
        print ("Running pre_mapping script for " + split + " dataset")
        mapping_path[split] = os.path.join(dir_with_codes_transforms, split + "_mapping")
        print('create mapping in ' + mapping_path[split])
        sys.argv = ['create_replace_mapping.py', '--data_path', dict_of_Adv_tsvpath[split],
                    '--save_path', mapping_path[split], '--batch_size', '30', '--num_replacements', '1']
        mapping_path[split] = create_replace_mapping.main()
        print('finished creating mapping in ' + mapping_path[split])
        print("Running replace_tokens script for " + split + " dataset")
        sys.argv = ['replace_tokens.py', '--source_data_path', dict_of_Adv_tsvpath[split],
                    '--dest_data_path', os.path.join(dir_with_codes_transforms, 'final_' + split + '.tsv'), '--mapping_json', mapping_path[split]]
        replace_tokens.main()
        print("Finished replace_tokens script run for " + split + " dataset")

        tsv_file = os.path.join(dir_with_codes_transforms, 'final_' + split + '.tsv')
        clean_data_sorce = os.path.join(path_to_data,split + ".jsonl")
        output_dir_final_json = os.path.join(dir_with_codes_transforms,"data_jsonl",split + "_new")
        print ("Running augmentData2nnInput script for " + split + " dataset")
        augmentData2nnInput.alignAugmentData2Source(tsv_file,clean_data_sorce,output_dir_final_json)
        print("Finished augmentData2nnInput script run for " + split + " dataset")





if __name__ == "__main__":
    path_to_data = '/tcmldrive/project/resources/data_codesearch/CodeSearchNet/python/'
    path_to_model = "/tcmldrive/project/results/python/20220314-163735/checkpoint-best-bleu/pytorch_model.bin"
    main(path_to_data, path_to_model)