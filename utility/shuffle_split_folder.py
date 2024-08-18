

import os
import glob
import random
import shutil



def shuffle_split_folder(
    target_folder,
    file_type_list=['JPG','jpg'],
    random_seed=5,
    split_ratio=0.1):

    folder_name = target_folder.split('/')[-1]
    folder_path_list = target_folder.split('/')[:-1]
    if target_folder.startswith('/'):
        folder_path_list.insert(0,'/')
    else:
        folder_path_list.insert(0,'.')
    save_folder_path = os.path.join(*folder_path_list)


    file_list = []
    for ty_ in file_type_list:
        temp_file_list = glob.glob(target_folder + f'/**/*.{ty_}', recursive=True)    
        file_list.extend(temp_file_list)

    random.seed(random_seed)
    file_list.sort()
    random.shuffle(file_list)


    length = len(file_list)
    
    cutting = int(length * split_ratio) + 1

    # front list and save
    front_list = file_list[:cutting]
    
    front_name = f'{folder_name}_{split_ratio}'
    front_path = os.path.join(save_folder_path, front_name)
    os.makedirs(front_path, exist_ok=True)

    for f_ in front_list:
        shutil.copy(f_, front_path)
    
    
    # tail list and save
    tail_list = file_list[cutting:]
    
    tail_name = f'{folder_name}_{1-split_ratio}'
    tail_path = os.path.join(save_folder_path, tail_name)
    os.makedirs(tail_path, exist_ok=True)

    for f_ in tail_list:
        shutil.copy(f_, tail_path)


if __name__ == '__main__':
    target_folder_path = '/home/ray/workspace/datasets/fish_disease_datasets/fish_disease_anomaly_dataset/rock_bream/Abnormal/back'
    target_folder_path = 'rock_bream/Abnormal/back_spike'
    target_folder_path = 'rock_bream/Abnormal/body_surface'
    target_folder_path = 'rock_bream/Abnormal/face'
    target_folder_path = 'rock_bream/Abnormal/mouth'
    target_folder_path = 'rock_bream/Abnormal/side_spots'
    target_folder_path = 'rock_bream/Abnormal/tail'
    target_folder_path = 'rock_bream/Normal'
    shuffle_split_folder(target_folder_path)
    print('end')