import glob
import shutil
import os

def only_python_files(
    target_folder='.',
    file_extends=['py'],
    save_folder='python_files',
    ):

    file_list = []
    for ext_ in file_extends:
        files = glob.glob(target_folder + f'/**/*.{ext_}', recursive=True)        
        file_list.extend(files)

    print(f'Found {len(file_list)} files.')
    
    for f_ in file_list:
        orginal_folder = f_.split('/')[:-1]
        save_folder_path = os.path.join(save_folder, *orginal_folder)
        os.makedirs(save_folder_path, exist_ok=True)
        shutil.copy(f_, save_folder_path)  # dst can be folder


if __name__ == '__main__':
    target_folder='.'
    file_extends=['py']
    save_folder='python_files'
    only_python_files(
        target_folder=target_folder,
        file_extends=file_extends,
        save_folder=save_folder,
    )
    print('end')
