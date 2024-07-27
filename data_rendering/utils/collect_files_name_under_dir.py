import os

def collect_files_name_under_dir(dir_path,save_path):
    """
    Collect all files name under the dir_path
    :param dir_path: the path of the directory
    :return: a list of file names
    """
    files = []
    save_path = os.path.join(save_path, 'files_name.txt')
    with open(save_path, 'w') as f:
        for file in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, file)):
                files.append(file)
                f.write(file + '\n')
    return files

if __name__ == '__main__':
    dir_path = '/Users/jaycinos/activezero2_official/data_rendering/mini_imagenet/IMagenet-master/tiny-imagenet-200/test/images'
    save_path = '/Users/jaycinos/activezero2_official/data_rendering/mini_imagenet'
    collect_files_name_under_dir(dir_path,save_path)