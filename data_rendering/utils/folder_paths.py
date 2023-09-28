import os.path as osp

TEXTURE_FOLDER = "/messytable-slow/mini-imagenet-tools/mini_imagenet/"
TEXTURE_LIST = "/messytable-slow/mini-imagenet-tools/mini_imagenet_list.txt"
OBJECT_DIR = "/rayc-fast/ICCV2021_Diagnosis/ocrtoc_materials/models/"
OBJECT_CSV_PATH = "/rayc-fast/ICCV2021_Diagnosis/ocrtoc_materials/objects.csv"
SCENE_DIR = "/rayc-fast/ICCV2021_Diagnosis/ocrtoc_materials/scenes"
TEXTURE_SQ_FOLDER = "/messytable-slow/mini-imagenet-tools/mini_imagenet_square/"
TEXTURE_SQ_LIST = "/messytable-slow/mini-imagenet-tools/mini_imagenet_square/list.txt"
ENV_MAP_FOLDER = "/messytable-slow/mini-imagenet-tools/rand_env/"
ENV_MAP_LIST = "/messytable-slow/mini-imagenet-tools/rand_env/list.txt"

if not osp.exists(TEXTURE_FOLDER):
    #please change to path of "TEXTURE_FOLDER" and "TEXTURE_LIST" in your machine to the path of "mini-imagenet" dataset(you need to download the dataset of mini-imagenet first by your self)
    TEXTURE_FOLDER = "/mnt/sda/activezero/mini-imagenet/"
    TEXTURE_LIST = "/mnt/sda/activezero/mini-imagenet/mini_imagenet_list.txt"

    OBJECT_DIR = "/home/rayu/Projects/ICCV2021_Diagnosis/ocrtoc_materials/models/"
    OBJECT_CSV_PATH = "/home/rayu/Projects/ICCV2021_Diagnosis/ocrtoc_materials/objects.csv"
    SCENE_DIR = "/home/rayu/Projects/ICCV2021_Diagnosis/ocrtoc_materials/scenes"
    # TEXTURE_SQ_FOLDER = "/media/DATA/LINUX_DATA/activezero2/datasets/mini_imagenet_square/"
    # TEXTURE_SQ_LIST = "/media/DATA/LINUX_DATA/activezero2/datasets/mini_imagenet_square/list.txt"
    ENV_MAP_FOLDER = "/media/DATA/LINUX_DATA/activezero2/datasets/rand_env/"
    ENV_MAP_LIST = "/media/DATA/LINUX_DATA/activezero2/datasets/rand_env/list.txt"