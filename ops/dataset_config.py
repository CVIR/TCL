import os

import yaml

def get_rootdataset(dataset):
    with open('root_dataset.yaml') as file:
        dataset_path = yaml.load(file, Loader=yaml.FullLoader)
        return dataset_path[dataset]


def return_ucf101(modality):
    filename_categories = 'data/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Frames/'
        filename_imglist_train = 'data/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'data/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    filename_categories = 'data/category_mini.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Frames'
        filename_imglist_train = 'data/train_videofolder_mini.txt'
        filename_imglist_val = 'data/val_videofolder_mini.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_mini_moments(modality):
    filename_categories = 'data/categories.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'Frames'
        filename_imglist_train = 'data/train_videofolder.txt'
        filename_imglist_val = 'data/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    filename_categories = 'data/classInd.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'Frames'
        filename_imglist_train = 'data/train_videofolder.txt'
        filename_imglist_val = 'data/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_charades_ego(modality):
    filename_categories = 157#'Charades_v1_classes.txt'
    if modality == 'RGB':
        prefix = '{:06d}.jpg'
        root_data = ROOT_DATASET + 'Frames'
        filename_imglist_train_1p = 'data/train_only1st_segments.txt'
        filename_imglist_train_3p = 'data/train_only3rd_segments.txt'
        filename_imglist_val_1p = 'data/test_only1st_segments.txt' 
        filename_imglist_val_3p = 'data/test_only3rd_segments.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train_1p, filename_imglist_train_3p, filename_imglist_val_1p, filename_imglist_val_3p, root_data, prefix


def return_charades_full(modality):
    filename_categories = 157
    if modality == 'RGB':
       prefix = '{:06d}.jpg'
       root_data = ROOT_DATASET + 'Frames'
       filename_imglist_train = 'data/train_segments.txt'#'data/train_videofolder.txt'
       filename_imglist_val = 'data/test_segments.txt' #'data/val_videofolder.txt'
    else:
       raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    global ROOT_DATASET
    ROOT_DATASET = get_rootdataset(dataset)
    dict_single = {'jester': return_jester, 'something': return_something, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51, 'mini-moments': return_mini_moments,
                   'kinetics': return_kinetics, 'charades_full':return_charades_full }
    dict_charades = {'charades_ego': return_charades_ego}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    elif dataset in dict_charades:
        file_categories, filename_imglist_train_1p, filename_imglist_train_3p, filename_imglist_val_1p, filename_imglist_val_3p, root_data, prefix = dict_charades[dataset](modality)
        file_imglist_train_1p = os.path.join(ROOT_DATASET, filename_imglist_train_1p)
        file_imglist_train_3p = os.path.join(ROOT_DATASET, filename_imglist_train_3p)
        file_imglist_val_1p = os.path.join(ROOT_DATASET, filename_imglist_val_1p)
        file_imglist_val_3p = os.path.join(ROOT_DATASET, filename_imglist_val_3p)
        return file_categories, file_imglist_train_1p, file_imglist_train_3p, file_imglist_val_1p, file_imglist_val_3p, root_data, prefix
    else:
        raise ValueError('Unknown dataset '+dataset)
      
    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
