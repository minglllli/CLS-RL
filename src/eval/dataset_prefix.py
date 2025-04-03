# change to your owm dataset path
def dataset_prefix(dataset):
    prefix = ''
    if dataset == 'Caltech101':
        prefix = 'Caltech101'
        path_prefix = prefix+'Dataset/caltech-101/'
    elif dataset == 'Imagenet':
        prefix = 'Imagenet'
        path_prefix = prefix+'Dataset/imagenet/val/'
    elif dataset == 'DescribableTextures':
        prefix = dataset
        path_prefix = prefix+'Dataset/dtd/images/'
    elif dataset == 'EuroSAT':
        prefix = dataset
        path_prefix = prefix+'Dataset/2750/'
    elif dataset == 'Food101':
        prefix = dataset
        path_prefix = prefix+'Dataset/food-101/images/'
    elif dataset == 'FGVCAircraft':
        prefix = dataset
        path_prefix = prefix+'Dataset/fgvc-aircraft-2013b/data/'
    elif dataset == 'OxfordFlowers':
        prefix = dataset
        path_prefix = prefix+'Dataset/jpg/'
    elif dataset == 'OxfordPets':
        prefix = dataset
        path_prefix = prefix+'Dataset/oxfordpets/images/'
    elif dataset == 'StanfordCars':
        prefix = dataset
        path_prefix = prefix+'Dataset/cars_test/'
    elif dataset == 'SUN397':
        prefix = dataset
        path_prefix = prefix+'Dataset/SUN397/'
    elif dataset == 'UCF101':
        prefix = dataset
        path_prefix = prefix+'Dataset/UCF-101-midframes/'
    return prefix, path_prefix