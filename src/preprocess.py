"""
Preprocess Cats vs Dogs dataset:
- Resize images to 224x224 RGB
- Split into train/val/test (80/10/10)
- Apply data augmentation to training set
- Save processed images to data/processed/{train,val,test}/cat|dog
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import random

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
IMG_SIZE = (224, 224)
SEED = 42
random.seed(SEED)

def make_dirs():
    for split in ['train', 'val', 'test']:
        for cls in ['cat', 'dog']:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def get_image_paths():
    cats = list((RAW_DIR / 'training_set' / 'cats').glob('*.jpg'))
    dogs = list((RAW_DIR / 'training_set' / 'dogs').glob('*.jpg'))
    test_cats = list((RAW_DIR / 'test_set' / 'cats').glob('*.jpg'))
    test_dogs = list((RAW_DIR / 'test_set' / 'dogs').glob('*.jpg'))
    return cats, dogs, test_cats, test_dogs

def split_data(cats, dogs):
    train_c, valtest_c = train_test_split(cats, test_size=0.2, random_state=SEED)
    val_c, test_c = train_test_split(valtest_c, test_size=0.5, random_state=SEED)
    train_d, valtest_d = train_test_split(dogs, test_size=0.2, random_state=SEED)
    val_d, test_d = train_test_split(valtest_d, test_size=0.5, random_state=SEED)
    return (train_c, val_c, test_c), (train_d, val_d, test_d)

def augment_image(img):
    # Simple augmentation: horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def process_and_save(img_path, out_dir, augment=False):
    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    if augment:
        img = augment_image(img)
    out_path = out_dir / img_path.name
    img.save(out_path)

def main():
    make_dirs()
    cats, dogs, test_cats, test_dogs = get_image_paths()
    train_c, val_c = train_test_split(cats, test_size=0.2, random_state=SEED)
    train_d, val_d = train_test_split(dogs, test_size=0.2, random_state=SEED)
    # Use provided test_set as test split
    for img_path in tqdm(train_c, desc='Processing train/cat'):
        process_and_save(img_path, PROCESSED_DIR / 'train' / 'cat', augment=True)
    for img_path in tqdm(train_d, desc='Processing train/dog'):
        process_and_save(img_path, PROCESSED_DIR / 'train' / 'dog', augment=True)
    for img_path in tqdm(val_c, desc='Processing val/cat'):
        process_and_save(img_path, PROCESSED_DIR / 'val' / 'cat')
    for img_path in tqdm(val_d, desc='Processing val/dog'):
        process_and_save(img_path, PROCESSED_DIR / 'val' / 'dog')
    for img_path in tqdm(test_cats, desc='Processing test/cat'):
        process_and_save(img_path, PROCESSED_DIR / 'test' / 'cat')
    for img_path in tqdm(test_dogs, desc='Processing test/dog'):
        process_and_save(img_path, PROCESSED_DIR / 'test' / 'dog')

if __name__ == '__main__':
    main()
