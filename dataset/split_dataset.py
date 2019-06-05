from shutil import copyfile
from os import listdir
from os.path import join

SRC_DIR = 'images'
TRAIN_DIR = '10k_images'
TEST_DIR = '500_images'

NUM_TEST = 500
NUM_TRAIN = 10000

# curr_num_train = 0
# for filename in listdir(SRC_DIR):
#     if curr_num_train == NUM_TRAIN: break
#
#     source = join(SRC_DIR, filename)
#     destination = join(TRAIN_DIR, filename)
#     copyfile(source, destination)
#     curr_num_train += 1

train_filenames = set()
for train_filename in listdir(TRAIN_DIR):
    train_filenames.add(train_filename)

curr_num_test = 0
for filename in listdir(SRC_DIR):
    if curr_num_test == NUM_TEST: break
    if filename not in train_filenames:
        source = join(SRC_DIR, filename)
        destination = join(TEST_DIR, filename)
        copyfile(source, destination)
        curr_num_test += 1
