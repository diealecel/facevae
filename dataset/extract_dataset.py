from multiprocessing import Pool, RawValue, Lock    # For multiprocessing.
from skimage.io import imread, imsave               # For reading and writing images.
from argparse import ArgumentParser                 # For argument parsing.
from warnings import filterwarnings                 # For catching warnings.
from urllib.error import HTTPError                  # For catching 404 exceptions.
from os.path import join, isfile                    # For save_image() and check_existence().
from time import sleep                              # For making workers sleep after fails.
from sys import stdout                              # For print_progress().
import csv                                          # For reading data.

filterwarnings('error')

# Updates progress bar as images are downloaded. Thread-safe.
class Progress(object):
    def __init__(self, value = 0):
        self.val = RawValue('i', value)
        self.lock = Lock()

    def update(self):
        with self.lock:
            self.val.value += 1
            print_progress(self.val.value, DATA_SZ, PROGRESS_STR)

    def decrease_size(self):
        with self.lock:
            global DATA_SZ
            DATA_SZ -= 1
            print_progress(self.val.value, DATA_SZ, PROGRESS_STR)


# Holds the information needed to extract a datum from the dataset.
class Datum():
    def __init__(self, id, url, x1, y1, x2, y2):
        self.id = id
        self.url = url
        self.x1, self.y1 = int(x1), int(y1)
        self.x2, self.y2 = int(x2), int(y2)

DATA_SZ = 964873

DATA_FILE = 'data.csv'
SAVE_DIR = 'images'
EXT = '.jpg'

PROGRESS_STR = 'Downloading images...'

ID_IDX = 0
URL_IDX = 1
X1_IDX, Y1_IDX, X2_IDX, Y2_IDX = 4, 5, 6, 7

data_processed = Progress()
NUM_WORKERS = 32
CHUNK_SZ = 256
MAX_REQ_ATTEMPTS = 3
WAIT_INTERVAL = 10

# Prints a progress bar to inform user of work being done.
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 2, bar_length = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = u'█' * filled_length + '-' * (bar_length - filled_length)

    completed_progress_bar = '\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)
    stdout.write(completed_progress_bar)

    # Print new line when complete
    if iteration == total: stdout.write('\n')
    stdout.flush()


# Downloads and returns the image from |datum|.
def download_image(datum):
    image = None
    for try_num in range(1, MAX_REQ_ATTEMPTS + 1):
        try: image = imread(datum.url); break
        except HTTPError as error:
            if error.code == 404: break
            else: sleep(try_num * WAIT_INTERVAL)

    if image is None: data_processed.decrease_size()
    return image


# Crops |image| according to |datum| and returns the cropped image.
def crop_image(image, datum):
    return image[datum.y1:datum.y2, datum.x1:datum.x2]


# Saves |image| according to the index in |datum|.
def save_image(image, datum):
    save_loc = join(SAVE_DIR, datum.id + EXT)
    while True:
        try: imsave(save_loc, image); break
        except: pass


# Returns whether or not |datum| has already been processed.
def already_extracted(datum):
    predicted_loc = join(SAVE_DIR, datum.id + EXT)
    return isfile(predicted_loc)


# Downloads, processes, and saves the image from |datum|.
def extract_image(datum):
    if not already_extracted(datum):
        image = download_image(datum)
        if image is None: return
        cropped_image = crop_image(image, datum)
        save_image(cropped_image, datum)

    data_processed.update()


# Returns the data table from |DATA_FILE|.
def get_data():
    data_points = []
    with open(DATA_FILE, 'r') as data_csv:
        data = csv.reader(data_csv)
        print('Reading data...')
        for datum_info in data:
            datum = Datum(datum_info[ID_IDX], datum_info[URL_IDX],
                          datum_info[X1_IDX], datum_info[Y1_IDX],
                          datum_info[X2_IDX], datum_info[Y2_IDX])
            data_points.append(datum)

    return data_points


# Extracts the cropped images detailed in |DATA_FILE|. Multiprocessing can
# be enabled via |multiprocessing_on|.
def extract_data(multiprocessing_on):
    data = get_data()

    if multiprocessing_on:
        print('Multiprocessing enabled, running on', str(NUM_WORKERS), 'threads.')
        with Pool(NUM_WORKERS) as pool:
            pool.map(extract_image, data, CHUNK_SZ)
    else:
        for datum in data:
            extract_image(datum)


def main(args):
    extract_data(args.multiprocessing)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-mp', '--multiprocessing',
                        help = 'enable multiprocessing', action = 'store_true')
    args = parser.parse_args()

    main(args)
