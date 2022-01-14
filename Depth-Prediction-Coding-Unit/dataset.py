import os
import sys
import re
import gzip
import numpy as np
import random
import tensorflow as tf

data_dir = './dataset_shuffled/'
INFO = './Info/'
IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_EXT_FEATURES = 1
NUM_LABEL_BYTES = 16
NUM_SAMPLE_LENGTH = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + 64 + (51 + 1) * NUM_LABEL_BYTES
DEFAULT_THR_LIST = [0.5, 1.5, 2.5]

PARTLY_TUNING_MODE = 0

TRAINSET_MAXSIZE = 10000000
VALIDSET_MAXSIZE = 10000000
TESTSET_MAXSIZE = 10000000

TRAINSET_READSIZE = 80000
VALIDSET_READSIZE = 60000
TESTSET_READSIZE = 60000

TRAIN_FILE_READER = []
VALID_FILE_READER = []
TEST_FILE_READER = []

TRAINSET = []
VALIDSET = []
TESTSET = []

# select training for which range of QP
MODEL_TYPE = 1

if MODEL_TYPE == 1:
    MODEL_NAME = 'qp22'
    SELECT_QP_LIST = [22]
    EVALUATE_QP_THR_LIST = [20, 25]
if MODEL_TYPE == 2:
    MODEL_NAME = 'qp27'
    SELECT_QP_LIST = [27]
    EVALUATE_QP_THR_LIST = [25, 30]
if MODEL_TYPE == 3:
    MODEL_NAME = 'qp32'
    SELECT_QP_LIST = [32]
    EVALUATE_QP_THR_LIST = [30, 35]
if MODEL_TYPE == 4:
    MODEL_NAME = 'qp37'
    SELECT_QP_LIST = [37]
    EVALUATE_QP_THR_LIST = [35, 40]

DATA_SWITCH = 0


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def read32(f):
    byte0 = ord(f.read(1)[0])
    byte1 = ord(f.read(1)[0])
    byte2 = ord(f.read(1)[0])
    byte3 = ord(f.read(1)[0])
    return byte0+(byte1 << 8)+(byte2 << 16)+(byte3 << 24)


class DataSet(tf.keras.utils.Sequence):
    def __init__(self, images, labels, qps, fake_data=False, dtype=tf.float32, shuffle=False, batch_size=50):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (
                images.shape, labels.shape))
            self._num_examples = images.shape[0]
            images = images.astype(np.float32)
            labels = labels.astype(np.int32)
            qps = qps.astype(np.float32)

        self._images = images
        self._labels = labels
        self._qps = qps
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.batch_size = batch_size
        self.indices = np.arange(self._num_examples)
        self.shuffle = shuffle
        self.shuffle_on_epoch_end()

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def qps(self):
        return self._qps

    def __len__(self):
        return self._num_examples // self.batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def shuffle_on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def correct_labels(self, labels):
        labels = tf.reshape(labels, [-1, 4, 4, 1])

        def aver_pool(x, k_width):
            x = tf.cast(x, tf.float32)
            return tf.nn.avg_pool(x, ksize=[1, k_width, k_width, 1], strides=[1, k_width, k_width, 1], padding='SAME')
        x16 = tf.nn.relu(labels-2)
        x32 = tf.nn.relu(aver_pool(labels, 2)-1)-tf.nn.relu(aver_pool(labels, 2)-2)
        x64 = tf.nn.relu(aver_pool(labels, 4)-0)-tf.nn.relu(aver_pool(labels, 4)-1)
        return [x64, x32, x16]

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return [self._images[indices], self._qps[indices]], self.correct_labels(self._labels[indices])


def get_data_set(file_reader, read_bytes, is_loop=True, dtype=np.uint8):

    data = file_reader.read_data(read_bytes, isloop=is_loop, dtype=dtype)
    data_bytes = len(data)
    assert data_bytes % NUM_SAMPLE_LENGTH == 0
    num_samples = int(data_bytes / NUM_SAMPLE_LENGTH)

    data = data.reshape(num_samples, NUM_SAMPLE_LENGTH)

    images = data[:, 0:4096].astype(np.float32)
    images = np.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    qps = np.random.choice(SELECT_QP_LIST, size=num_samples)
    qps = qps.reshape(num_samples, NUM_EXT_FEATURES)

    labels = np.zeros((num_samples, NUM_LABEL_BYTES))
    for i in range(num_samples):
        labels[i, :] = data[i, 4160+qps[i, 0]*NUM_LABEL_BYTES:4160+(qps[i, 0]+1)*NUM_LABEL_BYTES]

    return DataSet(images, labels, qps)


def get_train_valid_test_sets(DATA_SWITCH):
    global TRAINSET, VALIDSET, TESTSET

    if DATA_SWITCH == 0:  # full sets (the full program for generating these files will be open on GitHub soon)
        TRAINSET = 'AI_Train_2446725.dat_shuffled'
        VALIDSET = 'AI_Valid_143925.dat_shuffled'
        TESTSET = 'AI_Test_287850.dat_shuffled'
    elif DATA_SWITCH == 1:  # demo sets
        TRAINSET = 'AI_Train_5000.dat_shuffled'
        VALIDSET = 'AI_Valid_5000.dat_shuffled'
        TESTSET = 'AI_Test_5000.dat_shuffled'
    elif DATA_SWITCH == 2:  # choose other files if necessary
        pass


def get_file_size(file_path):
    try:
        size = os.path.getsize(file_path)
        return size
    except Exception as e:
        print(e)


class FileReader():
    def __init__(self) -> None:
        self._file_name = ''
        self._file_pointer = None
        self._current_bytes = 0
        self._max_bytes = 0

    def initialize(self, file_name, max_bytes):
        self._file_name = file_name
        print('Opening file: ', file_name)
        self._file_pointer = open(file_name, 'rb')
        self._current_bytes = 0

        file_bytes = get_file_size(file_name)
        if max_bytes < file_bytes:
            self._max_bytes = max_bytes
        else:
            self._max_bytes = file_bytes

    def get_file_name(self):
        return self._file_name

    def get_file_pointer(self):
        return self._file_pointer

    def get_max_bytes(self):
        return self._max_bytes

    def get_current_bytes(self):
        return self._current_bytes

    def read_data(self, read_bytes, isloop, dtype):
        if read_bytes > self._max_bytes:
            read_bytes = self._max_bytes
        if self._current_bytes + read_bytes <= self._max_bytes:
            buf = self._file_pointer.read(read_bytes)
            data = np.frombuffer(buf, dtype=dtype)
            self._current_bytes += read_bytes
        else:
            if isloop == False:
                self._file_pointer = open(self._file_name, 'rb')
                buf = self._file_pointer.read(read_bytes)
                data = np.frombuffer(buf, dtype=dtype)
                self._current_bytes = read_bytes
            else:
                buf = self._file_pointer.read(self._max_bytes-self._current_bytes)
                data1 = np.frombuffer(buf, dtype=dtype)
                self._file_pointer = open(self._file_name, 'rb')
                buf = self._file_pointer.read(read_bytes-(self._max_bytes-self._current_bytes))
                data2 = np.frombuffer(buf, dtype=dtype)
                data = np.concatenate((data1, data2))
                self._current_bytes = read_bytes-(self._max_bytes-self._current_bytes)
        return data


def read_data_sets(fake_data=False):
    global TRAIN_FILE_READER, VALID_FILE_READER, TEST_FILE_READER

    class DataSets(object):
        pass

    data_sets = DataSets()
    data_sets.train = []
    data_sets.validation = []
    data_sets.test = []
    data_sets.trainpart = []
    data_sets.testpart = []

    get_train_valid_test_sets(DATA_SWITCH)

    TRAIN_FILE_READER = FileReader()
    TRAIN_FILE_READER.initialize(os.path.join(data_dir, TRAINSET), TRAINSET_MAXSIZE * NUM_SAMPLE_LENGTH)

    VALID_FILE_READER = FileReader()
    VALID_FILE_READER.initialize(os.path.join(data_dir, VALIDSET), VALIDSET_MAXSIZE * NUM_SAMPLE_LENGTH)

    TEST_FILE_READER = FileReader()
    TEST_FILE_READER.initialize(os.path.join(data_dir, TESTSET), TESTSET_MAXSIZE * NUM_SAMPLE_LENGTH)

    change_train_data_set(data_sets)
    change_valid_data_set(data_sets)
    change_test_data_set(data_sets)

    return data_sets


def change_train_data_set(data_sets):
    global TRAIN_FILE_READER
    data_sets.train = get_data_set(TRAIN_FILE_READER, TRAINSET_READSIZE * NUM_SAMPLE_LENGTH)


def change_valid_data_set(data_sets):
    global VALID_FILE_READER
    data_sets.validation = get_data_set(VALID_FILE_READER, VALIDSET_READSIZE * NUM_SAMPLE_LENGTH)


def change_test_data_set(data_sets):
    global TEST_FILE_READER
    data_sets.test = get_data_set(TEST_FILE_READER, TESTSET_READSIZE * NUM_SAMPLE_LENGTH)


if __name__ == '__main__':
    obj = read_data_sets()
    train = obj.train
    print(train.__len__())
    
    print(train.__getitem__(0))
