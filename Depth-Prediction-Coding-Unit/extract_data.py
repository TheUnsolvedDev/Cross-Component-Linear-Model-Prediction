import os
import glob
import random
import numpy as np

YUV_PATH = './YUV_All/'
INFO_PATH = './Info/'
YUV_NAME_LIST_FULL = [
    'IntraTrain_768x512',
    'IntraTrain_1536x1024',
    'IntraTrain_2880x1920',
    'IntraTrain_4928x3264',
    'IntraValid_768x512',
    'IntraValid_1536x1024',
    'IntraValid_2880x1920',
    'IntraValid_4928x3264',
    'IntraTest_768x512',
    'IntraTest_1536x1024',
    'IntraTest_2880x1920',
    'IntraTest_4928x3264',
]
QP_LIST = [22, 27, 32, 37]
YUV_WIDTH_LIST_FULL = np.ravel(
    np.array([[768, 1536, 2880, 4928, 768, 1536, 2880, 4928, 768, 1536, 2880, 4928]])).astype(int)
YUV_HEIGHT_LIST_FULL = np.ravel(
    np.array([[512, 1024, 1920, 3264, 512, 1024, 1920, 3264, 512, 1024, 1920, 3264]])).astype(int)

INDEX_LIST_TRAIN = list(range(0, 4))
INDEX_LIST_VALID = list(range(4, 8))
INDEX_LIST_TEST = list(range(8, 12))


class FrameYUV:
    def __init__(self, Y, U, V) -> None:
        self._Y = Y
        self._U = U
        self._V = V


def get_file_size(path):
    try:
        size = os.path.getsize(path)
        return size
    except Exception as err:
        print(err)


def get_num_YUV420_frame(file, width, height):
    file_bytes = get_file_size(file)
    frame_bytes = width * height * 3 // 2
    assert(file_bytes % frame_bytes == 0)
    return file_bytes // frame_bytes


def read_YUV420_frame(fid, width, height):
    d00 = height // 2
    d01 = width // 2
    Y_buf = fid.read(width * height)
    Y = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [height, width])
    U_buf = fid.read(d01 * d00)
    U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [d00, d01])
    V_buf = fid.read(d01 * d00)
    V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [d00, d01])
    return FrameYUV(Y, U, V)


def get_file_list(index_list, yuv_path_ori=YUV_PATH, info_path=INFO_PATH, yuv_name_list_full=YUV_NAME_LIST_FULL, qp_list=QP_LIST):
    yuv_name_list = []
    yuv_file_list = []
    info_file_list = []
    for index in index_list:
        yuv_name_list.append(yuv_name_list_full[index])

    for i_qp in range(len(qp_list)):
        info_file_list.append([])
        for i_seq in range(len(index_list)):
            info_file_list[i_qp].append([])

    for i_seq in range(len(index_list)):
        yuv_file_list.append([])

    for i_seq in range(len(index_list)):
        yuv_file_temp = glob.glob(yuv_path_ori + yuv_name_list[i_seq] + '.yuv')
        assert len(yuv_file_temp) == 1
        yuv_file_list[i_seq] = yuv_file_temp[0]

        for i_qp in range(len(qp_list)):
            info_file_temp = glob.glob(info_path + 'Info*_' +
                                       yuv_name_list[i_seq] + '_*qp' + str(qp_list[i_qp]) + '*CUDepth.dat')
            assert len(info_file_temp) == 1
            info_file_list[i_qp][i_seq] = info_file_temp[0]

    return yuv_file_list, info_file_list


def read_info_frame(fid, width, height, mode):
    # read information of CU/TU partition
    assert(width % 8 == 0 and height % 8 == 0)
    if mode == 'CU':
        unit_width = 16
    elif mode == 'TU':
        unit_width = 8
    num_line_in_unit = height // unit_width
    num_column_in_unit = width // unit_width

    info_buf = fid.read(num_line_in_unit * num_column_in_unit)
    info = np.reshape(np.frombuffer(info_buf, dtype=np.uint8), [num_line_in_unit, num_column_in_unit])
    return info


def write_data(fid_out, frame_Y, qp_list, cu_depth_mat_list):
    width = np.shape(frame_Y)[1]
    height = np.shape(frame_Y)[0]
    assert(len(qp_list) == len(cu_depth_mat_list))
    n_qp = len(qp_list)
    n_line = height // 64
    n_col = width // 64
    for i_line in range(n_line):
        for i_col in range(n_col):
            buf_sample = (np.ones((4096 + 64 + 16 * 52,)) * 255).astype(np.uint8)
            patch_Y = frame_Y[i_line * 64: (i_line + 1) * 64, i_col * 64: (i_col + 1) * 64]
            buf_sample[0: 4096] = np.reshape(patch_Y, (4096,))
            for i_qp in range(n_qp):
                patch_cu_depth = cu_depth_mat_list[i_qp][i_line * 4: (i_line + 1) * 4, i_col * 4: (i_col + 1) * 4]
                i_start_in_buf = 4096 + 64 + qp_list[i_qp] * 16
                buf_sample[i_start_in_buf: i_start_in_buf + 16] = np.reshape(patch_cu_depth, (16,))
            fid_out.write(buf_sample)
    return n_line * n_col


def shuffle_samples(file, sample_length):
    file_bytes = get_file_size(file)
    assert(file_bytes % sample_length == 0)
    num_samples = file_bytes // sample_length
    index_list = random.sample(range(num_samples), num_samples)
    fid_in = open(file, 'rb')
    fid_out = open('./dataset_shuffled/'+file + '_shuffled', 'wb')
    for i in range(num_samples):
        fid_in.seek(index_list[i] * sample_length, 0)
        info_buf = fid_in.read(sample_length)
        fid_out.write(info_buf)
        if (i + 1) % 100 == 0:
            print('%s : %d / %d samples completed.' % (file, i + 1, num_samples))
    fid_in.close()
    fid_out.close()


def generate_data(index_list, save_file, yuv_path_ori=YUV_PATH, info_path=INFO_PATH, yuv_name_list_full=YUV_NAME_LIST_FULL, qp_list=QP_LIST,
                  yuv_width_list_full=YUV_WIDTH_LIST_FULL, yuv_height_list_full=YUV_HEIGHT_LIST_FULL):

    yuv_file_list, info_file_list = get_file_list(index_list)
    yuv_width_list = yuv_width_list_full[index_list]
    yuv_height_list = yuv_height_list_full[index_list]
    n_seq = len(yuv_file_list)
    n_qp = len(qp_list)

    fid_out = open(save_file, 'wb+')

    n_sample = 0
    for i_seq in range(n_seq):
        width = yuv_width_list[i_seq]
        height = yuv_height_list[i_seq]
        n_frame = get_num_YUV420_frame(yuv_file_list[i_seq], width, height)

        fid_yuv = open(yuv_file_list[i_seq], 'rb')
        fid_info_list = []
        for i_qp in range(n_qp):
            fid_info_list.append(open(info_file_list[i_qp][i_seq], 'rb'))

        for i_frame in range(n_frame):
            frame_YUV = read_YUV420_frame(fid_yuv, width, height)
            frame_Y = frame_YUV._Y
            cu_depth_mat_list = []
            for i_qp in range(n_qp):
                cu_depth_mat_list.append(read_info_frame(fid_info_list[i_qp], width, height, 'CU'))
            n_sample_one_frame = write_data(fid_out, frame_Y, qp_list, cu_depth_mat_list)
            n_sample += n_sample_one_frame
            print('Seq. %d / %d, %50s : frame %d / %d, %8d samples generated.' %
                  (i_seq + 1, n_seq, yuv_file_list[i_seq], i_frame + 1, n_frame, n_sample))

        fid_yuv.close()
        for i_qp in range(n_qp):
            fid_info_list[i_qp].close()

    fid_out.close()

    save_file_renamed = 'AI_%s_%d.dat' % (save_file, n_sample)
    os.rename(save_file, save_file_renamed)
    shuffle_samples(save_file_renamed, 4992)


if __name__ == '__main__':
    generate_data(INDEX_LIST_TRAIN, 'Train')

    generate_data(INDEX_LIST_VALID, 'Valid')

    generate_data(INDEX_LIST_TEST, 'Test')
