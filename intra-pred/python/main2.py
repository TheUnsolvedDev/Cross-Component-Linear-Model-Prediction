
import cv2
import numpy as np
import tensorflow as tf
import jax
from tqdm import tqdm


def PSNR(ref, block, BLOCK_SIZE):
    ref = np.float64(ref)
    block = np.float64(block)
    dif = np.sum(pow(ref - block, 2)) / (BLOCK_SIZE * BLOCK_SIZE)
    return 20 * np.log10(255.0 / np.sqrt(dif))


def direction(mode):
    delta = np.array([[0, -32,   0],
                      [-32, 128, -32],
                      [0, -32,   0]])

    # Displacement Vector
    d = [0, 2, 5, 9, 13, 17, 21, 26, 32]

    if mode >= 2 and mode <= 10:
        delta += np.array([[0,            0,  0],
                           [-32, 32+d[10-mode], 0],
                           [0,   -d[10-mode], 0]])
    elif mode >= 11 and mode <= 17:
        delta += np.array([[0,   -d[mode-10], 0],
                           [-32, 32+d[mode-10], 0],
                           [0,             0, 0]])
    elif mode >= 18 and mode <= 26:
        delta += np.array([[0,           -32, 0],
                           [-d[26-mode], 32+d[26-mode], 0],
                           [0,             0, 0]])
    elif mode >= 27 and mode <= 34:
        delta += np.array([[0,           -32,           0],
                           [0, 32+d[mode-26], -d[mode-26]],
                           [0,             0,           0]])
    elif mode > 34 or mode < 1:
        raise ValueError('invalid angular mode!')

    return np.float64(delta)


def pad(img, BLOCK_SIZE):
    h, w = np.shape(img)
    if h % BLOCK_SIZE != 0:
        img = np.lib.pad(img, [(0, BLOCK_SIZE - h % BLOCK_SIZE), (0, 0)], 'constant', constant_values=128)

    if w % BLOCK_SIZE != 0:
        img = np.lib.pad(img, [(0, 0), (0, BLOCK_SIZE - w % BLOCK_SIZE)], 'constant', constant_values=128)

    return img


def reference(img, i_block, j_block, i_max, j_max, BLOCK_SIZE):
    row_ref = [128] * BLOCK_SIZE
    col_ref = [128] * BLOCK_SIZE

    # Dealing with row_ref on the top
    if i_block != 0:
        row_ref = img[i_block * BLOCK_SIZE - 1, j_block * BLOCK_SIZE:(j_block + 1) * BLOCK_SIZE]
    # Dealing with col_ref on the left
    if j_block != 0:
        col_ref = img[i_block * BLOCK_SIZE:(i_block + 1) * BLOCK_SIZE, j_block * BLOCK_SIZE - 1]

    return np.array(col_ref).astype(np.uint8), np.array(row_ref).astype(np.uint8)


def break_block(img, BLOCK_SIZE):
    h, w = np.shape(img)
    i_max = h // BLOCK_SIZE
    j_max = w // BLOCK_SIZE
    block = [0] * int(i_max * j_max)
    pad = 0
    for i in range(i_max):
        for j in range(j_max):
            # Breaking img into blocks
            pad = img[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE, j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
            # Pad image blocks with reference points and zeros
            left, top = reference(img, i, j, i_max, j_max, BLOCK_SIZE)

            left = np.insert(np.float64(left), 0, 0, axis=0)

            top = np.float64(top)

            pad = np.insert(pad, 0, top, axis=0)
            pad = np.insert(pad, 0, left, axis=1)

            pad = np.lib.pad(pad, [(0, 1), (0, 1)], 'constant', constant_values=0)
            # Group small block, left and top reference points together
            block[i * j_max + j] = pad

    return np.array(block)


def group_blocks(blocks, shape, size=4):
    height, width = shape
    row_max = height // size
    col_max = width // size

    image = 0

    for i in range(row_max):
        tile_base = blocks[i*col_max]
        for j in range(col_max):
            tile = blocks[i*col_max+j]
            tile_base = np.concatenate((tile_base, tile), axis=1)
        if i == 0:
            image = tile_base
        else:
            image = np.concatenate((image, tile_base), axis=0)

    return image


def intra_pred_gpu(blocks, size=4):
    delta_matrix_up = []
    delta_matrix_hor = []
    delta_matrix_ver = []
    delta_matrix_dc = []

    length = len(blocks)
    for mode in range(1, 35):
        delta_up = direction(mode)

        delta_horizontal = np.copy(delta_up)
        delta_horizontal[:, 1] += delta_horizontal[:, 2]

        delta_vertical = np.copy(delta_up)
        delta_vertical[1, :] += delta_vertical[2, :]

        delta_last = np.copy(delta_vertical)
        delta_last[:, 1] += delta_last[:, 2]

        delta_matrix_up.append(delta_up)
        delta_matrix_hor.append(delta_horizontal)
        delta_matrix_ver.append(delta_vertical)
        delta_matrix_dc.append(delta_last)

    delta_matrix_up = np.array(delta_matrix_up)
    delta_matrix_hor = np.array(delta_matrix_hor)
    delta_matrix_ver = np.array(delta_matrix_ver)
    delta_matrix_dc = np.array(delta_matrix_dc)

    blocks_set = tf.repeat(blocks[:, tf.newaxis, :, :], repeats=34, axis=1)
    modes_set_up = tf.repeat(delta_matrix_up[tf.newaxis, :, :, :], repeats=length, axis=0)
    modes_set_hor = tf.repeat(delta_matrix_hor[tf.newaxis, :, :, :], repeats=length, axis=0)
    modes_set_ver = tf.repeat(delta_matrix_ver[tf.newaxis, :, :, :], repeats=length, axis=0)
    modes_set_dc = tf.repeat(delta_matrix_dc[tf.newaxis, :, :, :], repeats=length, axis=0)

    A = np.zeros((length, 34, size**2, size**2), dtype=np.float64)
    B = np.zeros((length, 34, size**2), dtype=np.float64)

    for row in range(1, size+1):
        if row == size:
            Delta = modes_set_hor
        else:
            Delta = modes_set_up
        for col in range(1, size+1):
            if col == size:
                if row != size:
                    Delta = modes_set_ver
                else:
                    Delta = modes_set_dc

            term1 = tf.cast(Delta, tf.float32)
            term2 = tf.cast(blocks_set[:, :, row-1:row+2, col-1:col+2], tf.float32)
            result = tf.multiply(term1, term2)
            res = result[:, :, 0, :]

            ind_j = (row - 1)*size + (col - 1)

            if row == 1:
                B[:, :, ind_j] += tf.negative(tf.reduce_sum(result[:, :, 0, :], axis=2)).numpy()
            if col == 1:
                B[:, :, ind_j] += tf.negative(tf.reduce_sum(result[:, :, :, 0], axis=2)).numpy()

            ind_k = ind_j
            j, k = row, col

            if j == 1 and k == 1:
                A[:, :, ind_j, ind_k:ind_k+2] = Delta[:, :, 1, 1:3]
                ind_k_bottom = ind_k + size
                A[:, :, ind_j, ind_k_bottom:ind_k_bottom+2] = Delta[:, :, 2, 1:3]
            elif j == 1 and k != 1:
                ind_k_bottom = ind_k + size
                if k != size:
                    A[:, :, ind_j, ind_k-1:ind_k+2] = Delta[:, :, 1, :]
                    A[:, :, ind_j, ind_k_bottom-1:ind_k_bottom+2] = Delta[:, :, 2, :]
                else:
                    A[:, :, ind_j, ind_k-1:ind_k+1] = Delta[:, :, 1, 0:2]
                    A[:, :, ind_j, ind_k_bottom-1:ind_k_bottom+1] = Delta[:, :, 2, 0:2]
            elif j != 1 and k == 1:
                A[:, :, ind_j, ind_k:ind_k+2] = Delta[:, :, 1, 1:3]
                ind_k_top = ind_k - size
                A[:, :, ind_j, ind_k_top:ind_k_top+2] = Delta[:, :, 0, 1:3]
                if j != size:
                    ind_k_bottom = ind_k + size
                    A[:, :, ind_j, ind_k_bottom:ind_k_bottom+2] = Delta[:, :, 2, 1:3]
            else:
                ind_k_top = ind_k - size
                ind_k_bottom = ind_k + size
                if j == size and k == size:
                    A[:, :, ind_j, ind_k-1:ind_k+1] = Delta[:, :, 1, 0:2]
                    A[:, :, ind_j, ind_k_top-1:ind_k_top+1] = Delta[:, :, 0, 0:2]
                elif j != size and k == size:
                    A[:, :, ind_j, ind_k-1:ind_k+1] = Delta[:, :, 1, 0:2]
                    A[:, :, ind_j, ind_k_top-1:ind_k_top+1] = Delta[:, :, 0, 0:2]
                    A[:, :, ind_j, ind_k_bottom-1:ind_k_bottom+1] = Delta[:, :, 2, 0:2]
                elif j == size and k != size:
                    A[:, :, ind_j, ind_k-1:ind_k+2] = Delta[:, :, 1, :]
                    A[:, :, ind_j, ind_k_top-1:ind_k_top+2] = Delta[:, :, 0, :]
                else:
                    A[:, :, ind_j, ind_k-1:ind_k+2] = Delta[:, :, 1, :]
                    A[:, :, ind_j, ind_k_top-1:ind_k_top+2] = Delta[:, :, 0, :]
                    A[:, :, ind_j, ind_k_bottom-1:ind_k_bottom+2] = Delta[:, :, 2, :]

    output_blocks, output_modes = [], []

    for i in tqdm(range(length)):
        for j in range(34):
            term1 = np.array(A[i][j])
            term2 = np.array(B[i][j])
            res = jax.numpy.linalg.solve(term1, term2).reshape(size, size)

            if j == 0:
                block_max = res
                psnr_max = PSNR(blocks_set[i][j][1:size+1, 1:size+1], res, size)
                mode_max = 1

            else:
                psnr = PSNR(blocks_set[i][j][1:size+1, 1:size+1], res, size)
                if psnr > psnr_max:
                    block_max = res
                    psnr_max = psnr
                    mode_max = j+1

        output_blocks.append(block_max)
        output_modes.append(mode_max)

    output_blocks = np.array(output_blocks)
    output_modes = np.array(output_modes)

    return output_blocks, output_modes


if __name__ == '__main__':
    a = np.array([[2, 1, 2, 1], [0, -9, 0, 9], [0, 1, -1, -5], [0, 1, -3, 0]], dtype=np.float64)
    b = np.array([6, 18, -13, 4], dtype=np.float64)

    print(GE(a, b))

    # img = np.ones((2, 8, 8), dtype=np.float64)

    # print(add_padding(img))
    # intra_pred_gpu(img)

    img = cv2.imread('kimono.png', 0)
    img = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
    cv2.imshow('img', img)
    if cv2.waitKey(0) or 0xFF == ord('q'):
        cv2.destroyAllWindows()

    shape = img.shape
    blocks = break_block(img, 4)

    blocks, modes = intra_pred_gpu(blocks, size=4)

    img = group_blocks(blocks, shape)
    cv2.imshow('img', img)
    if cv2.waitKey(0) or 0xFF == ord('q'):
        cv2.destroyAllWindows()
