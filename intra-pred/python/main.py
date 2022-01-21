
import cv2
import numpy as np
import tensorflow as tf
import jax,sys
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
            pad = img[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE, j * BLOCK_SIZE:(j + 1) * BLOCK_SIZE]
            left, top = reference(img, i, j, i_max, j_max, BLOCK_SIZE)
            left = np.insert(np.float64(left), 0, 0, axis=0)
            top = np.float64(top)
            pad = np.insert(pad, 0, top, axis=0)
            pad = np.insert(pad, 0, left, axis=1)
            pad = np.lib.pad(pad, [(0, 1), (0, 1)], 'constant', constant_values=0)
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


def intra_pred_gpu(block, BLOCK_SIZE=4):
    output = []
    out_mode = []
    block_num = len(block)

    for i in (range(block_num)):
        img = np.float64(block[i])

        psnr_max = 0
        block_max = 0
        mode_max = 0

        for mode in range(1, 35):

            delta = direction(mode)
            delta_right = np.copy(delta)
            delta_right[:, 1] += delta_right[:, 2]
            delta_bottom = np.copy(delta)
            delta_bottom[1, :] += delta_bottom[2, :]
            delta_last = np.copy(delta_bottom)
            delta_last[:, 1] += delta_last[:, 2]

            # A, B
            A = np.zeros(BLOCK_SIZE**4).astype(np.float64).reshape(BLOCK_SIZE**2, BLOCK_SIZE**2)
            B = np.zeros(BLOCK_SIZE**2).astype(np.float64)

            for j in range(1, BLOCK_SIZE + 1):
                if j == BLOCK_SIZE:
                    Delta = delta_bottom
                else:
                    Delta = delta
                for k in range(1, BLOCK_SIZE + 1):
                    if k == BLOCK_SIZE:
                        if j != BLOCK_SIZE:
                            Delta = delta_right
                        else:
                            Delta = delta_last
                    result = Delta * img[j-1:j+2, k-1:k+2]

                    ind_j = (j - 1) * BLOCK_SIZE + (k - 1)
                    # Calculating B[ind_j]
                    # Top References
                    if j == 1:
                        B[ind_j] += -np.sum(result[0, :])
                        # print ind_j, j

                    # Left References
                    if k == 1:
                        B[ind_j] += -np.sum(result[:, 0])
                        # print ind_j, k

                    ind_k = ind_j

                    # Calculating A[ind_j, :]
                    if j == 1 and k == 1:
                        A[ind_j, ind_k:ind_k+2] = Delta[1, 1:3]
                        ind_k_bottom = ind_k + BLOCK_SIZE
                        A[ind_j, ind_k_bottom:ind_k_bottom+2] = Delta[2, 1:3]
                    elif j == 1 and k != 1:
                        ind_k_bottom = ind_k + BLOCK_SIZE
                        if k != BLOCK_SIZE:
                            A[ind_j, ind_k-1:ind_k+2] = Delta[1, :]
                            A[ind_j, ind_k_bottom-1:ind_k_bottom+2] = Delta[2, :]
                        else:
                            A[ind_j, ind_k-1:ind_k+1] = Delta[1, 0:2]
                            A[ind_j, ind_k_bottom-1:ind_k_bottom+1] = Delta[2, 0:2]
                    elif j != 1 and k == 1:
                        A[ind_j, ind_k:ind_k+2] = Delta[1, 1:3]
                        ind_k_top = ind_k - BLOCK_SIZE
                        A[ind_j, ind_k_top:ind_k_top+2] = Delta[0, 1:3]
                        if j != BLOCK_SIZE:
                            ind_k_bottom = ind_k + BLOCK_SIZE
                            A[ind_j, ind_k_bottom:ind_k_bottom+2] = Delta[2, 1:3]
                    else:
                        ind_k_top = ind_k - BLOCK_SIZE
                        ind_k_bottom = ind_k + BLOCK_SIZE
                        if j == BLOCK_SIZE and k == BLOCK_SIZE:
                            A[ind_j, ind_k-1:ind_k+1] = Delta[1, 0:2]
                            A[ind_j, ind_k_top-1:ind_k_top+1] = Delta[0, 0:2]
                        elif j != BLOCK_SIZE and k == BLOCK_SIZE:
                            A[ind_j, ind_k-1:ind_k+1] = Delta[1, 0:2]
                            A[ind_j, ind_k_top-1:ind_k_top+1] = Delta[0, 0:2]
                            A[ind_j, ind_k_bottom-1:ind_k_bottom+1] = Delta[2, 0:2]
                        elif j == BLOCK_SIZE and k != BLOCK_SIZE:
                            A[ind_j, ind_k-1:ind_k+2] = Delta[1, :]
                            A[ind_j, ind_k_top-1:ind_k_top+2] = Delta[0, :]
                        else:
                            A[ind_j, ind_k-1:ind_k+2] = Delta[1, :]
                            A[ind_j, ind_k_top-1:ind_k_top+2] = Delta[0, :]
                            A[ind_j, ind_k_bottom-1:ind_k_bottom+2] = Delta[2, :]

            predict = jax.numpy.linalg.solve(A, B).reshape(BLOCK_SIZE, BLOCK_SIZE)
            if mode == 1:
                block_max = predict
                psnr_max = PSNR(img[1:BLOCK_SIZE+1, 1:BLOCK_SIZE+1], block_max, BLOCK_SIZE)
                mode_max = 1

            else:
                psnr = PSNR(img[1:BLOCK_SIZE+1, 1:BLOCK_SIZE+1], predict, BLOCK_SIZE)
                # print psnr
                if psnr > psnr_max:
                    psnr_max = psnr
                    block_max = predict
                    mode_max = mode
                    
        output.append(block_max)
        out_mode.append(mode_max)

        print('\rProgress:	', i + 1, '/', block_num,end='')
        sys.stdout.flush()

    return output, out_mode


if __name__ == '__main__':
    size = [4,8,16,32]
    # img = np.ones((2, 8, 8), dtype=np.float64)

    # print(add_padding(img))
    # intra_pred_gpu(img)

    img = cv2.imread('block_threads.png', 0)
    img = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
    cv2.imshow('img', img)
    if cv2.waitKey(0) or 0xFF == ord('q'):
        cv2.destroyAllWindows()

    shape = img.shape
    blocks = break_block(img, 8)

    blocks, modes = intra_pred_gpu(blocks, BLOCK_SIZE=size[1])

    img = group_blocks(blocks, shape,size[1])
    # cv2.imshow('img', img)
    cv2.imwrite('block_threads_pred.png', img)