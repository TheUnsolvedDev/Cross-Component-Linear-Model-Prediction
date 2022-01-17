from ast import Del
from operator import mod
from unittest import result
import cv2
import numpy as np
import tensorflow as tf
import pyopencl as cl
import pyopencl.array

NAME = 'NVIDIA CUDA'
platforms = cl.get_platforms()
devs = None
for platform in platforms:
    if platform.name == NAME:
        devs = platform.get_devices()
        print(devs)

# Set up a command queue:
ctx = cl.Context(devs)
queue = cl.CommandQueue(ctx)

gaussian = cl.Program(ctx, """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void gaussian(__global double *a, __global double *b,
                        __local double *a_loc,
                        const uint i, const uint w)
{
    uint gid            =   get_group_id(0);
    uint lid            =   get_local_id(0);
    double ratio        =   0.0;

    // Group Read
    a_loc[lid]          =   a[i * w + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid == i || lid < i)
        return;

    ratio               =   a[gid * w + i] / a_loc[i];

    a[gid * w + lid]    -=  ratio * a_loc[lid];

    if(lid == i)
        b[gid] -= ratio * b[i];
}
""").build().gaussian

gaussian.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32])


def PSNR(ref, block, BLOCK_SIZE):
    ref = np.float64(ref)
    block = np.float64(block)
    dif = np.sum(pow(ref - block, 2)) / (BLOCK_SIZE * BLOCK_SIZE)
    return 20 * np.log10(255.0 / np.sqrt(dif))


def GE(a, b):
    size = len(a)
    a_gpu = cl.array.to_device(queue, a)
    b_gpu = cl.array.to_device(queue, b)
    a_loc = cl.LocalMemory(np.float64().nbytes * size)

    for i in range(size):
        gaussian(queue, (size ** 2,), (size,), a_gpu.data, b_gpu.data, a_loc, i, size)

    res_b = b_gpu.get()
    res_a = a_gpu.get()
    res = res_b / res_a.diagonal()
    return res


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

# @tf.function


def add_padding(blocks, size=1):
    block = tf.pad(blocks, [[0, 0], [1, 0], [1, 0]], mode='constant', constant_values=128)
    block = tf.pad(block, [[0, 0], [0, 1], [0, 1]], mode='constant', constant_values=0)

    return block


def break_blocks(image, size=16):
    height, width = image.shape
    row_max = height // size
    col_max = width // size

    blocks = []
    for i in range(row_max):
        for j in range(col_max):
            blocks.append(image[i*size:(i+1)*size, j*size:(j+1)*size])
    return np.array(blocks)


def group_blocks(blocks, shape, size=16):
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

# @tf.function


def intra_pred_gpu(blocks, size=16):
    delta_matrix_up = []
    delta_matrix_hor = []
    delta_matrix_ver = []
    delta_matrix_dc = []
    blocks = add_padding(blocks)

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

    for row in range(1, size):
        if row == size:
            Delta = modes_set_hor
        else:
            Delta = modes_set_up
        for col in range(1, size):
            if col == size:
                if row != size:
                    Delta = modes_set_ver
                else:
                    Delta = modes_set_dc

            term1 = tf.cast(Delta, tf.float32)
            term2 = tf.cast(blocks_set[:, :, row-1:row+2, col-1:col+2], tf.float32)
            result = tf.multiply(term1, term2)
            res = result[:, :, 0, :]

            ind_row = (row - 1)*size + col - 1

            if row == 1:
                B[:, :, ind_row] += tf.negative(tf.reduce_sum(result[:, :, 0, :], axis=2)).numpy()
            if col == 1:
                B[:, :, ind_row] += tf.negative(tf.reduce_sum(result[:, :, :, 0], axis=2)).numpy()
                
            ind_col = ind_row
            
            if row == 1 and col == 1:
                A[:, :, ind_row, ind_col:ind_col + 2] = Delta[:, :, 1, 1:3]
                ind_col_bottom = ind_col + size
                A[:, :, ind_row, ind_col_bottom:ind_col_bottom + 2] = Delta[:, :, 2, 1:3]
            elif row == 1 and col != 1:
                ind_col_bottom = ind_col + size
                if col != size:
                    A[:, :, ind_row, ind_col-1:ind_col + 2] = Delta[:, :, 1, :]
                    A[:, :, ind_row, ind_col_bottom-1:ind_col_bottom + 2] = Delta[:, :, 2, :]
                else:
                    A[:, :, ind_row, ind_col-1:ind_col + 1] = Delta[:, :, 1, :2]
                    A[:, :, ind_row, ind_col_bottom-1:ind_col_bottom + 1] = Delta[:, :, 2, :2]
            elif row != 1 and col == 1:
                A[:,:,row,ind_col:ind_col + 2] = Delta[:,:,1,1:3]
                int_col_top = ind_col - size
                A[:,:,row,int_col_top:int_col_top + 2] = Delta[:,:,0,1:3]
                if col != size:
                    ind_col_bottom = ind_col + size
                    A[:,:,row,ind_col_bottom:ind_col_bottom + 2] = Delta[:,:,2,1:3]
                    
            else:
                ind_col_top = ind_col - size
                ind_col_bottom = ind_col + size
                if row == size and col == size:
                    A[:,:,row,ind_col-1:ind_col + 1] = Delta[:,:,1,:2]
                    A[:,:,row,ind_col_top-1:ind_col_top + 1] = Delta[:,:,0,:2]
                elif row != size and col == size:
                    A[:,:,row,ind_col-1:ind_col + 1] = Delta[:,:,1,:2]
                    A[:,:,row,ind_col_top-1:ind_col_top + 1] = Delta[:,:,0,:2]
                    A[:,:,row,ind_col_bottom-1:ind_col_bottom + 1] = Delta[:,:,2,:2]
                elif row == size and col != size:
                    A[:,:,row,ind_col-1:ind_col + 2] = Delta[:,:,1,:2]
                    A[:,:,row,ind_col_top-1:ind_col_top + 2] = Delta[:,:,0,:2]
                else:
                    A[:,:,row,ind_col-1:ind_col + 2] = Delta[:,:,1,:]
                    A[:,:,row,ind_col_top-1:ind_col_top + 2] = Delta[:,:,0,:]
                    A[:,:,row,ind_col_bottom-1:ind_col_bottom + 2] = Delta[:,:,2,:]
    print(A.shape)
    print(B.shape)


if __name__ == '__main__':
    a = np.array([[2, 1, 2, 1], [0, -9, 0, 9], [0, 1, -1, -5], [0, 1, -3, 0]], dtype=np.float64)
    b = np.array([6, 18, -13, 4], dtype=np.float64)

    # print(GE(a, b))

    img = np.ones((2, 8, 8), dtype=np.float64)
    # print(add_padding(img))
    # intra_pred_gpu(img)

    image = cv2.imread('block_threads.png', 0)

    shape = image.shape
    blocks = break_blocks(image)

    intra_pred_gpu(blocks, size=16)
