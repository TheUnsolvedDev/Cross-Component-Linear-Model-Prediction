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


def PSNR(a, b):
    return 10 * np.log10(1 / np.mean((a - b) ** 2))


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


def add_padding(block, size=1):
    block = np.pad(block, [1, 1], mode='constant', constant_values=128)
    block[0][0] = block[-1][-1] = 0
    block[-1, :] = block[:, -1] = 0
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


def intra_pred_gpu(block, size=16):
    delta_matrix = []

    for mode in range(1, 35):
        delta = direction(mode)

        delta_right = np.copy(delta)
        delta_right[:, 1] += delta_right[:, 2]

        delta_down = np.copy(delta)
        delta_down[1, :] += delta_down[2, :]

        delta_last = np.copy(delta_down)
        delta_last[:, 1] += delta_last[:, 2]

        delta_matrix.append(np.array([delta, delta_right, delta_down, delta_last]))

    delta_matrix = np.array(delta_matrix)

    A = np.zeros((size**2, size**2), dtype=np.float64)
    B = np.zeros((size**2), dtype=np.float64)


if __name__ == '__main__':
    a = np.array([[2, 1, 2, 1], [0, -9, 0, 9], [0, 1, -1, -5], [0, 1, -3, 0]], dtype=np.float64)
    b = np.array([6, 18, -13, 4], dtype=np.float64)

    # print(GE(a, b))

    img = np.ones((8, 8), dtype=np.float64)
    # intra_pred_gpu(img)
    print(add_padding(img))
    
    image = cv2.imread('block_threads.png', 0)
    cv2.imshow('image', image)
    if cv2.waitKey(0) or 0xFF == ord('q'):
        pass
    
    shape = image.shape
    blocks = break_blocks(image)
    
    image = group_blocks(blocks, shape)
    cv2.imshow('image', image)
    if cv2.waitKey(0) or 0xFF == ord('q'):
        pass
    
    cv2.destroyAllWindows()
    
