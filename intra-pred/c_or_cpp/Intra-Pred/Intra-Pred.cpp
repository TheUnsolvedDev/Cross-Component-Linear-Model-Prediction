
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cublas_api.h>

using namespace cv;
using namespace std;

int **direction(int mode)
{
    int **matrix = (int **)calloc(3, sizeof(int *));
    for (int i = 0; i < 3; i++)
    {
        matrix[i] = (int *)calloc(3, sizeof(int));
    }

    int d_vector[9] = {0, 2, 5, 9, 13, 17, 21, 26, 32};
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (i == 1 || j == 1)
            {
                matrix[i][j] = -32;
                if (i == 1 && j == 1)
                {
                    matrix[i][j] = 128;
                }
            }
            else
            {
                matrix[i][j] = 0;
            }
        }
    }

    if (mode >= 2 && mode <= 10)
    {
        matrix[1][0] += -32;
        matrix[1][1] += 32 + d_vector[10 - mode];
        matrix[2][1] += -d_vector[10 - mode];
    }
    else if (mode >= 11 && mode <= 17)
    {
        matrix[0][1] += -d_vector[mode - 10];
        matrix[1][1] += -32;
        matrix[1][1] += 32 + d_vector[mode - 10];
    }
    else if (mode >= 18 && mode <= 26)
    {
        matrix[0][1] += -32;
        matrix[1][0] += -d_vector[26 - mode];
        matrix[1][1] += 32 + d_vector[26 - mode];
    }
    else if (mode >= 27 && mode <= 34)
    {
        matrix[0][1] += -32;
        matrix[1][1] += 32 + d_vector[mode - 26];
        matrix[1][2] += -d_vector[mode - 26];
    }
    else
    {
        throw "Invalid Mode";
    }

    return matrix;
}

int **pad(int **block, int size)
{
    int rows = size + 2;
    int cols = size + 2;

    int **mat_pad = (int **)calloc(rows, sizeof(int *));
    for (int i = 0; i < rows; i++)
    {
        mat_pad[i] = (int *)calloc(cols, sizeof(int));
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (i == 0 && j == 0)
            {
                mat_pad[i][j] = 0;
            }
            else if (i == 0 || j == 0)
            {
                mat_pad[i][j] = 128;
            }
            else if (i == rows - 1 || j == cols - 1)
            {
                mat_pad[i][j] = 0;
            }
            else
            {
                mat_pad[i][j] = block[i - 1][j - 1];
            }
        }
    }
    mat_pad[0][cols - 1] = 0;
    mat_pad[rows - 1][0] = 0;
    return mat_pad;
}

void free_matrix(int **matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

int ***break_blocks(int **image, int rows, int columns, int size)
{
    int row_max = floor(rows / size);
    int col_max = floor(columns / size);

    int ***blocks = (int ***)calloc(row_max * col_max, sizeof(int **));
    for (int i = 0; i < row_max * col_max; i++)
    {
        blocks[i] = (int **)calloc((size + 2), sizeof(int *));
        for (int j = 0; j < (size + 2); j++)
        {
            blocks[i][j] = (int *)calloc((size + 2), sizeof(int));
        }
    }

    int **temp_block = (int **)calloc((size), sizeof(int *));
    int **temp_pad = NULL;
    for (int i = 0; i < (size); i++)
    {
        temp_block[i] = (int *)calloc((size), sizeof(int));
    }

    cout << row_max << " " << col_max << endl;
    for (int i = 0; i < row_max; i++)
    {
        for (int j = 0; j < col_max; j++)
        {
            for (int k = 0; k < size; k++)
            {
                for (int l = 0; l < size; l++)
                {
                    temp_block[k][l] = image[i * size + k][j * size + l];
                }
            }

            temp_pad = pad(temp_block, size);

            for (int k = 0; k < size + 2; k++)
            {
                for (int l = 0; l < size + 2; l++)
                {
                    blocks[i * col_max + j][k][l] = temp_pad[k][l];
                }
            }
        }
    }

    return blocks;
}

int **group_block(int ***block, int size, int rows, int columns)
{
    int row_max = floor(rows / size);
    int col_max = floor(columns / size);

    int **frame = (int **)calloc(row_max * size, sizeof(int *));
    for (int i = 0; i < row_max * size; i++)
    {
        frame[i] = (int *)calloc(col_max * size, sizeof(int));
    }

    for (int i = 0; i < row_max; i++)
    {
        for (int j = 0; j < col_max; j++)
        {
            for (int k = 0; k < size; k++)
            {
                for (int l = 0; l < size; l++)
                {
                    frame[i * size + k][j * size + l] = block[i * col_max + j][k][l];
                }
            }
        }
    }

    return frame;
}

int **convert_mat_to_array(Mat image)
{
    int **frame = (int **)calloc((int)image.rows, sizeof(int *));
    for (int i = 0; i < image.rows; i++)
    {
        frame[i] = (int *)calloc((int)image.cols, sizeof(int));
    }

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            frame[i][j] = image.at<uchar>(i, j);
        }
    }

    return frame;
}

Mat convert_array_to_mat(int **frame, int rows, int cols, int size)
{
    int row_max = floor(rows / size);
    int col_max = floor(cols / size);

    int row = row_max * size;
    int col = col_max * size;
    Mat image(row, col, CV_8UC1);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < col; j++)
        {
            image.at<uchar>(i, j) = (unsigned int)frame[i][j];
        }
    }
    return image;
}

int **right(int **delta)
{
    int **temp = delta;
    for (int i = 0; i < 3; i++)
    {
        temp[i][1] += temp[i][2];
    }
    return temp;
}

int **bottom(int **delta)
{
    int **temp = delta;
    for (int i = 0; i < 3; i++)
    {
        temp[1][i] += temp[1][i];
    }
    return temp;
}

int **last(int **delta_bottom)
{
    int **temp = delta_bottom;
    for (int i = 0; i < 3; i++)
    {
        temp[i][1] += temp[i][2];
    }
    return temp;
}

float PSNR(int **block1, int **block2, int size)
{
    float mse = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            mse += pow(block1[i][j] - block2[i][j], 2);
        }
    }
    mse /= (size * size);
    return 10 * log10(255 * 255 / mse);
}

void intra_pred(int ***blocks, int **frame, int **modes, int rows, int cols, int size)
{
    int row_max = floor(rows / size);
    int col_max = floor(cols / size);
    int block_num = row_max * col_max;
    int **temp_block, psnr_max = 0, mode_max = 0, block_max = 0;
    int total_modes = 35;
    int **delta, **delta_right, **delta_bottom, **delta_last, **Delta;
    int idx_r, idx_c, ind_r, ind_c, ind_c_bottom, ind_c_top;

    int **result = (int **)calloc(3, sizeof(int));
    for (int i = 0; i < 3; i++)
    {
        result[i] = (int *)calloc(3, sizeof(int));
    }
    int **A = (int **)calloc((size * size), sizeof(int *));
    for (int i = 0; i < (size * size); i++)
    {
        A[i] = (int *)calloc((size * size), sizeof(int));
    }
    int *B = (int *)calloc((size * size), sizeof(int));

    for (int i = 0; i < block_num; i++)
    {
        temp_block = blocks[i];
        psnr_max = 0;
        block_max = 0;
        mode_max = 0;

        for (int mode = 1; mode < total_modes; i++)
        {
            delta = direction(mode);
            delta_right = right(delta);
            delta_bottom = bottom(delta);
            delta_last = last(delta_bottom);

            for (int r = 1; r < size + 1; r++)
            {
                if (r == size)
                {
                    Delta = delta;
                }
                else
                {
                    Delta = delta_bottom;
                }
                for (int c = 1; c < size + 1; c++)
                {
                    if (c == size)
                    {
                        if (r != size)
                        {
                            Delta = delta_right;
                        }
                        else
                        {
                            Delta = delta_last;
                        }
                    }

                    // producting a patch from the image block
                    for (int l1 = r - 1; l1 < r + 2; l1++)
                    {
                        for (int l2 = c - 1; l2 < c + 2; l2++)
                        {
                            idx_r = l1 % 3;
                            idx_c = l2 % 3;
                            result[idx_r][idx_c] = Delta[idx_c][idx_r] * temp_block[l1][l2];
                        }
                    }

                    ind_r = (r - 1) * size + (c - 1);

                    if (r == 1)
                    {
                        for (int l1 = 0; l1 < 3; l1++)
                        {
                            B[ind_r] += result[l1][0];
                        }
                    }

                    if (c == 1)
                    {
                        for (int l1 = 0; l1 < 3; l1++)
                        {
                            B[ind_r] += result[0][l1];
                        }
                    }

                    ind_c = ind_r;

                    if (r == 1 && c == 1)
                    {
                        for (int l1 = ind_c, l2 = 1; l1 < ind_c + 2, l2 < 3; l1++, l2++)
                        {
                            A[ind_r][l1] = Delta[1][l2];
                        }
                        ind_c_bottom = ind_c + size;
                        for (int l1 = ind_c_bottom, l2 = 1; l1 < ind_c_bottom + 2, l2 < 3; l1++, l2++)
                        {
                            A[ind_r][l1] = Delta[2][l2];
                        }
                    }

                    else if (r == 1 && c != 1)
                    {
                        ind_c_bottom = ind_c + size;
                        if (c != size)
                        {
                            for (int l1 = ind_c - 1, l2 = 0; l1 < ind_c + 2, l2 < 3; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[1][l2];
                            }
                            for (int l1 = ind_c_bottom - 1, l2 = 0; l1 < ind_c_bottom + 2, l2 < 3; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[2][l2];
                            }
                        }
                        else
                        {
                            for (int l1 = ind_c - 1, l2 = 0; l1 < ind_c + 1, l2 < 2; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[1][l2];
                            }
                            for (int l1 = ind_c_bottom - 1, l2 = 0; l1 < ind_c_bottom + 1, l2 < 2; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[2][l2];
                            }
                        }
                    }
                    else if (r != 1 and c == 1)
                    {
                        for (int l1 = ind_c, l2 = 1; l1 < ind_c + 2, l2 < 3; l1++, l2++)
                        {
                            A[ind_r][l1] = Delta[1][l2];
                        }
                        ind_c_top = ind_c - size;
                        for (int l1 = ind_c_top, l2 = 1; l1 < ind_c_top + 2, l2 < 3; l1++, l2++)
                        {
                            A[ind_r][l1] = Delta[0][l2];
                        }
                        if (r != size)
                        {
                            ind_c_bottom = ind_c + size;
                            for (int l1 = ind_c_bottom, l2 = 1; l1 < ind_c_bottom + 2, l2 < 3; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[2][l2];
                            }
                        }
                    }
                    else
                    {
                        ind_c_top = ind_c - size;
                        ind_c_bottom = ind_c + size;

                        if (r == size && c == size)
                        {
                            for (int l1 = ind_c - 1, l2 = 0; l1 < ind_c + 1, l2 < 2; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[1][l2];
                            }
                            for (int l1 = ind_c_top - 1, l2 = 0; l1 < ind_c_top + 1, l2 < 2; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[0][l2];
                            }
                        }
                        else if (r != size && c == size)
                        {
                            for (int l1 = ind_c - 1, l2 = 0; l1 < ind_c + 1, l2 < 2; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[1][l2];
                            }
                            for (int l1 = ind_c_top - 1, l2 = 0; l1 < ind_c_top + 1, l2 < 2; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[0][l2];
                            }
                            for (int l1 = ind_c_bottom - 1, l2 = 0; l1 < ind_c_bottom + 1, l2 < 2; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[2][l2];
                            }
                        }
                        else if (r == size && c != size)
                        {
                            for (int l1 = ind_c - 1, l2 = 0; l1 < ind_c + 2, l2 < 3; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[1][l2];
                            }
                            for (int l1 = ind_c_top - 1, l2 = 0; l1 < ind_c_top + 2, l2 < 3; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[0][l2];
                            }
                        }
                        else
                        {
                            for (int l1 = ind_c - 1, l2 = 0; l1 < ind_c + 2, l2 < 3; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[1][l2];
                            }
                            for (int l1 = ind_c_top - 1, l2 = 0; l1 < ind_c_top + 2, l2 < 3; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[0][l2];
                            }
                            for (int l1 = ind_c_bottom - 1, l2 = 0; l1 < ind_c_bottom + 2, l2 < 3; l1++, l2++)
                            {
                                A[ind_r][l1] = Delta[2][l2];
                            }
                        }
                    }
                }
            }
        }
    }
}

float *solve(int **Aarray,int *Barray,int size)
{
    float *A = (float *)malloc(sizeof(float) * size * size * size * size);
    float *B = (float *)malloc(sizeof(float) * size * size);
    
    for(int i = 0;i < size*size;i++)
    {
        for(int j = 0;j < size*size;j++)
        {
            A[i*size*size+j] = Aarray[i][j];
        }
    }
    
    for(int i = 0;i < size*size;i++)
    {
        B[i] = Barray[i];
    }
    
    cublasStatus_t status;
    cudaError_t error;
    cusolverStatus_t cusolver_status;
    cublasHandle_t handle;
    cusolverDnHandle_t cusolver_handle;
    
    int N = size*size;
    float *A, *B1, *B;
    
    // A = N*N matrix
    // B = A*B1 = N*1 matrix
    
    float *d_A, *d_work, *d_B;
    int *d_pivot, *d_info, work;
    
    int info_gpu = 0;
    B1 = (float *)malloc(sizeof(float) * N);
    for(int i = 0;i < N;i++)
    {
        B1[i] = 0.0;
    }

    float alpha = 1.0, beta = 0.0;
    // B = A*B1
    cublasSgemv(handle,CUBLAS_OP_N,N,N,&alpha,A,N,B1,1,&beta,B,1);
    error = cudaGetDevice(0);
    
    cusolver_status = cusolverDnCreate(&cusolver_handle);
    
    //prepare memory on device
    cudaMalloc((void **)&d_A, N*N*sizeof(float));
    cudaMalloc((void **)&d_B, N*sizeof(float));
    cudaMalloc((void **)&d_pivot, N*sizeof(int));
    cudaMalloc((void **)&d_info, sizeof(int));
    
    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);
    
    //compute buffer size and allocate memory on device
    cusolver_status = cusolverDnSgetrf_bufferSize(cusolver_handle, N, N, d_A, N, &work);
    cudaMalloc((void **)&d_work, work*sizeof(float));
    
    cusolver_status = cusolverDnSgetrf(cusolver_handle, N, N, d_A, N, d_work, d_pivot, d_info);
    cusolver_status = cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N, N, 1, d_A, N, d_pivot, d_B, N, d_info);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(B1, d_B, N*sizeof(float), cudaMemcpyDeviceToHost);


    
    return B1;
}

int main()
{
    Mat image = imread("C:\\Users\\shuvr\\Pictures\\nvidia.jpg", IMREAD_GRAYSCALE);

    if (image.empty())
    {
        cout << "Image File "
             << "Not Found" << endl;

        cin.get();
        return -1;
    }
    imshow("Window Name 1", image);
    waitKey(0);

    int **frame = convert_mat_to_array(image);
    cout << "Image Size: " << image.rows << "x" << image.cols << endl;
    int ***blocks = break_blocks(frame, image.rows, image.cols, 8);
    cout << "Blocks: " << endl;
    int **grouped_blocks = group_block(blocks, 8, image.rows, image.cols);
    cout << "Grouped Blocks: " << endl;
    Mat grouped_image = convert_array_to_mat(frame, image.rows, image.cols, 8);

    imshow("Window Name 2", image);
    waitKey(0);

    free_matrix(frame, image.rows);

    return 0;
}
