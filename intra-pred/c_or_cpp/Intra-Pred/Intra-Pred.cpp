
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace cv;
using namespace std;

int** direction(int mode)
{
    int** matrix = (int**)malloc(sizeof(int*) * 3);
    for (int i = 0; i < 3; i++)
    {
        matrix[i] = (int*)malloc(sizeof(int) * 3);
    }

    int d_vector[9] = { 0, 2, 5, 9, 13, 17, 21, 26, 32 };
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

int** pad(int** block, int size)
{
    int rows = size + 2;
    int cols = size + 2;

    int** mat_pad = (int**)malloc(sizeof(int*) * rows);
    for (int i = 0; i < rows; i++)
    {
        mat_pad[i] = (int*)malloc(sizeof(int) * cols);
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

void free_matrix(int** matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

int*** break_blocks(int** image, int rows, int columns, int size)
{
    int row_max = floor(rows / size);
    int col_max = floor(columns / size);

    int*** blocks = (int***)malloc(sizeof(int**) * row_max * col_max);
    for (int i = 0; i < row_max * col_max; i++)
    {
        blocks[i] = (int**)malloc(sizeof(int*) * (size + 2));
        for (int j = 0; j < (size + 2); j++)
        {
            blocks[i][j] = (int*)malloc(sizeof(int) * (size + 2));
        }
    }

    int** temp_block = (int**)malloc(sizeof(int*) * (size));
    int** temp_pad = NULL;
    for (int i = 0; i < (size); i++)
    {
        temp_block[i] = (int*)malloc(sizeof(int) * (size));
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

int** group_block(int*** block, int size, int rows, int columns)
{
    int row_max = floor(rows / size);
    int col_max = floor(columns / size);

    int** frame = (int**)malloc(sizeof(int*) * row_max * size);
    for (int i = 0; i < row_max * size; i++)
    {
        frame[i] = (int*)malloc(sizeof(int) * col_max * size);
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

int** convert_mat_to_array(Mat image)
{
    int** frame = (int**)malloc(sizeof(int*) * (int)image.rows);
    for (int i = 0; i < image.rows; i++)
    {
        frame[i] = (int*)malloc(sizeof(int) * (int)image.cols);
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

Mat convert_array_to_mat(int** frame, int rows, int cols, int size)
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
    int** temp = delta;
    for(int i = 0;i < 3;i++)
    {
        temp[i][1] += temp[i][2];
    }
    return temp;
}

int **bottom(int **delta)
{
    int** temp = delta;
    for(int i = 0;i < 3;i++)
    {
        temp[1][i] += temp[1][i];
    }
    return temp;
}

int **last(int **delta_bottom)
{
    int** temp = delta_bottom;
    for(int i = 0;i < 3;i++)
    {
        temp[i][1] += temp[i][2];
    }
    return temp;
}

void intra_pred(int ***blocks,int **frame,int **modes,int rows,int cols,int size)
{
    int row_max = floor(rows / size);
    int col_max = floor(cols / size);
    int block_num = row_max * col_max;
    int** temp_block,psnr_max = 0,mode_max = 0,block_max = 0;
    int total_modes = 35;
    int** delta,**delta_right,**delta_bottom,**delta_last;

    int** A = (int**)malloc(sizeof(int*)*(size*size));
    for(int i = 0;i < (size*size);i++)
    {
        A[i] = (int*)malloc(sizeof(int) * (size*size));
    }
    int* B = (int*)malloc(sizeof(int) * (size * size));


    for(int i = 0;i < block_num;i++)
    {
        temp_block = blocks[i];
        psnr_max = 0;
        block_max = 0;
        mode_max = 0;

        for(int mode = 1; mode < total_modes;i++)
        {
            delta = direction(mode);
            delta_right = right(delta);
            delta_bottom = bottom(delta);
            delta_last = last(delta_bottom);
        }

    }
}

int main()
{
    Mat image = imread("C:\\Users\\shuvr\\Pictures\\nvidia.jpg",IMREAD_GRAYSCALE);
    
    if (image.empty())
    {
        cout << "Image File "
            << "Not Found" << endl;

        cin.get();
        return -1;
    }
    imshow("Window Name 1", image);
    waitKey(0);

    int** frame = convert_mat_to_array(image);
    cout << "Image Size: " << image.rows << "x" << image.cols << endl;
    int*** blocks = break_blocks(frame, image.rows, image.cols, 8);
    cout << "Blocks: " << endl;
    int** grouped_blocks = group_block(blocks, 8, image.rows, image.cols);
    cout << "Grouped Blocks: " << endl;
    Mat grouped_image = convert_array_to_mat(frame, image.rows, image.cols, 8);

    imshow("Window Name 2", image);
    waitKey(0);

    free_matrix(frame, image.rows);

    return 0;
}
