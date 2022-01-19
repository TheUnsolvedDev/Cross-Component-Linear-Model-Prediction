#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "pred.h"

using namespace cv;
using namespace std;

int **direction(int mode)
{
    int **matrix = (int **)malloc(sizeof(int *) * 3);
    for (int i = 0; i < 3; i++)
    {
        matrix[i] = (int *)malloc(sizeof(int) * 3);
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

    int **mat_pad = (int **)malloc(sizeof(int *) * rows);
    for (int i = 0; i < cols; i++)
    {
        mat_pad[i] = (int *)malloc(sizeof(int) * cols);
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

int main()
{
    // Mat image = imread("kimono.png", IMREAD_GRAYSCALE);

    // if (image.empty())
    // {
    //     cout << "Image File "
    //          << "Not Found" << endl;

    //     cin.get();
    //     return -1;
    // }
    // imshow("Window Name", image);
    // waitKey(0);

    int **matrix = direction(3);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }

    int **mat_pad = pad(matrix, 3);
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            cout << mat_pad[i][j] << " ";
        }
        cout << endl;
    }

    free_matrix(matrix, 3);
    free_matrix(mat_pad, 5);

    return 0;
}