#include <opencv2/opencv.hpp>
#include <stdlib.h>

typedef struct matrix2d{
    int **matrix;
    int row;
    int col;
} matrix2d;

typedef struct matrix3d{
    int ***matrix;
    int row;
    int col;
    int depth;
} matrix3d;
