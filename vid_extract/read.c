#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

void int2str(int xx, char *mstr)
{
    int remainder = 1;
    int ii = 0;
    while (*mstr)
        mstr++;     
    mstr = mstr - 5;
    for (ii = 0; ii < 9; ii++)
    {
        remainder = xx % 10;
        xx = xx / 10;
        *mstr = '0' + remainder;
        mstr--;
    }
}

void print_header(FILE *fout, int row, int col)
{
    fprintf(fout, "P2\n");
    fprintf(fout, "%d %d\n", col, row);
    fprintf(fout, "255\n");
}

void print_luma(FILE *fout, unsigned char luma[][176], int row, int col)
{
    int ii = 0;
    int jj = 0;
    int *temp;
    for (ii = 0; ii < row; ii++)
    {
        for (jj = 0; jj < col; jj++)
        {
            fprintf(fout, "%d\n", *(*luma + jj));
        }
        luma++;
    }
}

int get_frame_num(char *filename, int height, int width)
{
    int y_size = height * width;
    int uv_size = y_size / 4;
    int data = y_size + uv_size * 2;

    FILE *fp = fopen(filename, "rb");
    fseek(fp, 0, 2);
    long file_size = ftell(fp);

    int num_frames = file_size / data;
    fclose(fp);
    return num_frames;
}

int main(int argc, char *argv[])
{
    FILE *fptr, *fr;
    char filename[20];
    int info = 0, write_file = 1;

    unsigned char def_name[] = "default.pgm";
    char *mstr = NULL;
    mstr = def_name;

    if (argc == 1)
    {
        printf("No file name given.\n");
        return -1;
    }
    else
    {
        printf("Reading '%s' ....\n", argv[1]);
    }

    fptr = fopen(argv[1], "rb");
    if (fptr == NULL)
    {
        printf("File not found.\n");
        return -1;
    }
    else
    {
        printf("File found.\n");
    }
    fclose(fptr);

    int height = atoi(argv[2]), width = atoi(argv[3]);

    printf("Frame height: %d\n", height);
    printf("Frame width: %d\n", width);

    int num_frames = get_frame_num(argv[1], height, width);
    // printf("Number of frames: %d\n", num_frames);

    unsigned char frames_luma[144][176];
    unsigned char frames_cb[88][72];
    unsigned char frames_cr[88][72];
    printf("Number of frames: %d\n", num_frames);

    fptr = fopen(argv[1], "rb");
    printf("Reading frames ....\n");

    for (int i = 0; i < num_frames; i++)
    {
        fread(frames_luma, 1, height * width, fptr);
        fread(frames_cb, 1,88*72, fptr);
        fread(frames_cr, 1, 88*72, fptr);

        if (write_file)
        {
            int2str(i, mstr);
            // printf("Output file: %s\n", mstr);
            if (!(fr = fopen(mstr, "w")))
            {
                printf("Cannot open file %s Exiting . . . .\n", mstr);
                exit(-1);
            }
            print_header(fr, height, width);
            print_luma(fr, frames_luma, height, width);
        }
        printf("Frame %d done.\n", i);
    }

    return 0;
}