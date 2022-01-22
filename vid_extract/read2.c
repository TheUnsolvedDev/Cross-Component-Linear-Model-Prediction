#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct frame
{
    int height;
    int width;
    unsigned char **frame_luma;
    unsigned char **frame_cb;
    unsigned char **frame_cr;
} frame;

int height = 0;
int width = 0;

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

void print_luma(FILE *fout, frame *f, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            fprintf(fout, "%d\n", f->frame_luma[i][j]);
        }
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

void assign_memory(frame *f)
{
    f->frame_luma = (unsigned char **)malloc(sizeof(unsigned char *) * height);
    for (int i = 0; i < height; i++)
    {
        f->frame_luma[i] = (unsigned char *)malloc(sizeof(unsigned char) * width);
    }
    f->frame_cb = (unsigned char **)malloc(sizeof(unsigned char *) * (height / 2));
    for (int i = 0; i < height / 2; i++)
    {
        f->frame_cb[i] = (unsigned char *)malloc(sizeof(unsigned char) * (width / 2));
    }
    f->frame_cr = (unsigned char **)malloc(sizeof(unsigned char *) * (height / 2));
    for (int i = 0; i < height / 2; i++)
    {
        f->frame_cr[i] = (unsigned char *)malloc(sizeof(unsigned char) * (width / 2));
    }
}

void read_yuv(FILE *ptr, frame *f)
{
    for (int i = 0; i < height; i++)
    {
        fread(f->frame_luma[i], 1, width, ptr);
    }
    for (int i = 0; i < height / 2; i++)
    {
        fread(f->frame_cb[i], 1, width / 2, ptr);
    }
    for (int i = 0; i < height / 2; i++)
    {
        fread(f->frame_cr[i], 1, width / 2, ptr);
    }
}

void write_yuv(frame *f, int i, char *mstr)
{
    FILE *fr;
    int2str(i, mstr);

    if (!(fr = fopen(mstr, "w")))
    {
        printf("Cannot open file %s Exiting . . . .\n", mstr);
        exit(-1);
    }
    print_header(fr, height, width);
    print_luma(fr, f, height, width);
}

void free_memory(frame *f)
{
    printf("Freeing memory 1. . . .\n");
    for (int i = 0; i < height; i++)
    {
        free(f->frame_luma[i]);
    }
    free(f->frame_luma);
    
    printf("Freeing memory 2. . . .\n");
    for (int i = 0; i < height / 2; i++)
    {
        free(f->frame_cb[i]);
    }
    free(f->frame_cb);
    
    printf("Freeing memory 3. . . .\n");
    for (int i = 0; i < height / 2; i++)
    {
        free(f->frame_cr[i]);
    }
    free(f->frame_cr);
    
    free(f);
}

int main(int argc, char *argv[])
{
    FILE *fptr, *fr;
    char filename[20];
    int write_file = 1;

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

    height = atoi(argv[2]);
    width = atoi(argv[3]);
    int num_frames = get_frame_num(argv[1], height, width);
    int cb_height = height / 2, cb_width = width / 2;


    printf("Frame height: %d\n", height);
    printf("Frame width: %d\n", width);
    printf("Number of frames: %d\n", num_frames);

    fptr = fopen(argv[1], "rb");
    printf("Reading frames ....\n");
    fseek(fptr, 0, SEEK_SET);

    for (int i = 0; i < num_frames; i++)
    {
        printf("Frame %d\n", i);
        frame *f = (frame *)malloc(sizeof(frame) * num_frames);
        f->height = height;
        f->width = width;
                
        assign_memory(f);
        // printf("Memory assigned\n");
        
        read_yuv(fptr, f);
        // printf("Frame read\n");
        
        write_yuv(f, i, mstr);
        // printf("Frame written\n");
        
        free_memory(f);
        // printf("Memory freed\n");
    }

    return 0;
}
