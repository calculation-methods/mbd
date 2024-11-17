#ifndef _LAB1_
#define _LAB1_

#include <stdio.h>

typedef struct Matrix {
    double** data;
    unsigned int width;
    unsigned int height;
} Matrix;

Matrix* create_matrix(void);
void remove_matrix(Matrix*);
void resize_matrix(Matrix*, const int, const int);
void print_matrix(Matrix*, FILE*);
void scan_matrix(Matrix*, FILE*);
Matrix* multiple_matrix(Matrix*, Matrix*);
void exchange_str(Matrix*, int, int);
void exchange_col(Matrix*, int, int);
Matrix* transpose_matrix(Matrix*);

#endif