#ifndef _LAB2_
#define _LAB2_

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
Matrix* multiple_matrix(Matrix*, Matrix*);
Matrix* substruction(Matrix*, Matrix*);
double matrix_norm(Matrix*);
Matrix* gauss_method(Matrix*, Matrix*);

#endif
