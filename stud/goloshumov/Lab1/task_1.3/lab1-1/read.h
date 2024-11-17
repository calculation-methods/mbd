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
Matrix* multiplication(Matrix*, Matrix*);
void matrix_exchange_strings(Matrix*, int, int);
void matrix_exchange_columns(Matrix*, int, int);
Matrix* matrix_transposition(Matrix*);

#endif