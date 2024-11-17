#ifndef _LAB3_
#define _LAB3_

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
Matrix* (* const TDMA)(Matrix*, Matrix*);

#endif /* _LAB3_ */
