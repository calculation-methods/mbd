#include "read.h"
#include <stdlib.h>

Matrix* create_matrix(void) {
    Matrix* new_matrix = (Matrix*)malloc(sizeof(Matrix));
    new_matrix->width = new_matrix->height = 0;
    new_matrix->data = NULL;
    return new_matrix;
}
void remove_matrix(Matrix* matrix) {
    int i;
    if (!matrix)
        return;
    for (i = 0; i < matrix->height; i++)
        free(matrix->data[i]);
    free(matrix->data);
    matrix->data = NULL;
    matrix->height = matrix->width = 0;
}
void resize_matrix(Matrix* matrix, const int height, const int width) {
    if (height > 0) {
        matrix->data = (double**)realloc(matrix->data, sizeof(double*) * height);
        for (int i = matrix->height; i < height; i++) {
            matrix->data[i] = (double*)malloc(sizeof(double) * (width <= 0 ? matrix->width : width));
            for (int j = 0; j < width; j++)
                matrix->data[i][j] = 0;
        }
    }
    if (width > 0)
        for (int i = 0; i < matrix->height; i++) {
            matrix->data[i] = (double*)realloc(matrix->data[i], sizeof(double) * width);
            for (int j = matrix->width; j < width; j++)
                matrix->data[i][j] = 0;
        }
    if (width > 0)
        matrix->width = width;
    if (height > 0)
        matrix->height = height;
}
void print_matrix(Matrix* matrix, FILE* stream) {
    int i, j;
    if (!matrix->data)
        return;
    for (i = 0; i < matrix->height; i++) {
        fputc('[', stream);
        for (j = 0; j < matrix->width; j++)
            fprintf(stream, "%.5f  ", matrix->data[i][j]);
        fprintf(stream, "\b\b]\n");
    }
    fprintf(stream, "size: %d x %d\n", matrix->height, matrix->width);
}
void scan_matrix(Matrix* matrix, FILE* stream) {
    int i, j, c = 0;
    float a;
    for (i = 0; c != EOF; i++) {
        resize_matrix(matrix, i + 1, -1);
        c = 0;
        for (j = 0; c != '\n'; j++) {
            if (!i)
                resize_matrix(matrix, -1, j + 1);
            fscanf_s(stream, "%f", &a);
            matrix->data[i][j] = a;
            c = getc(stream);
            if (c == EOF) { 
                resize_matrix(matrix, i, -1);
                return;
            }
        }
    }
}

static inline float absolute(float a) {
    return a > 0 ? a : -a;
}

Matrix* tridiagonal_matrix_algorithm(Matrix* matrix, Matrix* vector) {
    Matrix* PQ, * result;
    int i;
    if (!matrix || !vector || matrix->width != 3 || matrix->height != vector->height || vector->width != 1)
        return NULL;
    PQ = create_matrix();
    resize_matrix(PQ, matrix->height - 1, 2);
    result = create_matrix();
    resize_matrix(result, matrix->height, 1);
    PQ->data[0][0] = -matrix->data[0][2] / matrix->data[0][1];
    PQ->data[0][1] = vector->data[0][0] / matrix->data[0][1];
    if (absolute(matrix->data[0][1]) < absolute(matrix->data[0][2]) ||
        absolute(matrix->data[matrix->height - 1][1]) < absolute(matrix->data[matrix->height - 1][2]))
        return NULL;
    for (i = 1; i < PQ->height; i++) {
        if (absolute(matrix->data[i][1]) < absolute(matrix->data[i][0]) + absolute(matrix->data[i][2]))
            return NULL;
        double temp = matrix->data[i][0] * PQ->data[i - 1][0] + matrix->data[i][1];
        PQ->data[i][0] = -matrix->data[i][2] / temp;
        PQ->data[i][1] = (vector->data[i][0] - matrix->data[i][0] * PQ->data[i - 1][1]) / temp;
    }
    i = result->height - 1;
    result->data[i][0] = (vector->data[i][0] - matrix->data[i][0] * PQ->data[i - 1][1]) /
        (matrix->data[i][0] * PQ->data[i - 1][0] + matrix->data[i][1]);
    for (i = result->height - 2; i >= 0; i--)
        result->data[i][0] = PQ->data[i][0] * result->data[i + 1][0] + PQ->data[i][1];
    remove_matrix(PQ);
    free(PQ);
    return result;
}
Matrix* (* const TDMA)(Matrix*, Matrix*) = tridiagonal_matrix_algorithm;