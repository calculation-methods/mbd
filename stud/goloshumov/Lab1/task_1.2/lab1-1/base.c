#include "read.h"
#include <stdlib.h>


Matrix* create_matrix(void) {
    Matrix* new_matrix = (Matrix*)malloc(sizeof(Matrix));
    new_matrix->width = new_matrix->height = 0;
    new_matrix->data = NULL;
    return new_matrix;
}


void remove_matrix(Matrix* matrix) {
    if (!matrix)
        return;
    for (int i = 0; i < matrix->height; i++)
        free(matrix->data[i]);
    free(matrix->data);
    matrix->data = NULL;
    matrix->height = matrix->width = 0;
}


void resize_matrix(Matrix* matrix, const int height, const int width) {
    if (height > 0) {
        matrix->data = (double**) realloc (matrix->data, sizeof(double*) * height);
        for (int i = matrix->height; i < height; i++) {
            matrix->data[i] = (double*) malloc(sizeof(double) * (width <= 0 ? matrix->width : width));
            for (int j = 0; j < width; j++)
                matrix->data[i][j] = 0;
        }
    }
    if (width > 0)
        for (int i = 0; i < matrix->height; i++) {
            matrix->data[i] = (double*) realloc (matrix->data[i], sizeof(double) * width);
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
    int c = 0;
    float a;
    for (int i = 0; c != EOF; i++) {
        resize_matrix(matrix, i + 1, -1);
        c = 0;
        for (int j = 0; c != '\n'; j++) {
            if (!i)
                resize_matrix(matrix, -1, j + 1);
            fscanf(stream, "%f", &a);
            matrix->data[i][j] = a;
            c = getc(stream);
            if (c == EOF) {
                resize_matrix(matrix, i, -1);
                return;
            }
        }
    }
}


Matrix* multiplication(Matrix* A, Matrix* B) {
    Matrix* result = create_matrix();
    int i, j, k;
    if (A->width != B->height)
        return NULL;
    resize_matrix(result, A->height, B->width);
    for (i = 0; i < result->height; i++)
        for (j = 0; j < result->width; j++)
            for (k = 0; k < A->width; k++)
                result->data[i][j] += A->data[i][k] * B->data[k][j];
    return result;
}


void matrix_exchange_strings(Matrix* matrix, int i1, int i2) {
    double temp;
    for (int j = 0; j < matrix->width; j++) {
        temp = matrix->data[i1][j];
        matrix->data[i1][j] = matrix->data[i2][j];
        matrix->data[i2][j] = temp;
    }
}


void matrix_exchange_columns(Matrix* matrix, int j1, int j2) {
    double temp;
    for (int i = 0; i < matrix->width; i++) {
        temp = matrix->data[i][j1];
        matrix->data[i][j1] = matrix->data[i][j2];
        matrix->data[i][j2] = temp;
    }
}


Matrix* matrix_transposition(Matrix* matrix) {
    Matrix* result;
    int i, j;
    if (!matrix)
        return NULL;
    result = create_matrix();
    resize_matrix(result, matrix->width, matrix->height);
    for (i = 0; i < result->height; i++)
        for (j = 0; j < result->width; j++)
            result->data[i][j] = matrix->data[j][i];
    return result;
}