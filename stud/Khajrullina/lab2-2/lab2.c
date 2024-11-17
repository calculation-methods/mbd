#include "lab2.h"
#include <stdlib.h>


static inline double absolute(const double a) {
    return a > 0 ? a : -a;
}
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
    int i, j;
    if (height > 0) {
        matrix->data = (double**)realloc(matrix->data, sizeof(double*) * height);
        for (i = matrix->height; i < height; i++) {
            matrix->data[i] = (double*)malloc(sizeof(double) * (width <= 0 ? matrix->width : width));
            for (j = 0; j < width; j++)
                matrix->data[i][j] = 0;
        }
    }
    if (width > 0)
        for (i = 0; i < matrix->height; i++) {
            matrix->data[i] = (double*)realloc(matrix->data[i], sizeof(double) * width);
            for (j = matrix->width; j < width; j++)
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
            fprintf(stream, "%.5Lf  ", matrix->data[i][j]);
        fprintf(stream, "\b\b]\n");
    }
    fprintf(stream, "size: %d x %d\n", matrix->height, matrix->width);
}
Matrix* multiple_matrix(Matrix* A, Matrix* B) {
    Matrix* result;
    int i, j, k;
    if (A->width != B->height)
        return NULL;
    result = create_matrix();
    resize_matrix(result, A->height, B->width);
    for (i = 0; i < result->height; i++)
        for (j = 0; j < result->width; j++)
            for (k = 0; k < A->width; k++)
                result->data[i][j] += A->data[i][k] * B->data[k][j];
    return result;
}
Matrix* substruction(Matrix* A, Matrix* B) {
    Matrix* result;
    int i, j;
    if (A->height != B->height || A->width != B->width)
        return NULL;
    result = create_matrix();
    resize_matrix(result, A->height, A->width);
    for (i = 0; i < result->height; i++)
        for (j = 0; j < result->width; j++)
            result->data[i][j] = A->data[i][j] - B->data[i][j];
    return result;
}
void exchange_str(Matrix* matrix, int i1, int i2) {
    double temp;
    int j;
    for (j = 0; j < matrix->width; j++) {
        temp = matrix->data[i1][j];
        matrix->data[i1][j] = matrix->data[i2][j];
        matrix->data[i2][j] = temp;
    }
}
void exchange_col(Matrix* matrix, int j1, int j2) {
    double temp;
    int i;
    for (i = 0; i < matrix->width; i++) {
        temp = matrix->data[i][j1];
        matrix->data[i][j1] = matrix->data[i][j2];
        matrix->data[i][j2] = temp;
    }
}
double matrix_norm(Matrix* matrix) {
    double sum = 0, norm = 0;
    int i, j;
    if (!matrix)
        return 0;
    for (i = 0; i < matrix->height; i++) {
        for (j = 0; j < matrix->width; j++)
            sum += absolute(matrix->data[i][j]);
        norm = sum > norm ? sum : norm;
        sum = 0;
    }
    return norm;
}
Matrix** LU_decomposition(Matrix* matrix) {
    int i, j, k, size = matrix->height, max;
    Matrix** LUP = (Matrix**)malloc(sizeof(Matrix*) * 3);
    if (matrix->height != matrix->width)
        return NULL;
    for (i = 0; i < 3; i++) {
        LUP[i] = create_matrix();
        resize_matrix(LUP[i], size, size);
    }
    for (i = 0; i < size; i++)
        LUP[2]->data[i][i] = LUP[0]->data[i][i] = 1;
    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            LUP[1]->data[i][j] = matrix->data[i][j];
    for (j = 0; j < size; j++) {
        max = j;
        for (i = j + 1; i < size; i++)
            if (absolute(LUP[1]->data[i][j]) > absolute(LUP[1]->data[max][j]))
                max = i;
        if (!LUP[1]->data[max][j])
            return NULL;
        exchange_str(LUP[1], j, max);
        exchange_str(LUP[2], j, max);
        exchange_str(LUP[0], j, max);
        exchange_col(LUP[0], j, max);
        for (i = j + 1; i < size; i++) {
            LUP[0]->data[i][j] = LUP[1]->data[i][j] / LUP[1]->data[j][j];
            for (k = j; k < size; k++)
                LUP[1]->data[i][k] -= LUP[0]->data[i][j] * LUP[1]->data[j][k];
        }
    }
    return LUP;
}
Matrix* LU_solve(Matrix** LUP, Matrix* vector) {
    Matrix* result;
    int i, j;
    if (!LUP)
        return NULL;
    result = multiple_matrix(LUP[2], vector);
    for (i = 0; i < result->height; i++)
        for (j = 0; j < i; j++)
            result->data[i][0] -= result->data[j][0] * LUP[0]->data[i][j];
    for (i = result->height - 1; i >= 0; i--) {
        for (j = result->height - 1; j > i; j--)
            result->data[i][0] -= result->data[j][0] * LUP[1]->data[i][j];
        result->data[i][0] /= LUP[1]->data[i][i];
    }
    return result;
}
Matrix* gauss_method(Matrix* matrix, Matrix* vector) {
    Matrix** LUP, * result;
    int i;
    if (matrix->height != vector->height || vector->width != 1) {
        return NULL;
    }
    LUP = LU_decomposition(matrix);
    result = LU_solve(LUP, vector);
    for (i = 0; i < 3; i++) {
        remove_matrix(LUP[i]);
        free(LUP[i]);
    }
    free(LUP);
    return result;
}
