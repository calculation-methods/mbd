#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "read.h"

int FIRST = 0;
int SECOND = 1;

double square_error(Matrix* least_squares, Matrix* vector) {
    double result = 0;
    int i, j;
    if (!least_squares || !vector || vector->width != 2 || least_squares->width != 1) {
        fprintf(stderr, "Invalid size of matrix");
        return 0;
    }
    for (i = 0; i < vector->height; i++) {
        double temp = least_squares->data[0][0] - vector->data[i][1];
        for (j = 1; j < least_squares->height; j++)
            temp += least_squares->data[j][0] * pow(vector->data[i][0], j);
        result += temp * temp;
    }
    return result;
}

Matrix* least_square_polynomial_degree_2(Matrix* vector) {
    Matrix* result;
    int i;
    double a11 = vector->height, a12 = 0, a22 = 0, a23 = 0, a33 = 0, b1 = 0, b2 = 0, b3 = 0;
    double c11, c12, c22, d1, d2;
    if (!vector || vector->width != 2) {
        fprintf(stderr, "Invalid size of matrix");
        return 0;
    }
    result = create_matrix();
    resize_matrix(result, 3, 1);
    for (i = 0; i < vector->height; i++) {
        a12 += vector->data[i][0];
        a22 += vector->data[i][0] * vector->data[i][0];
        a23 += vector->data[i][0] * vector->data[i][0] * vector->data[i][0];
        a33 += vector->data[i][0] * vector->data[i][0] * vector->data[i][0] * vector->data[i][0];
        b1 += vector->data[i][1];
        b2 += vector->data[i][1] * vector->data[i][0];
        b3 += vector->data[i][1] * vector->data[i][0] * vector->data[i][0];
    }
    c11 = a22 * a11 - a12 * a12;
    c12 = a23 * a11 - a12 * a22;
    c22 = a11 * a33 - a22 * a22;
    d1 = b2 * a11 - b1 * a12;
    d2 = b3 * a11 - b1 * a22;
    result->data[1][0] = (d1 * c22 - d2 * c12) / (c11 * c22 - c12 * c12);
    result->data[2][0] = (d2 * c11 - d1 * c12) / (c11 * c22 - c12 * c12);
    result->data[0][0] = (b1 - result->data[1][0] * a12 - result->data[2][0] * a22) / a11;
    return result;
}

Matrix* least_square_polynomial_degree_1(Matrix* vector) {
    Matrix* result;
    int i;
    double a11 = vector->height, a12 = 0, a22 = 0, b1 = 0, b2 = 0;
    if (!vector || vector->width != 2) {
        fprintf(stderr, "Invalid size of matrix");
        return 0;
    }
    result = create_matrix();
    resize_matrix(result, 2, 1);
    for (i = 0; i < vector->height; i++) {
        a12 += vector->data[i][0];
        a22 += vector->data[i][0] * vector->data[i][0];
        b1 += vector->data[i][1];
        b2 += vector->data[i][0] * vector->data[i][1];
    }
    result->data[0][0] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a12);
    result->data[1][0] = (b2 * a11 - b1 * a12) / (a11 * a22 - a12 * a12);
    return result;
}

void print_least_square(Matrix* least_square, FILE* stream) {
    int i;
    if (!least_square || least_square->width != 1) {
        fprintf(stderr, "Invalid size of matrix");
        return;
    }
    if (least_square->data[0][0] >= 0)
        fputc(' ', stream);
    fprintf(stream, "%.4f", least_square->data[0][0]);
    if (least_square->data[1][0] >= 0)
        fputc('+', stream);
    fprintf(stream, "%.4f*x", least_square->data[1][0]);
    for (i = 2; i < least_square->height; i++) {
        if (least_square->data[i][0] >= 0)
            fputc('+', stream);
        fprintf(stream, "%.4f*x^%d", least_square->data[i][0], i);
    }
    fputc('\n', stream);
}

int main(void) {
    int i;
    Matrix* result, * vector;
    FILE* fvector;

    fvector = fopen("task_3.3matrix.txt", "r");
    if (!fvector) {
        fprintf(stderr, "Incorrect name of file\n");
        return 0;
    }
    vector = create_matrix();
    scan_matrix(vector, fvector);
    fclose(fvector);

    if (FIRST && (result = least_square_polynomial_degree_1(vector))) {
        printf("Least square polynomial degree 1:\n");
        print_least_square(result, stdout);
        printf("Square error:\nE = %.4f\n", square_error(result, vector));
        remove_matrix(result);
        free(result);
    }
    else if (SECOND && (result = least_square_polynomial_degree_2(vector))) {
        printf("Least square polynomial degree 2:\n");
        print_least_square(result, stdout);
        printf("Square error:\nE = %.4f\n", square_error(result, vector));
        remove_matrix(result);
        free(result);
    }
    remove_matrix(vector);
    free(vector);
    return 0;
}
