#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read.h"

int DEBUG = 1;

double first_derivative(Matrix* vector, double x) {
    int i;
    double b, c, b_1;
    if (!vector || vector->width != 2) {
        fprintf(stderr, "Invalid size of matrix\n");
        return 0;
    }
    for (i = 1; i < vector->height; i++)
        if (vector->data[i - 1][0] <= x && x <= vector->data[i][0])
            break;
    if (i > vector->height - 2) {
        fprintf(stderr, "Too few information about the function\n");
        return 0;
    }

    b = (vector->data[i][1] - vector->data[i - 1][1]) / (vector->data[i][0] - vector->data[i - 1][0]);
    c = ((vector->data[i + 1][1] - vector->data[i][1]) / (vector->data[i + 1][0] - vector->data[i][0]) - b) /
        (vector->data[i + 1][0] - vector->data[i - 1][0]);
    if (DEBUG) {
        fprintf(stderr, "Approximate polynomial degree 2:\n%.4f", vector->data[i - 1][1]);
        if (b >= 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f(x", b);
        if (vector->data[i - 1][0] < 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f)", -vector->data[i - 1][0]);
        if (c >= 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f(x", c);
        if (vector->data[i - 1][0] < 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f)(x", -vector->data[i - 1][0]);
        if (vector->data[i][0] < 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f),\t[%.2f; %.2f]\n",
            -vector->data[i][0], vector->data[i - 1][0], vector->data[i][0]);
    }
    b_1 = (vector->data[i + 1][1] - vector->data[i][1]) / (vector->data[i + 1][0] - vector->data[i][0]);
    printf("f'(1.0000) = %f\n", b_1);
    printf("f'(1.0000) = %f\n", c + b);
    return b;
}
double second_derivative(Matrix* vector, double x) {
    int i;
    double b, c;
    if (!vector || vector->width != 2) {
        fprintf(stderr, "Invalid size of matrix\n");
        return 0;
    }
    for (i = 1; i < vector->height; i++)
        if (vector->data[i - 1][0] <= x && x <= vector->data[i][0])
            break;
    if (i > vector->height - 2) {
        fprintf(stderr, "Too few information about the function\n");
        return 0;
    }
    b = (vector->data[i][1] - vector->data[i - 1][1]) / (vector->data[i][0] - vector->data[i - 1][0]);
    c = ((vector->data[i + 1][1] - vector->data[i][1]) / (vector->data[i + 1][0] - vector->data[i][0]) - b) /
        (vector->data[i + 1][0] - vector->data[i - 1][0]);
    if (DEBUG) {
        fprintf(stderr, "Approximate polynomial degree 2:\n%.4f", vector->data[i - 1][1]);
        if (b >= 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f(x", b);
        if (vector->data[i - 1][0] < 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f)", -vector->data[i - 1][0]);
        if (c >= 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f(x", c);
        if (vector->data[i - 1][0] < 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f)(x", -vector->data[i - 1][0]);
        if (vector->data[i][0] < 0)
            fputc('+', stderr);
        fprintf(stderr, "%.4f),\t[%.2f; %.2f]\n",
            -vector->data[i][0], vector->data[i - 1][0], vector->data[i][0]);
    }
    return 2 * c;
}

int main(void) {
    int i;
    Matrix* vector;
    float x = 0.2;
    FILE* fvector;

    fvector = fopen("task_3.4matrix.txt", "r");
    if (!fvector) {
        fprintf(stderr, "Incorrect name of file\n");
        return 0;
    }
    vector = create_matrix();
    scan_matrix(vector, fvector);
    fclose(fvector);

    printf("f'(%.4f) = %.4f\n", x, first_derivative(vector, x));
    printf("f''(%.4f) = %.4f\n", x, second_derivative(vector, x));
    remove_matrix(vector);
    free(vector);
    return 0;
}