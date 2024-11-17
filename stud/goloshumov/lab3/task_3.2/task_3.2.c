#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "read.h"

int DEBUG = 0;

Matrix* create_cubic_spline(Matrix* matrix) {
    Matrix* spline, * tridiagonal_matrix, * vector, * solve;
    int i;
    if (!matrix || matrix->width != 2) {
        fprintf(stderr, "Invalid size of matrix\n");
    }
    spline = create_matrix();
    resize_matrix(spline, matrix->height, 5);

    tridiagonal_matrix = create_matrix(); 
    resize_matrix(tridiagonal_matrix, matrix->height - 2, 3);

    vector = create_matrix();
    resize_matrix(vector, matrix->height - 2, 1);

    for (i = 0; i < vector->height; i++) {
        tridiagonal_matrix->data[i][0] = (i == vector->height - 1 ? 0 : matrix->data[i + 1][0] - matrix->data[i][0]);
        tridiagonal_matrix->data[i][1] = 2 * (matrix->data[i + 2][0] - matrix->data[i][0]);
        tridiagonal_matrix->data[i][2] = (i == 0 ? 0 : matrix->data[i + 2][0] - matrix->data[i + 1][0]);
        vector->data[i][0] = 3 * ((matrix->data[i + 2][1] - matrix->data[i + 1][1]) / (matrix->data[i + 2][0] - matrix->data[i + 1][0]) -
            (matrix->data[i + 1][1] - matrix->data[i][1]) / (matrix->data[i + 1][0] - matrix->data[i][0]));
    }
    if (DEBUG) {
        fprintf(stderr, "Tridiagonal matrix:\n");
        print_matrix(tridiagonal_matrix, stderr);
        fprintf(stderr, "Vector of right part:\n");
        print_matrix(vector, stderr);
    }
    solve = TDMA(tridiagonal_matrix, vector);
    if (!solve) {
        fprintf(stderr, "Singular matrix\n");
        return NULL;
    }
    remove_matrix(tridiagonal_matrix);
    remove_matrix(vector);
    free(tridiagonal_matrix);
    free(vector);

    spline->data[0][3] = 0;
    for (i = 0; i < spline->height - 1; i++) {
        spline->data[i][0] = matrix->data[i][0];
        spline->data[i][1] = matrix->data[i][1];
        if (i < spline->height - 2)
            spline->data[i + 1][3] = solve->data[i][0];
        spline->data[i][2] = (matrix->data[i + 1][1] - matrix->data[i][1]) / (matrix->data[i + 1][0] - matrix->data[i][0]) -
            1.0 / 3.0 * (matrix->data[i + 1][0] - matrix->data[i][0]) * (2 * spline->data[i][3] + (i < spline->height - 2 ? spline->data[i + 1][3] : 0));
        spline->data[i][4] = ((i < spline->height - 2 ? spline->data[i + 1][3] : 0) - spline->data[i][3]) /
            (3 * (matrix->data[i + 1][0] - matrix->data[i][0]));
    }

    spline->data[spline->height - 1][0] = matrix->data[spline->height - 1][0];

    if (DEBUG) {
        fprintf(stderr, "Spline matrix\n");
        print_matrix(spline, stderr);
    }
    remove_matrix(solve);
    free(solve);
    return spline;
}
double cubic_spline(Matrix* spline, double x) {
    int i;
    for (i = 0; i < spline->height - 1; i++)
        if (spline->data[i][0] <= x && x <= spline->data[i + 1][0])
            return spline->data[i][1] + spline->data[i][2] * (x - spline->data[i][0]) +
            spline->data[i][3] * (x - spline->data[i][0]) * (x - spline->data[i][0]) +
            spline->data[i][4] * (x - spline->data[i][0]) * (x - spline->data[i][0]) *
            (x - spline->data[i][0]);
    if (DEBUG)
        fprintf(stderr, "Incorrect value of argument\n");
    return 0;
}
void print_cubic_spline(Matrix* spline, FILE* stream) {
    int i;
    for (i = 0; i < spline->height - 1; i++) {
        fprintf(stream, "[%.4f; %.4f]\n", spline->data[i][0], spline->data[i + 1][0]);
        if (spline->data[i][1] >= 0)
            fprintf(stream, " ");
        fprintf(stream, "%.4f", spline->data[i][1]);
        if (spline->data[i][2] >= 0)
            fprintf(stream, "+");
        fprintf(stream, "%.4f(x", spline->data[i][2]);
        if (spline->data[i][0] < 0)
            fprintf(stream, "+");
        fprintf(stream, "%.4f)", -spline->data[i][0]);
        if (spline->data[i][3] >= 0)
            fprintf(stream, "+");
        fprintf(stream, "%.4f*(x", spline->data[i][3]);
        if (spline->data[i][0] < 0)
            fprintf(stream, "+");
        fprintf(stream, "%.4f)^2", -spline->data[i][0]);
        if (spline->data[i][4] >= 0)
            fprintf(stream, "+");
        fprintf(stream, "%.4f*(x", spline->data[i][4]);
        if (spline->data[i][0] < 0)
            fprintf(stream, "+");
        fprintf(stream, "%.4f)^3\n\n", -spline->data[i][0]);
    }
}

int main(void) {
    float x = 2.66666667;
    Matrix* result, * matrix;
    FILE* fmatrix;

    fmatrix = fopen("task_3.2matrix.txt", "r");
    if (!fmatrix) {
        fprintf(stderr, "Incorrect name of file\n");
        return 0;
    }

    matrix = create_matrix();
    scan_matrix(matrix, fmatrix);
    fclose(fmatrix);

    if (result = create_cubic_spline(matrix)) {
        printf("Cubic spline:\n");
        print_cubic_spline(result, stdout);
        printf("Cubic spline value:\nS(%.4f) = %.4f\n", x, cubic_spline(result, x));
        remove_matrix(result);
        free(result);
    }
    remove_matrix(matrix);
    free(matrix);
    return 0;
}