#include <stdio.h>
#include <stdlib.h>
#include "read.h"

static inline double absolute(double a) {
    return a > 0 ? a : -a;
}


Matrix* tridiagonal_matrix_algorithm(Matrix* matrix, Matrix* vector) {
    Matrix* PQ, * result;
    int i;
    if (!matrix || !vector || matrix->width != 3 || matrix->height != vector->height || vector->width != 1) {
        fprintf(stderr, "Matrix or vector have has incorrect size\n");
        return NULL;
    }
    PQ = create_matrix();
    resize_matrix(PQ, matrix->height - 1, 2);
    result = create_matrix();
    resize_matrix(result, matrix->height, 1);

    PQ->data[0][0] = -matrix->data[0][2] / matrix->data[0][1];
    PQ->data[0][1] = vector->data[0][0] / matrix->data[0][1];
    if (absolute(matrix->data[0][1]) < absolute(matrix->data[0][2]) ||
        absolute(matrix->data[matrix->height - 1][1]) < absolute(matrix->data[matrix->height - 1][2])) {
        fprintf(stderr, "Singular matrix\n");
        return NULL;
    }

    for (i = 1; i < PQ->height; i++) {
        if (absolute(matrix->data[i][1]) < absolute(matrix->data[i][0]) + absolute(matrix->data[i][2])) {
            fprintf(stderr, "Singular matrix\n");
            return NULL;
        }
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

int main(void) {
    int i;
    Matrix* matrix = create_matrix(), * vector = create_matrix(), * result;
    FILE* fmatrix, * fvector;

    fmatrix = fopen("lab01-1matrix.txt", "r");
    fvector = fopen("lab01-1vector.txt", "r");

    if (!fmatrix || !fvector) {
        fprintf(stderr, "Invalid name of file\n");
        return 0;
    }

    scan_matrix(matrix, fmatrix);
    fclose(fmatrix);
    scan_matrix(vector, fvector);
    fclose(fvector);
    if (result = TDMA(matrix, vector)) {
        print_matrix(result, stdout);
        remove_matrix(result);
        free(result);
    }
    remove_matrix(vector);
    remove_matrix(matrix);
    free(vector);
    free(matrix);

    return 0;
}
