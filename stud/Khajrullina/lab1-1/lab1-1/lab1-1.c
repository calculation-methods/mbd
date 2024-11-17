#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lab1.h"

static inline double absolute(const double a) {
    return a > 0 ? a : -a;
}

Matrix** LU_decomposition(Matrix* matrix) {
    int size = matrix->height, max;
    Matrix** LUP = (Matrix**)malloc(sizeof(Matrix*) * 3);

    for (int i = 0; i < 3; i++) {
        LUP[i] = create_matrix();
        resize_matrix(LUP[i], size, size);
    }

    for (int i = 0; i < size; i++)
        LUP[2]->data[i][i] = LUP[0]->data[i][i] = 1;

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            LUP[1]->data[i][j] = matrix->data[i][j];

    for (int j = 0; j < size; j++) {
        max = j;
        for (int i = j + 1; i < size; i++)
            if (absolute(LUP[1]->data[i][j]) > absolute(LUP[1]->data[max][j]))
                max = i;
        exchange_str(LUP[1], j, max);
        exchange_str(LUP[2], j, max);
        exchange_str(LUP[0], j, max);
        exchange_col(LUP[0], j, max);

        for (int i = j + 1; i < size; i++) {
            LUP[0]->data[i][j] = LUP[1]->data[i][j] / LUP[1]->data[j][j];
            for (int k = j; k < size; k++)
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

Matrix* calculate_decisions(Matrix* matrix, Matrix* vector) {
    Matrix** LUP, * result;
    int i;
    LUP = LU_decomposition(matrix);
    result = LU_solve(LUP, vector);
    for (i = 0; i < 3; i++) {
        remove_matrix(LUP[i]);
        free(LUP[i]);
    }
    free(LUP);
    return result;
}

double get_determinant(Matrix* matrix) {
    Matrix** LUP = LU_decomposition(matrix);
    int i, j = 0;
    double det;
    if (!LUP)
        return 0;
    for (i = 0; i < LUP[2]->height; i++)
        if (!LUP[2]->data[i][i])
            j++;
    det = (j % 2 || !j) ? 1 : -1;
    for (i = 0; i < LUP[1]->height; i++)
        det *= LUP[1]->data[i][i];
    return det;
}

Matrix* matrix_inversion(Matrix* matrix) {
    Matrix** LUP = LU_decomposition(matrix), * inverse, * right_part, * temp;
    int i, j;
    if (!LUP)
        return NULL;
    inverse = create_matrix();
    resize_matrix(inverse, matrix->height, matrix->width);
    right_part = create_matrix();
    resize_matrix(right_part, matrix->height, 1);
    right_part->data[0][0] = 1;
    for (j = 0; j < inverse->width; j++) {
        temp = LU_solve(LUP, right_part);
        for (i = 0; i < inverse->height; i++)
            inverse->data[i][j] = temp->data[i][0];
        if (j != inverse->width - 1)
            exchange_str(right_part, j + 1, j);
        remove_matrix(temp);
        free(temp);
    }
    {
        for (i = 0; i < 3; i++) {
            remove_matrix(LUP[i]);
            free(LUP[i]);
        }
        free(LUP);
        remove_matrix(right_part);
        free(right_part);
    }
    return inverse;
}

int main(void) {
    int i;
    Matrix* matrix = create_matrix(), * vector = create_matrix(), * result;
    FILE* fmatrix, * fvector;

    fmatrix = fopen("lab1-1m.txt", "r");
    fvector = fopen("lab1-1v.txt", "r");

    if (fmatrix == NULL || fvector == NULL) {
        fprintf(stderr, "Invalid name of file\n");
        return 0;
    }

    scan_matrix(matrix, fmatrix);
    fclose(fmatrix);
    scan_matrix(vector, fvector);
    fclose(fvector);

    if (result = calculate_decisions(matrix, vector)) {
        printf("Solve of equation Ax=b\n");
        print_matrix(result, stdout);
        remove_matrix(result);
        free(result);
    }
    if (matrix->height == matrix->width) {
        printf("det(A) = %.4f\n", get_determinant(matrix));
    }
    if (result = matrix_inversion(matrix)) {
        printf("A(-1) = \n");
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