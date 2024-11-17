#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lab1.h"

static inline double absolute(const double a) {
    return a > 0 ? a : -a;
}
Matrix* create_rotation_matrix(unsigned int size, unsigned int l, unsigned int m, double phi) {
    Matrix* rotate;
    int i;
    if (size <= l || size <= m || l == m)
        return NULL;
    rotate = create_matrix();
    resize_matrix(rotate, size, size);
    for (i = 0; i < size; i++)
        rotate->data[i][i] = 1;
    rotate->data[l][l] = rotate->data[m][m] = cos(phi);
    rotate->data[l][m] = -(rotate->data[m][l] = sin(phi));
    return rotate;
}
double error(Matrix* matrix) {
    int i, j;
    double err = 0;
    if (!matrix)
        return 0;
    for (i = 0; i < matrix->height; i++)
        for (j = i + 1; j < matrix->width; j++)
            err += matrix->data[i][j] * matrix->data[i][j];
    return sqrt(err);
}
Matrix* rotation_method(Matrix* matrix, Matrix** lambda, double epsilon) {
    int i, j, i_max, j_max, step;
    double max = 0, err = epsilon + 1;
    Matrix* result, * rotate, * temp, * temp2;
    if (!matrix || matrix->height != matrix->width) {
        fprintf(stderr, "Not square matrix\n");
        return NULL;
    }
    *lambda = create_matrix();
    resize_matrix(*lambda, matrix->height, matrix->width);
    result = create_rotation_matrix((*lambda)->height, 1, 0, 0);
    for (i = 0; i < matrix->height; i++)
        for (j = 0; j < matrix->width; j++)
            (*lambda)->data[i][j] = matrix->data[i][j];

    for (step = 1; err >= epsilon; step++) {
        for (i = 0; i < (*lambda)->height; i++)
            for (j = i + 1; j < (*lambda)->width; j++)
                if (max < absolute((*lambda)->data[i][j])) {
                    max = absolute((*lambda)->data[i][j]);
                    i_max = i;
                    j_max = j;
                }
        rotate = create_rotation_matrix((*lambda)->height, i_max, j_max,
            0.5 * atan(2 * (*lambda)->data[i_max][j_max] /
                ((*lambda)->data[i_max][i_max] - (*lambda)->data[j_max][j_max])));
        max = 0;
        temp = multiple_matrix(result, rotate);
        for (i = 0; i < result->height; i++)
            for (j = 0; j < result->width; j++)
                result->data[i][j] = temp->data[i][j];
        remove_matrix(temp);
        free(temp);
        temp = transpose_matrix(rotate);
        temp2 = multiple_matrix(temp, *lambda);
        remove_matrix(temp);
        free(temp);
        remove_matrix(*lambda);
        free(*lambda);
        *lambda = multiple_matrix(temp2, rotate);
        remove_matrix(temp2);
        free(temp2);
        remove_matrix(rotate);
        free(rotate);
        err = error(*lambda);
    }
    for (i = 0; i < (*lambda)->height; i++)
        (*lambda)->data[i][0] = (*lambda)->data[i][i];
    resize_matrix(*lambda, (*lambda)->height, 1);
    return result;
}

int main(void) {
    int i;
    double epsilon;
    Matrix* matrix = create_matrix(), * vector, * result;
    FILE* fmatrix, * fepsilon;

    fmatrix = fopen("lab1-4m.txt", "r"); 
    fepsilon = fopen("lab1-4e.txt", "r");
    fscanf(fepsilon, "%lf", &epsilon);

    if (epsilon <= 0) {
        fprintf(stderr, "Negative value of error\n");
        return 0;
    }
    if (!fmatrix || !fepsilon) {
        fprintf(stderr, "Invalid name of file\n");
        return 0;
    }

    scan_matrix(matrix, fmatrix);
    fclose(fmatrix);
    fclose(fepsilon);

    if (result = rotation_method(matrix, &vector, epsilon)) {
        printf("Vectors: \n");
        print_matrix(result, stdout);
        printf("Values: \n");
        print_matrix(vector, stdout);
        remove_matrix(result);
        free(result);
        remove_matrix(vector);
        free(vector);
    }
    remove_matrix(matrix);
    free(matrix);
    return 0;
}
