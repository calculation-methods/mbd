#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "read.h"

int SEIDEL = 0;

static inline double absolute(const double a) {
    return a > 0 ? a : -a;
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


int estimation_of_number_of_iteration(double eps, double a, double b) {
    return (int)(log(eps * (1 - a) / b) / log(a));
}


double error(Matrix* vector_prev, Matrix* vector_cur, double alpha_norm) {
    int i;
    double epsilon_k;
    Matrix* temp = create_matrix();
    resize_matrix(temp, vector_prev->height, 1);
    for (i = 0; i < vector_prev->height; i++)
        temp->data[i][0] = vector_prev->data[i][0] - vector_cur->data[i][0];
    if (alpha_norm < 1)
        epsilon_k = matrix_norm(temp) * alpha_norm / (1 - alpha_norm);
    else
        epsilon_k = matrix_norm(temp);
    remove_matrix(temp);
    free(temp);
    return epsilon_k;
}


Matrix* seidel_method(Matrix* alpha, Matrix* betta, double epsilon) {
    Matrix* current, * previous;
    double alpha_norm = matrix_norm(alpha), err = epsilon + 1;
    int i, j, step;
    if (alpha_norm >= 1) {
        fprintf(stderr, "Norm of ALPHA is %.4f\n", alpha_norm);
        return NULL;
    }
    current = create_matrix();
    resize_matrix(current, betta->height, 1);
    for (i = 0; i < betta->height; i++)
        current->data[i][0] = betta->data[i][0];
    previous = create_matrix();
    resize_matrix(previous, betta->height, 1);
    for (step = 1; err >= epsilon; step++) {
        for (i = 0; i < betta->height; i++)
            previous->data[i][0] = current->data[i][0];
        for (i = 0; i < betta->height; i++) {
            current->data[i][0] = betta->data[i][0];
            for (j = 0; j < alpha->width; j++)
                current->data[i][0] += alpha->data[i][j] * (j < i ? current->data[j][0] : previous->data[j][0]);
        }
        err = error(previous, current, alpha_norm);
    }
    printf("%d\n", step);
    remove_matrix(previous);
    free(previous);
    return current;
}


Matrix* jacobi_method(Matrix* alpha, Matrix* betta, double epsilon) {
    Matrix* current, * previous;
    double alpha_norm = matrix_norm(alpha), err = epsilon + 1;
    int i, step;
    if (alpha_norm > 1) {
        fprintf(stderr, "Norm of ALPHA is %.4f\n", alpha_norm);
        return NULL;
    }
    if (alpha_norm != 1)
        printf("Estimate of number of iteration: k + 1 less then %d\n",
            estimation_of_number_of_iteration(epsilon, alpha_norm, matrix_norm(betta)));
    current = create_matrix();
    resize_matrix(current, betta->height, 1);
    for (i = 0; i < betta->height; i++)
        current->data[i][0] = betta->data[i][0];
    previous = create_matrix();
    resize_matrix(previous, betta->height, 1);
    for (step = 1; err >= epsilon; step++) {
        for (i = 0; i < betta->height; i++)
            previous->data[i][0] = current->data[i][0];
        remove_matrix(current);
        free(current);
        current = multiplication(alpha, previous);
        for (i = 0; i < betta->height; i++)
            current->data[i][0] += betta->data[i][0];
        err = error(previous, current, alpha_norm);
    }
    printf("%d\n", step);
    remove_matrix(previous);
    free(previous);
    return current;
}

Matrix* simple_iteration_method(Matrix* matrix, Matrix* vector, double epsilon) {
    Matrix* alpha, * betta, * result;
    double norm;
    int i, j;
    if (!matrix || !vector || matrix->width != matrix->height || matrix->height != vector->height || vector->width != 1) {
        fprintf(stderr, "Matrix or vector have has the incorrect size\n");
        return NULL;
    }
    for (i = 0; i < matrix->height; i++)
        if (!matrix->data[i][i]) {
            fprintf(stderr, "Singular matrix\n");
            return NULL;
        }

    alpha = create_matrix();
    resize_matrix(alpha, matrix->height, matrix->width);
    betta = create_matrix();
    resize_matrix(betta, vector->height, 1);
    for (i = 0; i < matrix->height; i++) {
        for (j = 0; j < matrix->width; j++)
            if (i == j)
                alpha->data[i][j] = 0;
            else
                alpha->data[i][j] = -matrix->data[i][j] / matrix->data[i][i];
        betta->data[i][0] = vector->data[i][0] / matrix->data[i][i];
    }

    result = SEIDEL ? seidel_method(alpha, betta, epsilon) : jacobi_method(alpha, betta, epsilon);
    remove_matrix(alpha);
    remove_matrix(betta);
    free(alpha);
    free(betta);

    return result;
}

int main(void) {
    int i;
    double epsilon;
    Matrix* matrix = create_matrix(), * vector = create_matrix(), * result;
    FILE* fmatrix, * fvector;

    fmatrix = fopen("lab01-1matrix.txt", "r");
    fvector = fopen("lab01-1vector.txt", "r");

    printf("Enter the calculation accuracy ");
    scanf("%lf", &epsilon);
    if (epsilon <= 0) {
        fprintf(stderr, "Negative value of error\n");
        return 0;
    }
    if (!fmatrix || !fvector) {
        fprintf(stderr, "Invalid name of file\n");
        return 0;
    }

    scan_matrix(matrix, fmatrix);
    fclose(fmatrix);
    scan_matrix(vector, fvector);
    fclose(fvector);
    if (result = simple_iteration_method(matrix, vector, epsilon)) {
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
