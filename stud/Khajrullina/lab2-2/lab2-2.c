#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lab2.h"

int n = 0;
Matrix* values;
double constanta = 0.5;

void values_init(void) {
    values = create_matrix();
    resize_matrix(values, 2, 1);
    values->data[0][0] = values->data[1][0] = 1;
}

Matrix* Function(Matrix* vector) {
    Matrix* result;
    if (vector->height != 2 || vector->width != 1) {
        fprintf(stderr, "Invalid size of vector\n");
        return NULL;
    }
    result = create_matrix();
    resize_matrix(result, 2, 1);
    result->data[0][0] = 2 * pow(vector->data[0][0], 2) - vector->data[1][0] + pow(vector->data[1][0], 2) - 2;
    result->data[1][0] = vector->data[0][0] - pow(vector->data[1][0] + 2, 0.5) + 1;
    return result;
}

Matrix* Jacobi_matrix(Matrix* vector) {
    Matrix* result;
    if (vector->height != 2 || vector->width != 1) {
        fprintf(stderr, "Error\n");
        return NULL;
    }
    result = create_matrix();
    resize_matrix(result, 2, 2);
    result->data[0][0] = 4 * vector->data[0][0];      
    result->data[0][1] = 2 * vector->data[1][0] - 1;
    result->data[1][0] = 1;                         
    result->data[1][1] = -0.5/sqrt(vector->data[1][0] + 2);
    return result;
}

Matrix* simple_iteration_function(Matrix* vector) {
    Matrix* result;
    if (vector->height != 2 || vector->width != 1) {
        fprintf(stderr, "Error\n");
        return NULL;
    }
    result = create_matrix();
    resize_matrix(result, 2, 1);
    result->data[0][0] = pow(vector->data[1][0] + 2, 0.5) - 1;
    result->data[1][0] = pow(vector->data[1][0] + 2 - 2 * pow(vector->data[0][0],2),0.5);
    return result;
}

Matrix* simple_iteration_method(Matrix* (*Function)(Matrix*), double eps) {
    Matrix* current, * prev, * error_vector;
    int iter;
    double err;
    if (!values || values->width != 1)
        return NULL;
    prev = create_matrix();
    resize_matrix(prev, values->height, 1);
    for (iter = 0; iter < values->height; iter++)
        prev->data[iter][0] = values->data[iter][0];
    current = Function(prev);
    if (!current)
        return NULL;
    error_vector = substruction(current, prev);
    for (iter = 1; constanta / (1 - constanta) * (err = matrix_norm(error_vector)) > eps; iter++) {
        remove_matrix(prev);
        free(prev);
        prev = current;
        current = Function(prev);
        remove_matrix(error_vector);
        free(error_vector);
        error_vector = substruction(current, prev);
    }
    printf("Iterations: %d\n", iter);
    remove_matrix(error_vector);
    free(error_vector);
    remove_matrix(prev);
    free(prev);
    return current;
}

Matrix* newton_method(Matrix* (*Function)(Matrix*), Matrix* (*Jacobi_matrix)(Matrix*), double eps) {
    Matrix* current, * prev, * jacobi_matrix, * error_vector, * temp;
    int iter;
    double err;
    if (!values || values->width != 1)
        return NULL;
    prev = create_matrix();
    resize_matrix(prev, values->height, 1);

    for (iter = 0; iter < values->height; iter++)
        prev->data[iter][0] = values->data[iter][0];

    jacobi_matrix = Jacobi_matrix(prev);
    current = multiple_matrix(jacobi_matrix, prev);
    temp = Function(prev);
    error_vector = substruction(current, temp);

    remove_matrix(current);
    remove_matrix(temp);
    free(current);
    free(temp);

    current = gauss_method(jacobi_matrix, error_vector);

    remove_matrix(error_vector);
    free(error_vector);
    remove_matrix(jacobi_matrix);
    free(jacobi_matrix);

    if (!current)
        return NULL;
    error_vector = substruction(current, prev);

    for (iter = 1; (err = matrix_norm(error_vector)) > eps; iter++) {
        remove_matrix(prev);
        free(prev);
        prev = current;
        remove_matrix(error_vector);
        free(error_vector);
        jacobi_matrix = Jacobi_matrix(prev);
        current = multiple_matrix(jacobi_matrix, prev);
        temp = Function(prev);
        error_vector = substruction(current, temp);
        remove_matrix(current);
        remove_matrix(temp);
        free(current);
        free(temp);
        current = gauss_method(jacobi_matrix, error_vector);
        remove_matrix(error_vector);
        free(error_vector);
        error_vector = substruction(current, prev);
        remove_matrix(jacobi_matrix);
        free(jacobi_matrix);
    }
    printf("Iterations: %d\n", iter);

    remove_matrix(error_vector);
    free(error_vector);
    remove_matrix(prev);
    free(prev);
    return current;
}

int main(void) {
    int i;
    float eps;
    scanf("%f", &eps);
    Matrix* result;
    values_init();
    if (eps <= 0) {
        fprintf(stderr, "Error\n");
        return 0;
    }
    if (n) { 
        printf("Newton method:\n");
        result = newton_method(Function, Jacobi_matrix, eps);
        print_matrix(result, stdout);
        remove_matrix(result);
        free(result);
    }
    else {
        printf("Simple iteration method:\n");
        result = simple_iteration_method(simple_iteration_function, eps);
        print_matrix(result, stdout);
        remove_matrix(result);
        free(result);
    }

    return 0;
}