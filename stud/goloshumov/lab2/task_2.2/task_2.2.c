#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "read.h"

int NEWTON = 1;

Matrix* BEGIN_VALUES;
double SIMPLE_ITERATION_CONSTANT = 0.5;

void begin_values_initialization(void) {
    BEGIN_VALUES = create_matrix();
    resize_matrix(BEGIN_VALUES, 2, 1);
    BEGIN_VALUES->data[0][0] = BEGIN_VALUES->data[1][0] = 1;
}

void begin_values_removing(void) {
    remove_matrix(BEGIN_VALUES);
    free(BEGIN_VALUES);
}

Matrix* Function(Matrix* vector) {
    Matrix* result;
    if (vector->height != 2 || vector->width != 1) {
        fprintf(stderr, "Invalid size of vector\n");
        return NULL;
    }
    result = create_matrix();
    resize_matrix(result, 2, 1);
    result->data[0][0] = vector->data[0][0] - cos(vector->data[1][0]) - 1;
    result->data[1][0] = vector->data[1][0] - logl(vector->data[0][0] + 1) - 1;
    return result;
}

Matrix* Jacobi_matrix(Matrix* vector) {
    Matrix* result;
    if (vector->height != 2 || vector->width != 1) {
        fprintf(stderr, "Invalid size of vector\n");
        return NULL;
    }
    result = create_matrix();
    resize_matrix(result, 2, 2);
    result->data[0][0] = 1;                                      result->data[0][1] = sin(vector->data[1][0]);
    result->data[1][0] = -1 / (vector->data[0][0] + 1);          result->data[1][1] = 1;
    return result;
}

Matrix* simple_iteration_function(Matrix* vector) {

    Matrix* result;
    if (vector->height != 2 || vector->width != 1) {
        fprintf(stderr, "Invalid size of vector\n");
        return NULL;
    }
    result = create_matrix();
    resize_matrix(result, 2, 1);
    result->data[0][0] = cos(vector->data[1][0]) + 1;
    result->data[1][0] = logl(vector->data[0][0] + 1.0) + 1;
    return result;
}

Matrix* simple_iteration_method(Matrix* (*Function)(Matrix*), double epsilon) {
    Matrix* current, * previous, * error_vector;
    int step;
    double err;
    if (!BEGIN_VALUES || BEGIN_VALUES->width != 1)
        return NULL;
    previous = create_matrix();
    resize_matrix(previous, BEGIN_VALUES->height, 1);
    for (step = 0; step < BEGIN_VALUES->height; step++)
        previous->data[step][0] = BEGIN_VALUES->data[step][0];
    current = Function(previous);
    if (!current)
        return NULL;
    error_vector = substruction(current, previous);
    for (step = 1; SIMPLE_ITERATION_CONSTANT / (1 - SIMPLE_ITERATION_CONSTANT) *
        (err = matrix_norm(error_vector)) > epsilon; step++) {

        remove_matrix(previous);
        free(previous);
        previous = current;
        current = Function(previous);
        remove_matrix(error_vector);
        free(error_vector);
        error_vector = substruction(current, previous);
    }
    printf("steps: %d\n", step);

    remove_matrix(error_vector);
    free(error_vector);
    remove_matrix(previous);
    free(previous);
    return current;
}

Matrix* newton_method(Matrix* (*Function)(Matrix*), Matrix* (*Jacobi_matrix)(Matrix*), double epsilon) {
    Matrix* current, * previous, * jacobi_matrix, * error_vector, * temp;
    int step;
    double err;
    if (!BEGIN_VALUES || BEGIN_VALUES->width != 1)
        return NULL;
    previous = create_matrix();
    resize_matrix(previous, BEGIN_VALUES->height, 1);

    for (step = 0; step < BEGIN_VALUES->height; step++)
        previous->data[step][0] = BEGIN_VALUES->data[step][0];

    jacobi_matrix = Jacobi_matrix(previous);
    current = multiplication(jacobi_matrix, previous);
    temp = Function(previous);
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
    error_vector = substruction(current, previous);

    for (step = 1; (err = matrix_norm(error_vector)) > epsilon; step++) {

        remove_matrix(previous);
        free(previous);
        previous = current;
        remove_matrix(error_vector);
        free(error_vector);
        jacobi_matrix = Jacobi_matrix(previous);
        current = multiplication(jacobi_matrix, previous);
        temp = Function(previous);
        error_vector = substruction(current, temp);
        remove_matrix(current);
        remove_matrix(temp);
        free(current);
        free(temp);
        current = gauss_method(jacobi_matrix, error_vector);
        remove_matrix(error_vector);
        free(error_vector);
        error_vector = substruction(current, previous);
        remove_matrix(jacobi_matrix);
        free(jacobi_matrix);
    }
    printf("steps: %d\n", step);

    remove_matrix(error_vector);
    free(error_vector);
    remove_matrix(previous);
    free(previous);
    return current;
}

int main(void) {
    int i;
    float epsilon;
    printf("Enter the calculation accuracy:");
    scanf_s("%f", &epsilon);

    Matrix* result;
    begin_values_initialization();

    if (epsilon <= 0) {
        fprintf(stderr, "Negative value of error\n");
        return 0;
    }
    printf("newton_method:\n");
    result = newton_method(Function, Jacobi_matrix, epsilon);
    print_matrix(result, stdout);
    remove_matrix(result);
    free(result);


    printf("simple_iteration_method:\n");
    result = simple_iteration_method(simple_iteration_function, epsilon);
    print_matrix(result, stdout);
    remove_matrix(result);
    free(result);

    begin_values_removing();

    return 0;
}