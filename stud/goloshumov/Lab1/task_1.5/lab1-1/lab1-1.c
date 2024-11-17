#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "read.h"

static inline double absolute(const double a) {
    return a > 0 ? a : -a;
}
static inline sign(const double a) {
    return a > 0 ? 1 : (a < 0 ? -1 : 0);
}


Matrix** QR_decomposition(Matrix* matrix) {
    Matrix** QR, * vector, * temp, * temp2;
    int i, j, k;
    if (!matrix || matrix->height != matrix->width) {
        fprintf(stderr, "Not square matrix\n");
        return NULL;
    }
    QR = (Matrix**)malloc(sizeof(Matrix*) * 2);
    QR[0] = create_matrix();
    resize_matrix(QR[0], matrix->height, matrix->width);
    QR[1] = create_matrix();
    resize_matrix(QR[1], matrix->height, matrix->width);

    for (i = 0; i < matrix->height; i++) {
        for (j = 0; j < matrix->width; j++)
            QR[1]->data[i][j] = matrix->data[i][j];
        QR[0]->data[i][i] = 1;
    }

    for (j = 0; j < matrix->width - 1; j++) {
        double sum = 0;
        vector = create_matrix();
        resize_matrix(vector, matrix->height, 1);
        for (i = 0; i < matrix->height; i++)
            vector->data[i][0] = i < j ? 0 : QR[1]->data[i][j];
        for (i = j; i < vector->height; i++)
            sum += vector->data[i][0] * vector->data[i][0];
        vector->data[j][0] += sign(vector->data[j][0]) * sqrt(sum);

        sum = 0;
        temp = matrix_transposition(vector);
        temp2 = multiplication(vector, temp);
        remove_matrix(temp);
        free(temp);

        for (i = j; i < vector->height; i++)
            sum += temp2->data[i][i];
        for (i = 0; i < vector->height; i++)
            for (k = 0; k < vector->height; k++)
                temp2->data[i][k] = (i == k ? 1 : 0) - 2 * temp2->data[i][k] / sum;
        temp = multiplication(QR[0], temp2);

        for (i = 0; i < QR[0]->height; i++)
            for (k = 0; k < QR[0]->width; k++)
                QR[0]->data[i][k] = temp->data[i][k];

        remove_matrix(temp);
        free(temp);

        temp = multiplication(temp2, QR[1]);
        for (i = 0; i < QR[1]->height; i++)
            for (k = 0; k < QR[1]->width; k++)
                QR[1]->data[i][k] = temp->data[i][k];

        remove_matrix(temp);
        free(temp);
        remove_matrix(temp2);
        free(temp2);
        remove_matrix(vector);
        free(vector);
    }
    return QR;
}


int stop_criterion(Matrix* matrix, double epsilon) {
    int mark = 0, i, j;

    for (j = 0; j < matrix->width - 1; j++) {
        double error = 0;
        for (i = j + 1; i < matrix->height; i++)
            error += matrix->data[i][j] * matrix->data[i][j];
        if (sqrt(error) > epsilon)
            if (sqrt(error - matrix->data[j + 1][j] * matrix->data[j + 1][j]) > epsilon || mark)
                return 0;
            else
                mark = 1;
        else
            mark = 0;
    }
    return 1;
}


Matrix** QR_algorithm(Matrix* matrix, double epsilon) {
    Matrix** QR, * lambda;
    int step, i, j;
    lambda = create_matrix();
    resize_matrix(lambda, matrix->height, matrix->width);
    for (i = 0; i < matrix->height; i++)
        for (j = 0; j < matrix->width; j++)
            lambda->data[i][j] = matrix->data[i][j];
    for (step = 1; !stop_criterion(lambda, epsilon); step++) {
        QR = QR_decomposition(lambda);
        remove_matrix(lambda);
        free(lambda);
        lambda = multiplication(QR[1], QR[0]);

        remove_matrix(QR[0]);
        free(QR[0]);
        remove_matrix(QR[1]);
        free(QR[1]);
        free(QR);
    }
    QR = (Matrix**)malloc(sizeof(Matrix*) * 2);
    QR[0] = create_matrix();
    QR[1] = create_matrix();
    for (j = 0; j < lambda->width - 1; j++) {
        double sum = 0;
        for (i = j + 1; i < lambda->height; i++)
            sum += lambda->data[i][j] * lambda->data[i][j];
        if (epsilon > sqrt(sum)) {
            resize_matrix(QR[0], QR[0]->height + 1, 1);
            QR[0]->data[QR[0]->height - 1][0] = lambda->data[j][j];
        }
        else {
            double b = lambda->data[j][j] + lambda->data[j + 1][j + 1];
            double c = lambda->data[j][j] * lambda->data[j + 1][j + 1] - lambda->data[j][j + 1] * lambda->data[j + 1][j];
            resize_matrix(QR[1], QR[1]->height + 2, 2);
            QR[1]->data[QR[1]->height - 1][0] = QR[1]->data[QR[1]->height - 2][0] = 0.5 * b;
            QR[1]->data[QR[1]->height - 1][1] = -(QR[1]->data[QR[1]->height - 2][1] = 0.5 * sqrt(4 * c - b * b));
            j++;
        }
    }
    if (QR[0]->height + QR[1]->height != lambda->height) {
        resize_matrix(QR[0], QR[0]->height + 1, 1);
        QR[0]->data[QR[0]->height - 1][0] = lambda->data[lambda->height - 1][lambda->height - 1];
    }
    remove_matrix(lambda);
    free(lambda);
    return QR;
}

int main(void) {
    int i;
    float epsilon;
    Matrix* matrix = create_matrix(), ** result;
    FILE* fmatrix;

    fmatrix = fopen("lab01-1matrix.txt", "r");

    printf("Enter the calculation accuracy:");
    scanf("%f", &epsilon);
    if (epsilon <= 0) {
        fprintf(stderr, "Negative value of error\n");
        return 0;
    }
    if (!fmatrix) {
        fprintf(stderr, "Invalid name of file\n");
        return 0;
    }

    scan_matrix(matrix, fmatrix);
    fclose(fmatrix);
    if (result = QR_algorithm(matrix, epsilon)) {
        if (result[0]->height) {
            printf("Real eigenvalues\n");
            print_matrix(result[0], stdout);
        }
        if (result[1]->height) {
            printf("Complex eigenvalues\n");
            print_matrix(result[1], stdout);
        }
        remove_matrix(result[0]);
        free(result[0]);
        remove_matrix(result[1]);
        free(result[1]);
        free(result);
    }
    remove_matrix(matrix);
    free(matrix);
    return 0;
}
