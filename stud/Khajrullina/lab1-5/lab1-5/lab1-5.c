#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lab1.h"

static inline double absolute(const double a) {
    return a > 0 ? a : -a;
}
static inline sign(const double a) {
    return a > 0 ? 1 : (a < 0 ? -1 : 0);
}
Matrix** QR_decomposition(Matrix* matrix) {
    Matrix** QR, * vector, * mat1, * mat2;
    int i, j, k;
    if (!matrix || matrix->height != matrix->width) {
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
        double s = 0;
        vector = create_matrix();
        resize_matrix(vector, matrix->height, 1);
        for (i = 0; i < matrix->height; i++)
            vector->data[i][0] = i < j ? 0 : QR[1]->data[i][j];
        for (i = j; i < vector->height; i++)
            s += vector->data[i][0] * vector->data[i][0];
        vector->data[j][0] += sign(vector->data[j][0]) * sqrt(s);

        s = 0;
        mat1 = transpose_matrix(vector);
        mat2 = multiple_matrix(vector, mat1);
        remove_matrix(mat1);
        free(mat1);

        for (i = j; i < vector->height; i++)
            s += mat2->data[i][i];
        for (i = 0; i < vector->height; i++)
            for (k = 0; k < vector->height; k++)
                mat2->data[i][k] = (i == k ? 1 : 0) - 2 * mat2->data[i][k] / s;
        mat1 = multiple_matrix(QR[0], mat2);

        for (i = 0; i < QR[0]->height; i++)
            for (k = 0; k < QR[0]->width; k++)
                QR[0]->data[i][k] = mat1->data[i][k];

        remove_matrix(mat1);
        free(mat1);

        mat1 = multiple_matrix(mat2, QR[1]);
        for (i = 0; i < QR[1]->height; i++)
            for (k = 0; k < QR[1]->width; k++)
                QR[1]->data[i][k] = mat1->data[i][k];

        remove_matrix(mat1);
        free(mat1);
        remove_matrix(mat2);
        free(mat2);
        remove_matrix(vector);
        free(vector);
    }
    return QR;
}
int stop(Matrix* matrix, double epsilon) {
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
    Matrix** QR, * alpha;
    int step, i, j;
    alpha = create_matrix();
    resize_matrix(alpha, matrix->height, matrix->width);
    for (i = 0; i < matrix->height; i++)
        for (j = 0; j < matrix->width; j++)
            alpha->data[i][j] = matrix->data[i][j];
    for (step = 1; !stop(alpha, epsilon); step++) {
        QR = QR_decomposition(alpha);
        remove_matrix(alpha);
        free(alpha);
        alpha = multiple_matrix(QR[1], QR[0]);

        remove_matrix(QR[0]);
        free(QR[0]);
        remove_matrix(QR[1]);
        free(QR[1]);
        free(QR);
    }
    QR = (Matrix**)malloc(sizeof(Matrix*) * 2);
    QR[0] = create_matrix();
    QR[1] = create_matrix();
    for (j = 0; j < alpha->width - 1; j++) {
        double s = 0;
        for (i = j + 1; i < alpha->height; i++)
            s += alpha->data[i][j] * alpha->data[i][j];
        if (epsilon > sqrt(s)) {
            resize_matrix(QR[0], QR[0]->height + 1, 1);
            QR[0]->data[QR[0]->height - 1][0] = alpha->data[j][j];
        } else {
            double b = alpha->data[j][j] + alpha->data[j + 1][j + 1];
            double c = alpha->data[j][j] * alpha->data[j + 1][j + 1] - alpha->data[j][j + 1] * alpha->data[j + 1][j];
            resize_matrix(QR[1], QR[1]->height + 2, 2);
            QR[1]->data[QR[1]->height - 1][0] = QR[1]->data[QR[1]->height - 2][0] = 0.5 * b;
            QR[1]->data[QR[1]->height - 1][1] = -(QR[1]->data[QR[1]->height - 2][1] = 0.5 * sqrt(4 * c - b * b));
            j++;
        }
    }
    if (QR[0]->height + QR[1]->height != alpha->height) {
        resize_matrix(QR[0], QR[0]->height + 1, 1);
        QR[0]->data[QR[0]->height - 1][0] = alpha->data[alpha->height - 1][alpha->height - 1];
    }
    remove_matrix(alpha);
    free(alpha);
    return QR;
}

int main(void) {
    int i;
    double epsilon;
    Matrix* matrix = create_matrix(), ** result;
    FILE* fmatrix, * fepsilon;

    fmatrix = fopen("lab1-5m.txt", "r");
    fepsilon = fopen("lab1-5e.txt", "r");
    fscanf(fepsilon, "%lf", &epsilon);
    if (epsilon <= 0) {
        fprintf(stderr, "Negative value of error\n");
        return 0;
    }
    if (!fmatrix || !fepsilon) {
        return 0;
    }

    scan_matrix(matrix, fmatrix);
    fclose(fmatrix);
    fclose(fepsilon);
    if (result = QR_algorithm(matrix, epsilon)) {
        if (result[0]->height) {
            printf("Real values:\n");
            print_matrix(result[0], stdout);
        }
        if (result[1]->height) {
            printf("Complex values:\n");
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
