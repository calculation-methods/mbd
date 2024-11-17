#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const double BEGIN_VALUE = 1.0;

double Function(double x) {
    return (x * x * x) + x * x - x - 0.5;
}

double Derivative(double x) {
    return 3 * x * x + 2 * x - 1;
}

double simple_iteration_function(double x) {
    return pow(0.5 + x - x * x, 1.0 / 3.0);
}

double absolute(double a) {
    return a > 0 ? a : -a;
}

double simple_iteration_method(double (*Function)(double), double epsilon) {
    const double SIMPLE_ITERATION_CONSTANT =  0.01;
    double result, previous = BEGIN_VALUE;
    int step;
    for (step = 1; epsilon < SIMPLE_ITERATION_CONSTANT / (1 - SIMPLE_ITERATION_CONSTANT) * absolute((result = Function(previous)) - previous); step++) {
        previous = result;
    }
    printf("steps: %d\n", step);
    return result;
}

double newton_method(double (*Function)(double), double (*Derivative)(double), double epsilon) {
    double result, previous = BEGIN_VALUE;
    int step;
    for (step = 1; epsilon < absolute((result = previous - Function(previous) / Derivative(previous)) - previous); step++) {
        previous = result;
    }
    printf("steps: %d\n", step);
    return result;
}

int main(void) {
    float epsilon;
    
    printf("Enter calculation accuracy:");
    scanf("%f", &epsilon);

    if (epsilon <= 0) {
        fprintf(stderr, "Negative value of error\n");
        return 0;
    }
    printf("Newton method:\n");
    printf("x* = %.4f\n", newton_method(Function, Derivative, epsilon));
    printf("Simple iteration method:\n");
    printf("x* = %.4f\n", simple_iteration_method(simple_iteration_function, epsilon));
    
    return 0;
}