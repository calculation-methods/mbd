#include <stdio.h>
#include <string.h>

double Function(double x) {
    return (3*x + 4) / (2 * x + 7);
}

double rectangle_method(double (*Function)(double), double a, double b, double step) {
    double x, sum = 0;
    if (a >= b) {
        fprintf(stderr, "Incorrect limits of interval\n");
        return 0;
    }
    for (x = a + step; x < b; x += step)
        sum += step * Function(x - step / 2);
    x -= step;
    sum += (b - x) * Function((x + b) / 2);
    return sum;
}

double trapezoidal_method(double (*Function)(double), double a, double b, double step) {
    double x, sum = 0;
    if (a >= b) {
        fprintf(stderr, "Incorrect limits of interval\n");
        return 0;
    }
    for (x = a + step; x < b; x += step)
        sum += 0.5 * step * (Function(x) + Function(x - step));
    x -= step;
    sum += 0.5 * (b - x) * (Function(x) + Function(b));
    return sum;
}

double simpson_method(double (*Function)(double), double a, double b, double step) {
    double x, sum = 0;
    if (a >= b) {
        fprintf(stderr, "Incorrect limits of interval\n");
        return 0;
    }
    for (x = a + step; x < b; x += step)
        sum += step * (Function(x) + Function(x - step) + 4 * Function(x - step / 2)) / 6.0;
    x -= step;
    sum += (b - x) * (Function(x) + Function(b) + 4 * Function((x + b) / 2)) / 6.0;
    return sum;
}

double runge_romberg_method(double first_estimate, double first_step, double second_estimate, double second_step) {
    double k = second_step / first_step;
    if (first_step == second_step) {
        fprintf(stderr, "Equal step of estimates\n");
        return 0;
    }
    return first_estimate + (first_estimate - second_estimate) / (k * k - 1);
}

int main(int argc, char* argv[]) {
    int i;
    float left_limit = -2, right_limit = 2, step1 = 1.0, step2 = 0.5;
    double rectangle1, rectangle2, trapezoidal1, trapezoidal2, simpsons1, simpsons2;

    printf("Step = %.2f\nRectangel method:\n", step1);
    printf("I(f) from %.2f to %.2f = %.4f\n",
        left_limit, right_limit, rectangle1 = rectangle_method(Function, left_limit, right_limit, step1));
    printf("Trapezoidal method:\n");
    printf("I(f) from %.2f to %.2f = %.4f\n",
        left_limit, right_limit, trapezoidal1 = trapezoidal_method(Function, left_limit, right_limit, step1));
    printf("Simpson's method:\n");
    printf("I(f) from %.2f to %.2f = %.8f\n",
        left_limit, right_limit, simpsons1 = simpson_method(Function, left_limit, right_limit, step1));

    printf("\nStep = %.2f\nRectangel method:\n", step2);
    printf("I(f) from %.2f to %.2f = %.4f\n",
        left_limit, right_limit, rectangle2 = rectangle_method(Function, left_limit, right_limit, step2));
    printf("Trapezoidal method:\n");
    printf("I(f) from %.2f to %.2f = %.4f\n",
        left_limit, right_limit, trapezoidal2 = trapezoidal_method(Function, left_limit, right_limit, step2));
    printf("Simpson's method:\n");
    printf("I(f) from %.2f to %.2f = %.8f\n",
        left_limit, right_limit, simpsons2 = simpson_method(Function, left_limit, right_limit, step2));

    printf("\nRunge-Romberg's estimation of result\nRectangel method:\n");
    printf("I(f) from %.2f to %.2f = %.8f\n",
        left_limit, right_limit, runge_romberg_method(rectangle1, step1, rectangle2, step2));
    printf("Trapezoidal method:\n");
    printf("I(f) from %.2f to %.2f = %.8f\n",
        left_limit, right_limit, runge_romberg_method(trapezoidal1, step1, trapezoidal2, step2));
    printf("Simpson's method:\n");
    printf("I(f) from %.2f to %.2f = %.10f\n",
        left_limit, right_limit, runge_romberg_method(simpsons1, step1, simpsons2, step2));
    return 0;
}