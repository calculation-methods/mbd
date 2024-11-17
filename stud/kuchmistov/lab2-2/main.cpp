#include <iostream>
#include <cmath>
#include <utility>

using namespace std;

const int MAX_ITERATIONS = 1000;

double F1(double x1, double x2) {
    return pow(x1, 2) / pow(3, 2) + pow(x2, 2) / pow(3.0 / 2.0, 2) - 1;
}

double F2(double x1, double x2) {
    return 3 * x2 - exp(x1) - x1;
}

double G1(double x1, double x2) {
    return sqrt(1 - pow(x2, 2) / 4.5) * 3.0 / 2.0;
}

double G2(double x1, double x2) {
    return (exp(x1) + x1) / 3.0;
}

double dF1_dx1(double x1, double x2) {
    return 2 * x1 / pow(3, 2);
}

double dF1_dx2(double x1, double x2) {
    return 2 * x2 / pow(3.0 / 2.0, 2);
}

double dF2_dx1(double x1, double x2) {
    return -exp(x1) - 1;
}

double dF2_dx2(double x1, double x2) {
    return 3;
}

pair<double, double> simple_iteration(double accuracy, double x1, double x2, int &iters) {
    while (iters < 1000) {
        double new_x1 = G1(x1, x2);
        double new_x2 = G2(x1, x2);

        if (fabs(new_x1 - x1) < accuracy && fabs(new_x2 - x2) < accuracy) {
            return {new_x1, new_x2};
        }

        x1 = new_x1;
        x2 = new_x2;
        iters++;
    }

    return {x1, x2};
}

pair<double, double> newton_method(double x1, double x2, double epsilon) {
    double x1_new, x2_new;
    int iterations = 0;

    do {
        double determinant = dF1_dx1(x1, x2) * dF2_dx2(x1, x2) - dF1_dx2(x1, x2) * dF2_dx1(x1, x2);
        x1_new = x1 - (F1(x1, x2) * dF2_dx2(x1, x2) - F2(x1, x2) * dF1_dx2(x1, x2)) / determinant;
        x2_new = x2 - (F2(x1, x2) * dF1_dx1(x1, x2) - F1(x1, x2) * dF2_dx1(x1, x2)) / determinant;

        iterations++;
        if (iterations > MAX_ITERATIONS) {
            cout << "Newton's Method did not converge within max iterations." << endl;
            break;
        }

        x1 = x1_new;
        x2 = x2_new;

    } while (abs(F1(x1, x2)) > epsilon || abs(F2(x1, x2)) > epsilon);

    cout << endl;
    cout << "Newton's Method:" << endl;
    cout << "Root (x1): " << x1_new << endl;
    cout << "Root (x2): " << x2_new << endl;
    cout << "Number of iterations: " << iterations << endl;

    return make_pair(x1_new, x2_new);
}

int main() {
    double x1 = 1.5;
    double x2 = 1.5;

    double epsilon;
    cout << "Enter the value of epsilon: ";
    cin >> epsilon;

    pair<double, double> result_newton = newton_method(x1, x2, epsilon);
    int iterations = 0;
    pair<double, double> result_simple = simple_iteration(epsilon, x1, x2, iterations);

    cout << endl;
    cout << "Simple Iteration Method:" << endl;
    cout << "Root (x1): " << result_simple.first << endl;
    cout << "Root (x2): " << result_simple.second << endl;
    cout << "Number of iterations: " << iterations << endl;

    return 0;
}
