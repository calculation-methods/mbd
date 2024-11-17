#include <iostream>
#include <cmath>

using namespace std;

const double INITIAL_VALUE = 1.5;
const double EPSILON = 1e-6;

double function(double x) {
    return pow(x, 3) - 2 * pow(x, 2) - 10 * x + 15;
}

double derivative(double x) {
    return 3 * pow(x, 2) - 4 * x - 10;
}

double iter_function(double x){
    return pow(2 * pow(x, 2) + 10 * x - 15, 1.0 / 3.0);
}

double absolute(double a) {
    return a > 0 ? a : -a;
}

double simple_iteration(double x0, double eps) {
    double x = iter_function(x0);
    double diff =  abs(x - x0);
    x0 = x;
    int k = 0;
    while (diff >= eps) {
        x =  iter_function(x0);
        diff =  abs(x - x0);
        x0 = x;
        k++;
    }
    cout << "Simple Iteration Method:" << endl;
    cout << "Root: " << x << endl;
    cout << "Number of iterations: " << k << endl;
    return x;
}

double newton_method(double (*func)(double), double (*deriv)(double), double epsilon) {
    double result, previous = INITIAL_VALUE;
    int iterations = 0;
    do {
        result = previous - func(previous) / deriv(previous);
        iterations++;
        previous = result;
    } while (absolute(func(result)) >= epsilon);
    cout << endl;
    cout << "Newton's Method:" << endl;
    cout << "Root: " << result << endl;
    cout << "Number of iterations: " << iterations << endl;
    return result;
}


int main() {
    double epsilon = EPSILON;

    cout << "Enter the value of epsilon: ";
    cin >> epsilon;

    if (epsilon <= 0) {
        cerr << "Error: Epsilon should be positive." << endl;
        return 1;
    }

    simple_iteration(INITIAL_VALUE, epsilon);
    newton_method(function, derivative, epsilon);

    return 0;
}
