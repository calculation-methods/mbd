#include <iostream>
#include <cmath>
#include <fstream>


using namespace std;

double g(double x) {
    // Переделал функцию для метода простой итерации: теперь имеет такой вид = (2x + 1)^(1/4)
    // x^4 = 2x + 1   --->   (2x+1)^(0.25)
    return pow(2 * x + 1, 0.25);
}

double f(double x) {
    return pow(x, 4) - 2*x - 1;
}

double simpleIteration(double initialGuess, double eps, int &iterations) {
    double x0 = initialGuess;
    double x1;

    iterations = 0;
    do {
        x1 = x0;
        x0 = g(x1);
        iterations++;
    } while (abs(x1 - x0) >= eps);


    return x0;
}


double df(double x) {
    return 4 * pow(x, 3) - 2;
}


double newtonMethod(double initialGuess, double eps, int &iterations) {
    double x0 = initialGuess;

    iterations = 0;
    do {
        double fx = f(x0);
        double dfx = df(x0);

        double x1 = x0 - fx / dfx;
        iterations++;

        if (abs(x1 - x0) < eps)
            return x1;

        x0 = x1;
    } while (true);
}

int main() {
    double initialGuess = 1;
    double eps = 0.0001;
    int iterations_newton = 0;
    int iterations = 0;
    ofstream fout("answer1.txt");
    double root = simpleIteration(initialGuess, eps, iterations);
    double root2 = newtonMethod(initialGuess, eps, iterations_newton);

    fout << "Solution with iteration method" << " with eps = " << eps << ":\n"  << root << endl;
    fout << "count of iterations: " << iterations_newton << endl;
    fout << "Solution with Newton method" << " with eps = " << eps << ":\n"  << root2 << endl;
    fout << "count of iterations: " << iterations << endl;

    return 0;
}