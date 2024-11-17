#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

double sumOfSquaredErrors(const vector<double>& x, const vector<double>& y, const vector<double>& y_approx) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double error = y[i] - y_approx[i];
        sum += error * error;
    }
    return sum;
}

vector<double> findPolynomialCoefficients(const vector<double>& x, const vector<double>& y, int degree) {
    int n = x.size();
    vector<double> sumX(2 * degree + 1, 0.0);
    vector<double> sumY(degree + 1, 0.0);
    vector<double> a(degree + 1, 0.0);

    for (int i = 0; i < n; ++i) {
        double xi = x[i];
        double yi = y[i];
        for (int j = 0; j <= 2 * degree; ++j) {
            sumX[j] += pow(xi, j);
        }
        for (int j = 0; j <= degree; ++j) {
            sumY[j] += yi * pow(xi, j);
        }
    }

    vector<vector<double>> A(degree + 1, vector<double>(degree + 2, 0.0));

    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            A[i][j] = sumX[i + j];
        }
        A[i][degree + 1] = sumY[i];
    }

    for (int i = 0; i < degree; ++i) {
        for (int k = i + 1; k <= degree; ++k) {
            double ratio = A[k][i] / A[i][i];
            for (int j = 0; j <= degree + 1; ++j) {
                A[k][j] -= ratio * A[i][j];
            }
        }
    }

    for (int i = degree; i >= 0; --i) {
        a[i] = A[i][degree + 1];
        for (int j = i + 1; j <= degree; ++j) {
            if (j != degree + 1) {
                a[i] -= A[i][j] * a[j];
            }
        }
        a[i] /= A[i][i];
    }
    return a;
}

double evaluatePolynomial(const vector<double>& a, double x) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * pow(x, i);
    }
    return result;
}

int main() {
    vector<double> x = {-0.9, 0.0, 0.9, 1.8, 2.7, 3.6};
    vector<double> y = {-1.2689, 0.0, 1.2689, 2.6541, 4.4856, 9.9138};
    int n = x.size();

    vector<double> a1 = findPolynomialCoefficients(x, y, 1);
    cout << "First-degree polynomial coefficients:" << endl;
    cout << "a0 = " << fixed << setprecision(4) << a1[0] << ", a1 = " << a1[1] << endl;

    vector<double> y_approx1(n);
    for (int i = 0; i < n; ++i) {
        y_approx1[i] = evaluatePolynomial(a1, x[i]);
    }

    double sum_sq_error_1 = sumOfSquaredErrors(x, y, y_approx1);
    cout << "Sum of squared errors for first-degree polynomial: " << sum_sq_error_1 << endl;

    vector<double> a2 = findPolynomialCoefficients(x, y, 2);
    cout << "\nSecond-degree polynomial coefficients:" << endl;
    cout << "a0 = " << a2[0] << ", a1 = " << a2[1] << ", a2 = " << a2[2] << endl;

    vector<double> y_approx2(n);
    for (int i = 0; i < n; ++i) {
        y_approx2[i] = evaluatePolynomial(a2, x[i]);
    }

    double sum_sq_error_2 = sumOfSquaredErrors(x, y, y_approx2);
    cout << "Sum of squared errors for second-degree polynomial: " << sum_sq_error_2 << endl;

    return 0;
}
