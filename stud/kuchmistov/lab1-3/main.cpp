#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> simpleIterationMethod(const vector<vector<double>>& A, const vector<double>& b, double epsilon) {
    int n = A.size();
    vector<double> x(n, 0.0);
    vector<double> x_new(n);

    int iterations = 0;
    double error = epsilon + 1.0;

    while (error > epsilon) {
        for (int i = 0; i < n; ++i) {
            double sum = b[i];
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum -= A[i][j] * x[j];
                }
            }
            x_new[i] = sum / A[i][i];
        }

        error = 0.0;
        for (int i = 0; i < n; ++i) {
            error = max(error, abs(x_new[i] - x[i]));
            x[i] = x_new[i];
        }

        iterations++;
    }

    cout << "Iterations: " << iterations << endl;
    return x;
}

vector<double> gaussSeidelMethod(const vector<vector<double>>& A, const vector<double>& b, double epsilon) {
    int n = A.size();
    vector<double> x(n, 0.0);
    vector<double> x_new(n);

    int iterations = 0;
    double error = epsilon + 1.0;

    while (error > epsilon) {
        for (int i = 0; i < n; ++i) {
            double sum1 = 0.0;
            for (int j = 0; j < i; ++j) {
                sum1 += A[i][j] * x_new[j];
            }

            double sum2 = 0.0;
            for (int j = i + 1; j < n; ++j) {
                sum2 += A[i][j] * x[j];
            }

            x_new[i] = (b[i] - sum1 - sum2) / A[i][i];
        }

        error = 0.0;
        for (int i = 0; i < n; ++i) {
            error = max(error, abs(x_new[i] - x[i]));
            x[i] = x_new[i];
        }

        iterations++;
    }

    cout << "Iterations: " << iterations << endl;
    return x;
}

int main() {
    vector<vector<double>> A = {
        {-22, -2, -6, 6},
        {3, -17, -3, 7},
        {2, 6, -17, 5},
        {-1, -8, 8, 23}
    };
    
    vector<double> b = {96, -26, 35, -234};
    double epsilon = 1e-6;

    cout << "SimpleIterationMethod:" << endl;
    vector<double> solution_simple_iteration = simpleIterationMethod(A, b, epsilon);
    cout << "x1 = " << solution_simple_iteration[0] << ", x2 = " << solution_simple_iteration[1] << ", x3 = " << solution_simple_iteration[2] << ", x4 = " << solution_simple_iteration[3] << endl;

    cout << endl;

    cout << "SeidelMethod:" << endl;
    vector<double> solution_gauss_seidel = gaussSeidelMethod(A, b, epsilon);
    cout << "x1 = " << solution_gauss_seidel[0] << ", x2 = " << solution_gauss_seidel[1] << ", x3 = " << solution_gauss_seidel[2] << ", x4 = " << solution_gauss_seidel[3] << endl;

    return 0;
}
