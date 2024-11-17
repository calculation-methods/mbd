#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

vector<double> simple_iteration(vector<vector<double>>& matrix, vector<double>& b, double eps, int &iter, int n) {

    vector<double> x(n, 0.0);
    while (true) {
        iter++;
        vector<double> x_new(n, 0.0);
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    sum += matrix[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / matrix[i][i];
        }
        double diff = 0.0;
        for (int i = 0; i < n; i++) {
            diff = max(diff, fabs(x[i] - x_new[i]));
        }
        if (diff < eps) {
            break;
        }
        x = x_new;
    }
    return x;
}

vector<double> Seidel(vector<vector<double>>& matrix, vector<double>& b, double eps, int& iter, int n) {

    vector<double> x(n, 0.0);
    vector<double> x_new(n, 0.0);
    while (true) {
        iter++;
        for (int i = 0; i < n; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            for (int j = 0; j < i; j++) {
                sum1 += matrix[i][j] * x_new[j];
            }
            for (int j = i + 1; j < n; j++) {
                sum2 += matrix[i][j] * x[j];
            }
            x_new[i] = (b[i] - sum1 - sum2) / matrix[i][i];
        }
        double diff = 0.0;
        for (int i = 0; i < n; i++) {
            diff = max(diff, fabs(x[i] - x_new[i]));
        }
        if (diff < eps) {
            break;
        }
        x = x_new;
    }
    return x;
}

int main() {
    int n = 4;
   vector<vector<double>> matrix(n, vector <double>(n, 0));
    vector<double> b(n, 0);
    ifstream in1("matrix.txt"), in2("b.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++){
            in1 >> matrix[i][j];
        }
    }

    for (int i = 0; i < n; i++)
    {
        in2 >> b[i];
    }
    double eps = 0.0001;
    int iteration_1 = 0;
    int iteration_2 = 0;

    vector<double> x_1 = simple_iteration(matrix, b, eps, iteration_1, n);
    ofstream out("output.txt");
    out << "Решение системы методом простых итераций:" << endl;
    for (int i = 0; i < n; ++i)
        out << x_1[i] << endl;
    out << "Количество итераций: " << iteration_1 << endl;

    vector<double> x_2 = Seidel(matrix, b, eps, iteration_2, n);
    out << "Решением системы методом Зейделя:" << endl;
    for (int i = 0; i < n; ++i)
        out << x_2[i] << endl;
    out << "Количество итераций: " << iteration_2 << endl;

    return 0;
}