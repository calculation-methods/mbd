#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

void solve_system(vector<vector<double>>& a, vector<double>& b, vector<double>& x, int n) {
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (abs(a[j][i]) > abs(a[pivot][i])) {
                pivot = j;
            }
        }
        swap(a[i], a[pivot]);
        swap(b[i], b[pivot]);
        for (int j = i + 1; j < n; ++j) {
            double factor = a[j][i] / a[i][i];
            for (int k = i; k < n; ++k) {
                a[j][k] -= factor * a[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
    x.assign(n, 0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += a[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / a[i][i];
    }
}

vector<double> MNK(vector<double>& x, vector<double>& y, int degree, int n) {
    vector<vector<double>> a(degree + 1, vector<double>(degree + 1, 0));
    vector<double> b(degree + 1, 0);
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k < n; ++k) {
                a[i][j] += pow(x[k], i + j);
            }
        }
        for (int k = 0; k < n; ++k) {
            b[i] += pow(x[k], i) * y[k];
        }
    }
    vector<double> coefficients;
    int n1 = a.size();
    solve_system(a, b, coefficients, n1);
    return coefficients;
}

double g(vector<double>& coefficients, double x) {
    double result = 0;
    for (size_t i = 0; i < coefficients.size(); ++i) {
        result += coefficients[i] * pow(x, i);
    }
    return result;
}

double sum_of_errors(vector<double>& x, vector<double>& y, vector<double>& coefficients, int n) {
    double sum = 0, error;
    for (int i = 0; i < n; ++i) {
        error = y[i] - g(coefficients, x[i]);
        sum += error * error;
    }
    return sum;
}

int main(){
    int n = 6, m = 1;
    vector<double> x = {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
    vector<double> y = {-1.8415, 0.0, 1.8415, 2.9093, 3.1411, 3.2432};
    ofstream fout("output.txt");
    vector<double> coefficients_degree_1 = MNK(x, y, m, n);
    double sum_degree_1 = sum_of_errors(x, y, coefficients_degree_1, n);
    fout << "Коэффициенты приближающего многочлена первой степени:" << endl;
    for (int i = 0; i < m+1; ++i) {
        fout << coefficients_degree_1[i] << " ";
    }
    fout << endl;
    fout << "Сумма квадратов ошибок: " << sum_degree_1 << endl;

    m = 2;
    vector<double> coefficients_degree_2 = MNK(x, y, m, n);
    double sum_degree_2 = sum_of_errors(x, y, coefficients_degree_2, n);
    fout << "Коэффициенты приближающего многочлена второй степени:" << endl;
    for (int i = 0; i < m+1; ++i) {
        fout << coefficients_degree_1[i] << " ";
    }
    fout << endl;
    fout << "Сумма квадратов ошибок: " << sum_degree_2 << endl;
    return 0;
}