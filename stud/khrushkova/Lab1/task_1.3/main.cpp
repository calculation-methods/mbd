#include <vector>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cmath>

using namespace std;

vector<vector<double>> plus_matrix(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2) {
    int n = matrix1.size(), m = matrix1[0].size();
    vector<vector<double>> res(n, vector<double>(m));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            res[i][j] = matrix1[i][j] + matrix2[i][j];
    return res;
}

vector<vector<double>> multiply(const vector<vector<double>>& matrix1, const vector<vector<double>>& matrix2) {
    int n1 = matrix1.size(), m1 = matrix1[0].size(), m2 = matrix2[0].size();
    vector<vector<double>> res(n1, vector<double>(m2, 0));
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < m2; ++j) {
            double cntr = 0;
            for (int k = 0; k < m1; ++k) {
                cntr += matrix1[i][k] * matrix2[k][j];
            }
            res[i][j] = cntr;
        }
    }
    return res;
}

double get_current_eps(const vector<vector<double>>& v1, const vector<vector<double>>& v2) {
    double eps = 0;
    int n = v1.size();
    for(int i=0; i<n; i++)
        eps += pow(v1[i][0] - v2[i][0], 2);
    return sqrt(eps);
}

vector<vector<double>> simple_iterations(vector<vector<double>>& a, vector<vector<double>>& b, double EPS0) {
    vector<vector<double>> x_previous = b, x_current;
    while (get_current_eps(x_current = plus_matrix(b, multiply(a, x_previous)), x_previous) > EPS0)
        x_previous = x_current;
    return x_previous;
}


vector<vector<double>> seidel_method(const vector<vector<double>>& a, const vector<vector<double>>& b, double EPS0) {
    vector<vector<double>> x_previous = b, x_current = x_previous;
    bool flag = true;
    double eps = 0;
    int n = b.size();
    while (flag or eps > EPS0) {
        flag = false;
        for(int i = 0; i < n; i++) {
            x_current[i][0] = 0;
            for(int j=0; j < n; j++) {
                if (i == j)
                    x_current[i][0] += b[i][0];
                else if (i < j)
                    x_current[i][0] += a[i][j]*x_previous[j][0];
                else
                    x_current[i][0] += a[i][j]*x_current[j][0];
            }
        }
        eps = get_current_eps(x_current, x_previous);
        x_previous = x_current;
    }
    return x_previous;
}


int main() {
    ifstream in("input.txt");
    int n; // Количество неизвестных
    in >> n;
    vector<vector<double>> X(n), Y(n);

    //Чтение матрицы коэффицентов
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j) {
            int number;
            in >> number;
            X[i].push_back(number);
        }

    //Чтение матрицы значений
    for (int i=0; i<n; ++i) {
        int number;
        in >> number;
        Y[i].push_back(number);
    }

    in.close();

    ofstream out("output.txt");

    for(int i=0; i < n; i++) {
        double a = X[i][i];
        if (a == 0)
            continue;
        Y[i][0] /= a;
        for (int j=0; j < n; j++)
            X[i][j] = (i == j) ? 0 : -X[i][j]/a;
    }

    out << "Iterations result:\n";
    for (auto row: simple_iterations(X, Y, 0.001)){
        for (auto el: row)
            out << el << " ";
        out << endl;
    }
    out << endl;

    out << "Seidel result:\n";
    for (auto row: seidel_method(X, Y, 0.001)){
        for (auto el: row)
            out << el << " ";
        out << endl;
    }
    out.close();
    return 0;
}
