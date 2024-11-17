#include <vector>
#include <iostream>
#include <fstream>
#include <cstdio>

using namespace std;

// Реализация алгоритма из методички
vector<vector<double>> tridiagonal(vector<vector<double>>& coefficients, vector<vector<double>>& results) {
    int n = coefficients.size();
    double a = 0, b = coefficients[0][0], c = coefficients[0][1], d = results[0][0];
    vector<double> P(n, 0), Q(n, 0);

    P[0] = -c/b;
    Q[0] = d/b;
    for (int i=1; i < n-1; i++){
        a = coefficients[i][i-1];
        b = coefficients[i][i];
        c = coefficients[i][i+1];
        d = results[i][0];

        P[i] = -c/(b + a*P[i-1]);
        Q[i] = (d - a*Q[i-1])/(b + a*P[i-1]);
    }

    a = coefficients[n-1][n-2];
    b = coefficients[n-1][n-1];
    c = 0;
    d = results[n-1][0];

    Q[n-1] = (d - a * Q[n-2]) / (b + a * P[n-2]);

    vector<vector<double>> result(n);
    for(int i=0; i<n; i++)
        result[i].push_back(0);

    result[n-1][0] = Q[n-1];
    for (int i = n-2; i > -1; i--)
        result[i][0] = P[i]*result[i+1][0] + Q[i];

    return result;
}

int main() {
    ifstream in("input.txt");
    int n; // Количество неизвестных
    in >> n;
    vector<vector<double>> X(n), Y(n);

    // Чтение матрицы коэффицентов
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

    vector<vector<double>> result = tridiagonal(X, Y);
    ofstream out("output.txt");

    // Вывод результата в файл
    out << "Result:" << endl;
    for (int i=0; i<n; i++)
        out << "x" << i+1 << " = " << result[i][0] << endl;

    out.close();
    return 0;
}