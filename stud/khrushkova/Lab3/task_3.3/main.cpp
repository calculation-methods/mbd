#include <bits/stdc++.h>
using namespace std;

// Функции из 1.1
pair<vector<vector<double>>, vector<vector<double>>> LU(vector<vector<double>>& X, vector<vector<double>>& Y, int n) {
    vector<vector<double>> L(n);
    vector<vector<double>> U = X;

    // Заполнение нижней диагональной матрицы нулями
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            L[i].push_back(0);


    for (int k=0; k<n; k++) {
        // Меняем местами текущую строку с одной из нижестоящих, если диагнальный элемент на этой строке нулевой
        if (U[k][k] == 0) {
            for (int i=k+1; i<n; i++) {
                if (U[i][k] != 0) {
                    swap(X[k], X[i]);
                    swap(Y[k], Y[i]);
                    swap(U[k], U[i]);
                    swap(L[k], L[i]);
                    break;
                }
            }
        }
        // Заполненеие диагональных матриц с помощью формул из методички, логики и здравого смысла
        // Стоит помнить, что могут возникать небольшие погрешности при делении
        L[k][k] = 1;
        for (int i=k+1; i<n; i++) {
            L[i][k] = U[i][k]/U[k][k];
            if (U[i][k] == 0)
                continue;
            for(int j=k; j<n; j++)
                U[i][j] -= L[i][k]*U[k][j];

        }
    }

    return make_pair(L, U);
}


vector<vector<double>> calculate_result(vector<vector<double>>& X, vector<vector<double>>& Y, int n) {
    vector<vector<double>> L, U;
    tie(L, U) = LU(X, Y, n);
    vector<vector<double>> res = Y;

    int m = res[0].size();
    // Ещё одна реализация формул из методички
    for (int k=0; k<m; k++)
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                res[i][k] -= res[j][k]*L[i][j];
    for (int k=0; k<m; k++) {
        for (int i=n-1; i>-1; i--) {
            for (int j=i+1; j<n; j++) {
                res[i][k] -= res[j][k]*U[i][j];
            }
            res[i][k] /= U[i][i];
        }
    }

    return res;
}

int main() {
    vector<double> x = {0.0, 1.7, 3.4, 5.1, 6.8, 8.5}, y = {0.0, 3.0038, 5.2439, 7.3583, 9.4077, 11.415};
    
    vector<double> s(7, 0);
    for (int i=0; i<6; i++){
        for (int j=0; j<4; j++)
            s[j] += pow(x[i], j+1);
        for (int j=0; j<3; j++)
            s[j+4] += y[i]*pow(x[i], j);
    }

    vector<vector<double>> X = {
        {6.0, s[0]},
        {s[0], s[1]}
    };

    vector<vector<double>> Y = {
        {s[4]},
        {s[5]}
    };

    vector<vector<double>> coefficents = calculate_result(X, Y, 2);
    cout << "Coefficents " << coefficents[0][0] << " " << coefficents[1][0];
    double loss = 0;
    for (int i = 0; i < 6; i++)
        loss += pow(coefficents[0][0] + coefficents[1][0] * x[i] - y[i], 2);
    cout << "\t\t\tLoss = " << loss << endl;

    X = {
        {6, s[0], s[1]},
        {s[0], s[1], s[2]},
        {s[1], s[2], s[3]}
    };

    Y = {
        {s[4]},
        {s[5]},
        {s[6]}
    };

    coefficents = calculate_result(X, Y, 3);
    cout << "Coefficents " << coefficents[0][0] << " " << coefficents[1][0] << " " << coefficents[2][0];
    loss = 0;
    for (int i = 0; i < 6; i++)
        loss += pow(coefficents[0][0] + coefficents[1][0] * x[i] + coefficents[2][0] * pow(x[i], 2) - y[i], 2);
    cout << "\t\tLoss = " << loss << endl;

    return 0;
}