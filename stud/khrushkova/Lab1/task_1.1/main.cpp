#include <bits/stdc++.h>

using namespace std;


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


void write_matrix_to_file(const vector<vector<double>>& matrix, ofstream& out) {
    // Вывод матрицы в файл
    int n = matrix.size(), m = matrix[0].size();
    for(int i=0; i<n; ++i){
        for(int j=0; j<m; j++)
            out << matrix[i][j] << " ";
        out << endl;
    }
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
    vector<vector<double>> L, U;
    tie(L, U) = LU(X, Y, n);

    out << "U =" << endl;
    write_matrix_to_file(U, out);

    out << endl << "L =" << endl;
    write_matrix_to_file(L, out);


    // Вычисление определителя матрицы
    double determinant = 1;
    for (int i=0; i<n; i++)
        determinant *= U[i][i];
    out << endl << "Determinant = " << determinant << endl;


    // Поиск решений
    vector<vector<double>> result = calculate_result(X, Y, n);
    out << endl << "Result =" << endl;
    write_matrix_to_file(result, out);


    // Инициализация единичной матрицы
    vector<vector<double>> E(n);
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            if (i != j)
                E[i].push_back(0);
            else
                E[i].push_back(1);
    vector<vector<double>> invert = calculate_result(X, E, n); // Нахождение обратной матрицы
    out << endl << "Inverted =" << endl;
    write_matrix_to_file(invert, out);

    out.close();
    return 0;
}