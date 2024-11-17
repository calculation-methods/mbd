#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

double Norm(vector<double>& A, int n){
    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += A[i] * A[i];
    }
    double norm = sqrt(sum);
    return norm;
}

int SimpleIter(vector<vector<double>>& A, vector<double>& b, int n, double eps, ofstream& fout){
    vector<double> x_old (n, 0);
    vector<double> x (n, 0);
    vector<double> subtract (n, 0);
    double e = 1;
    for (int i = 0; i < n; i++){
        x_old[i] = b[i] / A[i][i];
    }
    int k = 0;
    while (e > eps){
        for (int i = 0; i < n; i++){
            subtract[i] = 0;
        }
        for (int i = 0; i < n; i++){
            x[i] = b[i] / A[i][i];
            for (int j = 0; j < n; j++){
                if (i == j)
                    continue;
                else
                    x[i] -= A[i][j] / A[i][i] * x_old[j];
            }
        }
        for (int i = 0; i < n; i++){
            subtract[i] = fabs(x[i] - x_old[i]);
        }
        for (int i = 0; i < n; i++){
            x_old[i] = x[i];
        }
        e = Norm(subtract, n);
        k++;
    }
    fout << "Решение методом простых итераций:" << endl;
    for (int i = 0; i < n; i++)
        fout << x_old[i] << " ";
    fout <<endl << "Количество итераций:"<< k << endl;
    return k;
}

int Zeidel(vector<vector<double>>& A, vector<double>& b, int n, double eps, ofstream& fout){
    vector<double> x (n, 0);
    vector<double> subtract (n, 0);
    vector<double> x_new (n, 0);
    double s1 = 0, s2 = 0;
    double e = 1;
    int k = 0;
    while (e >= eps){
        for (int i = 0; i < n; i++)
            x[i] = x_new[i];
        for (int i = 0; i < n; i++){
            s1 = 0;
            s2 = 0;
            for (int j = 0; j < i; j++){
                s1 += A[i][j] * x_new[j];
            }
            for (int j = i + 1; j < n; j++){
                s2 += A[i][j] * x[j];
            }
            x_new[i] = (b[i] - s1 - s2) / A[i][i];
        }
        for (int i = 0; i < n; i++){
            subtract[i] = fabs(x[i] - x_new[i]);
        }
        e = Norm(subtract, n);
        k++;
    }
    fout << "Решение методом Зейделя:" << endl;
    for (int i = 0; i < n; i++)
        fout << x[i] << " ";
    fout <<endl << "Количество итераций:"<< k << endl;
    return k;
}

int main() {
    int n = 4;
    double eps = 0.0001;
    ofstream fout;
    fout.open("output.txt");
    vector<vector<double>> A = {
            {-19, 2, -1, -8},
            {2, 14, 0, -4},
            {6, -5, -20, -6},
            {-6, 4, -2, 15},
    };
    vector<double> b = {38, 20, 52, 43};
    int k1 = SimpleIter(A, b, n, eps, fout);
    int k2 =  Zeidel(A, b, n, eps, fout);
    if (k1 < k2)
        fout << "Метод простых итераций сходится быстрее метода Зейделя"<< endl;
    else if (k1 == k2)
        fout << "Методы сходятся одинаково" << endl;
    else
        fout << "Метод Зейделя сходится быстрее метода простых итераций" << endl;
    return 0;
}
