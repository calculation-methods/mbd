#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double calculate_norma(const vector<vector<double>>& A) {
    double g = 0.0;
    
    for (int i = 0; i < A.size(); i++)
    {
        double h = 0;
        for (int j = 0; j < A.size(); j++)
        {
            if (A[i][j] > 0)
                h = h + A[i][j];
            else 
                h = h - A[i][j];
        }
        if (h > g) g = h;
    }
    return g;
}

void jacobi_method(const vector<vector<double>> matrix, vector<vector<double>> &matA, 
                                    const vector<double> &vecd, vector<double> &vecb){
    for (int i = 0; i < matrix.size(); i++)
    {
        if (matrix[i][i] == 0)
            break;
        for (int j = 0; j < matrix.size(); j++)
        {
            if (i != j)
                matA[i][j] = -matrix[i][j]/matrix[i][i];
            else
                matA[i][j] = 0;
        }
        vecb[i] = vecd[i]/matrix[i][i];
    }
}

void simple_iteration(const vector<vector<double>> matA, const vector<double> &vecb, vector<double> &x,  vector<double> &y, 
                            double f, const double g, const double e, int &k, const int n){
    while (f > e)
    {
        for (int i = 0; i < n; i++)
            y[i] = x[i];
        for (int i = 0; i < n; i++)
        {
            x[i] = vecb[i];
            for (int j = 0; j < n; j++)
                x[i] = x[i] + matA[i][j]*y[j];     
        }
        f = 0;
        for (int i = 0; i < n; i++)
        {
            if ((x[i] - y[i]) > f) f = x[i] - y[i];
            if ((y[i] - x[i]) > f) f = y[i] - x[i];
        }
        f = f*g/(1 - g);
        k = k + 1;
    }
}

void seidel_method(const vector<vector<double>> matA, const vector<double> &vecb, vector<double> &x,  vector<double> &y, 
                            double f, const double g, const double e, int &k, const int n){
    while (f > e)
    {
        for (int i = 0; i < n; i++)
            y[i] = x[i];
        for (int i = 0; i < n; i++)
        {
            x[i] = vecb[i];
            for (int j = 0; j < i-1; j++)
                x[i] = x[i] + matA[i][j]*x[j];
            for (int j = i - 1; j < n; j++)
                x[i] = x[i] + matA[i][j]*y[j];    
        }
        f = 0;
        for (int i = 0; i < n; i++)
        {
            if ((x[i] - y[i]) > f) f = x[i] - y[i];
            if ((y[i] - x[i]) > f) f = y[i] - x[i];
        }
        f = f*g/(1 - g);
        k = k + 1;
    }
}


int main() {
    vector<vector<double>> matrix(4, vector <double>(4, 0)), matA(4, vector <double>(4, 0));
    vector<double> vecd(4, 0), vecb(4, 0), x(4, 0), y(4, 0);
    double e;
    int n = matrix.size();

    ifstream fina("matrix.txt"), finb("col.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fina >> matrix[i][j];
    }
    for (int i = 0; i < vecd.size(); i++)
    {
        finb >> vecd[i];
    }
    finb >> e;

    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;

    fout << "Заданная точность: " << e << endl;

    jacobi_method(matrix, matA, vecd, vecb);

    double g = calculate_norma(matA);
    
    if (g >= 1)
    {
        fout << "Условие сходимости не выполнено!" << endl;
        return 0;
    }

    double f = 0;
    for (int i = 0; i < n; i++)
    {
        x[i] = vecb[i];
        if (x[i] > f) f = x[i];
        if (-x[i] > f) f = -x[i];
    }
    f = f*g/(1 - g);
    int k = 0, k0 = 0;

    double f0 = f;
    vector<double> x0 = x, y0 = y;

    simple_iteration(matA, vecb, x, y, f, g, e, k, n);

    fout << "Решение системы методом простых итераций:" << endl;
    for (size_t i = 0; i < x.size(); ++i)
        fout << "x[" << i << "] = " << x[i] << endl;
    fout << "Количество итераций: " << k << endl;

    f = f0;
    y = y0;
    x = x0;
    k0 = k;
    k = 0;

    seidel_method(matA, vecb, x, y, f, g, e, k, n);

    fout << "Решение системы методом Зейделя:" << endl;
    for (size_t i = 0; i < x.size(); ++i)
        fout << "x[" << i << "] = " << x[i] << endl;
    fout << "Количество итераций: " << k << endl;

    if (k0 > k)
        fout << "Метод Зейделя сходится быстрее метода простых итераций" << endl;
    else if (k > k0)
        fout << "Метод простых итераций сходится быстрее метода Зейделя" << endl;
    else 
        fout << "Методы сходятся с одинаковой скоростью" << endl;       

    return 0;
}