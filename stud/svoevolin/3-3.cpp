#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "matrix.h" 

using namespace std;

struct polynomial {
private:
    vector<double> v;

public:
    polynomial(vector<double> _v = {}) {
        v = _v;
    }

    double size() const {
        return v.size();
    }

    double operator[](int idx) const {
        return v[idx];
    }

    double& operator[](int idx) {
        return v[idx];
    }

    double calculate(double x) const {
        double res = 0;
        double cur = 1;
        for (int i = 0; i < v.size(); i++) {
            res += cur * v[i];
            cur = cur * x;
        }
        return res;
    }
};

ostream& operator<<(ostream& stream, polynomial a)
{
    for (int i = a.size() - 1; i > 0; i--)
        stream << ((i == a.size() - 1) ? a[i] : abs(a[i])) << "*" << "x^" << i << ((a[i - 1] > 0) ? '+' : '-');
    stream << abs(a[0]);
    return stream;
}

polynomial least_squares(const vector<double>& x, const vector<double>& y, int m, double& sum_sq_error) {
    int n = x.size(); 
    m++;  

    matrix phi(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            phi[i][j] = pow(x[i], j);
        }
    }

    matrix G = transposition(phi) * phi;
    matrix Y(n, 1);
    for (int i = 0; i < n; i++)
        Y[i][0] = y[i];
    matrix Z = transposition(phi) * Y;
    matrix A = solve_gauss(lu_decomposition(G), Z);

    vector<double> a(m);
    for (int i = 0; i < m; i++)
        a[i] = A[i][0];

    sum_sq_error = 0.0;
    for (int i = 0; i < n; i++) {
        double approx_y = 0.0;
        for (int j = 0; j < m; j++) {
            approx_y += a[j] * pow(x[i], j);
        }
        sum_sq_error += pow(y[i] - approx_y, 2);
    }

    return polynomial(a);
}

int main() {
    vector<double> x = {-0.7, -0.4, -0.1, 0.2, 0.5, 0.8};
    vector<double> y = {1.6462, 1.5823, 1.571, 1.5694, 1.5472, 1.4435};

    ofstream fout("answer3-3.txt");
    fout.precision(5);
    fout << fixed;

    double error1;
    polynomial p1 = least_squares(x, y, 1, error1);

    double error2;
    polynomial p2 = least_squares(x, y, 2, error2);

    fout << "Приближающий многочлен 1-ой степени: ";
    fout << p1 << endl;

    fout << "Приближающий многочлен 2-ой степени: ";
    fout << p2 << endl;

    fout << "Сумма квадратов ошибок для 1-ой степени: " << error1 << endl;
    fout << "Сумма квадратов ошибок для 2-ой степени: " << error2 << endl;
}

