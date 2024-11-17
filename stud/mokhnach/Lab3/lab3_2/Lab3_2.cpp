#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

class matrix
{
    private:
        vector <vector <double>> _obj;
    public:
        int  cols = 0, rows = 0;

        matrix() {}
        matrix(int _rows, int _cols)
        {
            rows = _rows;
            cols = _cols;
            _obj = vector <vector <double>>(rows, vector <double>(cols));
        }

        vector <double> &operator[](int i)
        {
            return _obj[i];
        }

        operator double()
        {
            return _obj[0][0];
        }

};

istream& operator>>(istream& stream, matrix& m)
{
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
            stream >> m[i][j];
    }
    return stream;
}

ostream& operator<<(ostream& stream, matrix m)
{
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
            stream << m[i][j] << ' ';
        stream << '\n';
    }
    return stream;
}


matrix solve_SLE (matrix& A, matrix& b)
{
    int n = A.cols;
    vector <double> p(n), q(n);
    matrix ans(n, 1);
    p[0] = -A[0][1] / A[0][0];
    q[0] = b[0][0] / A[0][0];
    for (int i = 1; i < n; i++)
    {
        if (i != n - 1)
            p[i] = -A[i][i + 1] / (A[i][i] + A[i][i - 1] * p[i - 1]);
        else
            p[i] = 0;
            q[i] = (b[i][0] - A[i][i - 1] * q[i - 1]) / (A[i][i] + A[i][i - 1] * p[i - 1]);
    }
    ans[n - 1][0] = q[n - 1];
    for (int i = n - 2; i >= 0; i--)
        ans[i][0] = p[i] * ans[i + 1][0] + q[i];
    return ans;
}

double spline(vector<double> &x, vector<double> &y, double dot) {
    int n = x.size() - 1;
    vector<double> a(n), b(n), c(n), d(n), h(n);
    for (int i = 0; i < n; ++i) {
        h[i] = x[i + 1] - x[i];
    }
    matrix A(n - 1, n - 1), B(n - 1, 1);
    for (int i = 0; i < n - 1; ++i) {
        A[i][i] = 2 * (h[i] + h[i + 1]);
        if (i > 0)
            A[i][i - 1] = h[i];
        if (i < n - 2)
            A[i][i + 1] = h[i + 1];
        B[i][0] = 3 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i]);
    }
    matrix C = solve_SLE(A,B);
    for (int i = 0; i < n; ++i) {
        if (i == 0)
            c[i] = 0;
        else
            c[i] = C[i - 1][0];
    }
    for (int i = 0; i < n; ++i) {
        a[i] = y[i];
        if (i < n - 1) {
            b[i] = (y[i + 1] - y[i]) / h[i] - 1.0 / 3 * h[i] * (c[i + 1] + 2 * c[i]);
            d[i] = (c[i + 1] - c[i]) / (3 * h[i]);
        } else {
            b[i] = (y[i + 1] - y[i]) / h[i] - 2.0 / 3 * h[i] * c[i];
            d[i] = (-1) * c[i] / (3 * h[i]);
        }
    }
    auto it = lower_bound(x.begin(), x.end(), dot);
    int interval = it - x.begin() - 1;
    if (interval == -1)
        interval = 0;
    ofstream fout("answer.txt");
    fout<< "Cubic spline:\n";
    fout << "S(x) = " << a[interval] << " + " << b[interval] << " * (x - " << x[interval] << ") + " << c[interval] << " * (x - " << x[interval] << ")^2 + " << d[interval] << " * (x - " << x[interval] << ")^3\n";
    double res = a[interval] + b[interval] * (dot - x[interval]) + c[interval] * pow((dot - x[interval]), 2) + d[interval] * pow((dot - x[interval]), 3);
    fout << "f(x*) = " << res;
    return res;
}

int main() {
    vector<double> x = {1.0, 1.9, 2.8, 3.7, 4.6};
    vector<double> y = {2.8069, 1.8279, 1.6091, 1.5713, 1.5663};
    double X = 2.66666667;
    double res = spline(x, y, X);
    return 0;
}