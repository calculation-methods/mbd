#include <iostream>
#include <vector>
#include <ccomplex>
#include <fstream>

using namespace std;

using cmd = complex <double>;
const double pi = acos(-1);

struct matrix
{
    int rows = 0, cols = 0;
    vector <vector <double>> v;

    matrix() {}
    matrix(int _rows, int _cols)
    {
        rows = _rows;
        cols = _cols;
        v = vector <vector <double>>(rows, vector <double>(cols));
    }

    vector <double>& operator[](int row)
    {
        return v[row];
    }

    operator double()
    {
        return v[0][0];
    }
};

istream& operator>>(istream& stream, matrix& a)
{
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
            stream >> a[i][j];
    }
    return stream;
}

ostream& operator<<(ostream& stream, matrix a)
{
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
            stream << a[i][j] << ' ';
        stream << '\n';
    }
    return stream;
}

matrix operator+(matrix lhs, matrix rhs)
{
    if (lhs.rows != rhs.rows || rhs.cols != lhs.cols)
        return matrix(0, 0);
    matrix res(lhs.rows, lhs.cols);
    for (int i = 0; i < rhs.rows; i++)
    {
        for (int j = 0; j < res.cols; j++)
            res[i][j] = lhs[i][j] + rhs[i][j];
    }
    return res;
}

matrix operator-(matrix lhs, matrix rhs)
{
    if (lhs.rows != rhs.rows || rhs.cols != lhs.cols)
        return matrix(0, 0);
    matrix res(lhs.rows, lhs.cols);
    for (int i = 0; i < rhs.rows; i++)
    {
        for (int j = 0; j < res.cols; j++)
            res[i][j] = lhs[i][j] - rhs[i][j];
    }
    return res;
}

matrix operator*(double lhs, matrix rhs)
{
    for (int i = 0; i < rhs.rows; i++)
    {
        for (int j = 0; j < rhs.cols; j++)
            rhs[i][j] *= lhs;
    }
    return rhs;
}

matrix operator*(matrix lhs, matrix rhs)
{
    if (lhs.cols != rhs.rows)
        return matrix(0, 0);
    matrix res(lhs.rows, rhs.cols);
    for (int i = 0; i < res.rows; i++) 
    {
        for (int j = 0; j < res.cols; j++)
        {
            res[i][j] = 0;
            for (int k = 0; k < lhs.cols; k++)
                res[i][j] += lhs[i][k] * rhs[k][j];
        }
    }
    return res;
}

matrix transposition(matrix a)
{
    matrix res(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < a.cols; j++)
            res[j][i] = a[i][j];
    }
    return res;
}

vector <int> swp;

pair <matrix, matrix> lu_decomposition(matrix a)
{
    int n = a.rows;
    matrix l(n, n);
    swp = vector <int>(0);
    for (int k = 0; k < n; k++)
    {
        matrix prev = a;
        int idx = k;
        for (int i = k + 1; i < n; i++)
        {
            if (abs(prev[idx][k]) < abs(prev[i][k]))
                idx = i;
        }
        swap(prev[k], prev[idx]);
        swap(a[k], a[idx]);
        swap(l[k], l[idx]);
        swp.push_back(idx);
        for (int i = k + 1; i < n; i++)
        {
            double h = prev[i][k] / prev[k][k];
            l[i][k] = h;
            for (int j = k; j < n; j++)
                a[i][j] = prev[i][j] - h * prev[k][j];

        }
    }
    for (int i = 0; i < n; i++)
        l[i][i] = 1;
    return { l, a };
}

matrix solve_triag(matrix a, matrix b, bool up)
{
    int n = a.rows;
    matrix res(n, 1);
    int d = up ? -1 : 1;
    int first = up ? n - 1 : 0;
    for (int i = first; i < n && i >= 0; i += d)
    {
        res[i][0] = b[i][0];
        for (int j = 0; j < n; j++)
        {
            if (i != j)
                res[i][0] -= a[i][j] * res[j][0];
        }
        res[i][0] = res[i][0] / a[i][i];
    }
    return res;
}

matrix solve_gauss(pair <matrix, matrix> lu, matrix b)
{
    for (int i = 0; i < swp.size(); i++)
        swap(b[i], b[swp[i]]);
    matrix z = solve_triag(lu.first, b, false);
    matrix x = solve_triag(lu.second, z, true);
    return x;
}

matrix inverse(matrix a)
{
    int n = a.rows;
    matrix b(n, 1);
    pair <matrix, matrix> lu = lu_decomposition(a);
    matrix res(n, n);
    for (int i = 0; i < n; i++)
    {
        b[max(i - 1, 0)][0] = 0;
        b[i][0] = 1;
        matrix col = solve_gauss(lu, b);
        for (int j = 0; j < n; j++)
            res[j][i] = col[j][0];
    }
    return res;
}

double abs(matrix a)
{
    double mx = 0;
    for (int i = 0; i < a.rows; i++)
    {
        double s = 0;
        for (int j = 0; j < a.cols; j++)
            s += abs(a[i][j]);
        mx = max(mx, s);
    }
    return mx;
}

double sign(double x)
{
    return x > 0 ? 1 : -1;
}

pair <matrix, matrix> qr_decomposition(matrix a)
{
    int n = a.rows;
    matrix e(n, n);
    for (int i = 0; i < n; i++)
        e[i][i] = 1;
    matrix q = e;
    for (int i = 0; i < n - 1; i++)
    {
        matrix v(n, 1);
        double s = 0;
        for (int j = i; j < n; j++)
            s += a[j][i] * a[j][i];
        v[i][0] = a[i][i] + sign(a[i][i]) * sqrt(s);
        for (int j = i + 1; j < n; j++)
            v[j][0] = a[j][i];
        matrix h = e - (2.0 / double(transposition(v) * v)) * (v * transposition(v));
        q = q * h;
        a = h * a;
    }
    return { q, a };
}

vector <cmd> qr_eigenvalues(matrix a, double eps)
{
    int n = a.rows;
    vector <cmd> prev(n);
    while (true)
    {
        pair <matrix, matrix> p = qr_decomposition(a);
        a = p.second * p.first;
        vector <cmd> cur;
        for (int i = 0; i < n; i++)
        {
            if (i < n - 1 && abs(a[i + 1][i]) > 1e-7)
            {
                double b = -(a[i][i] + a[i + 1][i + 1]);
                double c = a[i][i] * a[i + 1][i + 1] - a[i][i + 1] * a[i + 1][i];
                double d = b * b - 4 * c;
                cmd sgn = (d > 0) ? cmd(1, 0) : cmd(0, 1);
                d = sqrt(abs(d));
                cur.push_back(0.5 * (-b - sgn * d));
                cur.push_back( 0.5 * (-b + sgn * d));
                i++;
            }
            else
                cur.push_back(a[i][i]);
        }
        bool ok = true;
        for (int i = 0; i < n; i++)
            ok = ok && abs(cur[i] - prev[i]) < eps;
        if (ok)
            break;
        prev = cur;
    }
    return prev;
}

int main() 
{   
    ofstream fout("answer.txt");
    fout.precision(5);
    fout << fixed;
    ifstream fin("matrix5.txt");
    matrix a(3, 3);
    fin >> a;
    pair <matrix, matrix> p = qr_decomposition(a);
    fout << "QR разложение:\nQ =\n" << p.first << "\nR =\n" << p.second;
    cout << p.first * p.second << '\n' << transposition(p.first) << '\n' << inverse(p.first);
    vector <cmd> v = qr_eigenvalues(a, 0.01);
    fout << "\nСобственные значения QR методом с e=0.1:\n";
    for (int i = 0; i < v.size(); i++)
        fout << v[i] << ' ';
}
