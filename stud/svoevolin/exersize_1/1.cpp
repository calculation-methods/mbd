#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

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

double determinant(matrix a) 
{
    int n = a.rows;
    pair <matrix, matrix> lu = lu_decomposition(a);
    double det = 1;
    for (int i = 0; i < n; i++)
        det *= lu.second[i][i];
    return det;
}

int main() 
{
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
    ifstream fina("matrix1.txt"), finb("column1.txt");
    matrix a(4, 4), b(4, 1);
    fina >> a;
    finb >> b;
    pair <matrix, matrix> p = lu_decomposition(a);
    fout << "LU разложение:\nL =\n" << p.first << "\nU =\n" << p.second;
    fout << "\nLU=\n" << p.first * p.second;
    fout << "\nРешение системы:\n" << solve_gauss(p, b);
    fout << "\nОбратная матрица:\n" << inverse(a);
    fout << "\nОбратная на исходную:\n" << a*inverse(a);
    fout << "\nОпределитель: " << determinant(a) << "\n"; 
}
