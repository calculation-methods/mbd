#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int n = 5;
class matrix
{
    private:
        vector <vector <double>> obj;
    public:
        int  cols = 0, rows = 0;

        matrix() {}
        matrix(int _rows, int _cols)
        {
            rows = _rows;
            cols = _cols;
            obj = vector <vector <double>>(rows, vector <double>(cols));
        }

        vector <double>& operator[](int i)
        {
            return obj[i];
        }

        operator double()
        {
            return obj[0][0];
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

int main() 
{
    matrix A(n, n), b(n, 1);
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
    ifstream fin_A("A_matrix.txt"), fin_b("b_column.txt");
    fin_A >> A;
    fin_b >> b;
    //Finding an answer
    matrix ans;
    ans = solve_SLE(A, b);
    fout << "Ответ:\n" << ans;
}