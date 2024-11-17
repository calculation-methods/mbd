#include <iostream>
#include <vector>
#include <fstream>
#include <string> 

using namespace std;

int n = 4;
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

        operator double()
        {
            return obj[0][0];
        }

        vector <double>& operator[](int i)
        {
            return obj[i];
        }
    double get_abs()
    {
        double mx = 0;
        for (int i = 0; i < rows; i++)
        {
            double s = 0;
            for (int j = 0; j < cols; j++)
                s += std::abs(obj[i][j]);
            mx = max(mx, s);
        }
        return mx;
    }
};


//****Определение операторов****//
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

matrix operator*(double a, matrix b)
{
    for (int i = 0; i < b.rows; i++)
    {
        for (int j = 0; j < b.cols; j++)
            b[i][j] *= a;
    }
    return b;
}

matrix operator+(matrix a, matrix b)
{
    if (a.rows != b.rows || b.cols != a.cols)
        return matrix(0, 0);
    matrix res(a.rows, a.cols);
    for (int i = 0; i < b.rows; i++)
    {
        for (int j = 0; j < res.cols; j++)
            res[i][j] = a[i][j] + b[i][j];
    }
    return res;
}

matrix operator-(matrix a, matrix b)
{
    if (a.rows != b.rows || b.cols != a.cols)
        return matrix(0, 0);
    matrix res(a.rows, a.cols);
    for (int i = 0; i < b.rows; i++)
    {
        for (int j = 0; j < res.cols; j++)
            res[i][j] = a[i][j] - b[i][j];
    }
    return res;
}

matrix operator*(matrix a, matrix b)
{
    if (a.cols != b.rows)
        return matrix(0, 0);
    matrix res(a.rows, b.cols);
    for (int i = 0; i < res.rows; i++) 
    {
        for (int j = 0; j < res.cols; j++)
        {
            res[i][j] = 0;
            for (int k = 0; k < a.cols; k++)
                res[i][j] += a[i][k] * b[k][j];
        }
    }
    return res;
}

//****Решение****//
void set_alpha_beta(matrix& alpha, matrix& beta, matrix a, matrix b) {
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            alpha[i][j] = -a[i][j] / a[i][i];
        alpha[i][i] = 0;
    }
    for (int i = 0; i < n; i++)
        beta[i][0] = b[i][0] / a[i][i];
}
matrix solve_SLE_iterations(matrix a, matrix b, double eps, int& iterations_number)
{
    matrix alpha(n, n), beta(n, 1);
    set_alpha_beta(alpha, beta, a, b);
    
    matrix ans = beta, diff;
    double m = abs(a);
    double cur = m;
    double eps_k = 2 * eps;
    iterations_number = 0;

    while (eps_k > eps)
    {
        matrix prev = ans;
        ans = beta + alpha * ans;
        diff = ans - prev;
        if (m < 1)
            eps_k = cur / (1 - m) * diff.get_abs();
        else
            eps_k = diff.get_abs();
        cur = cur * m;
        iterations_number++;
    }
    return ans;
}

matrix solve_SLE_seidel(matrix a, matrix b, double eps, int& iterations_number)
{
    matrix alpha(n, n), beta(n, 1);
    set_alpha_beta(alpha, beta, a, b);
    
    matrix ans = beta, diff;
    double m = abs(a);
    double cur = m;
    double eps_k = 2 * eps;
    iterations_number = 0;
    while (eps_k > eps)
    {
        matrix prev = ans;
        for (int i = 0; i < n; i++)
        {
            double cur = beta[i][0];
            for (int j = 0; j < n; j++)
                cur += alpha[i][j] * ans[j][0];
            ans[i][0] = cur;
        }
        diff = ans - prev;
        if (m < 1)
            eps_k = cur / (1 - m) * diff.get_abs();
        else
            eps_k = diff.get_abs();
        cur = cur * m;
        iterations_number++;
    }
    return ans;
}

int main() 
{
    matrix A(n, n), b(n, 1);
    float eps;
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
    ifstream fin_A("A_matrix.txt"), fin_b("b_column.txt"),fin_eps("accuracy.txt");
    fin_A >> A;
    fin_b >> b;
    fin_eps >> eps;

    int iterations_number = 0;
    fout << "Эпсилон: " << to_string(eps) << endl;
    fout << "Решение методом итераций:\n" << solve_SLE_iterations(A, b, eps, iterations_number) << "Итераций: ";
    fout << iterations_number;
    fout << "\n" << "Решение методом Зейделя:\n" << solve_SLE_seidel(A, b, eps, iterations_number) << "Итераций: ";
    fout << iterations_number;
}