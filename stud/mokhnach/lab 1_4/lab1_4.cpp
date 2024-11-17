#include <iostream>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

double pi = acos(-1);
int n = 3;

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
    matrix get_transposed()
    {
        matrix result(cols, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                result[j][i] = obj[i][j];
        }
        return result;
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


matrix operator*(matrix a, matrix b)
{
    matrix result(a.rows, b.cols);
    for (int i = 0; i < result.rows; i++) 
    {
        for (int j = 0; j < result.cols; j++)
        {
            result[i][j] = 0;
            for (int k = 0; k < a.cols; k++)
                result[i][j] += a[i][k] * b[k][j];
        }
    }
    return result;
}

matrix get_rotation_matrix(matrix A, int i, int j) {
    
    matrix rotation(n, n);
    double phi = pi / 4;
    if (abs(A[i][i] - A[j][j]) > 0.0000001)
        phi = 0.5 * atan((2 * A[i][j]) / (A[i][i] - A[j][j]));

    for (int i = 0; i < n; i++)
        rotation[i][i] = 1;

    rotation[i][i] = cos(phi);
    rotation[i][j] = -sin(phi);
    rotation[j][j] = cos(phi);
    rotation[j][i] = sin(phi);
    return rotation;
}

int turns_method(matrix A, matrix& self_vec, matrix& self_value, double eps)
{
    int iterations_number = 0;
    double eps_k = 2 * eps;
    for (int i = 0; i < n; i++)
        self_vec[i][i] = 1;
    
    while (eps_k > eps)
    {
        int i_max = 1, j_max = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (abs(A[i_max][j_max]) < abs(A[i][j]))
                {
                    i_max = i;
                    j_max = j;
                }
            }
        }
        matrix rotation = get_rotation_matrix(A, i_max, j_max);

        self_vec = self_vec * rotation;
        A = rotation.get_transposed() * A * rotation;
        eps_k = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
                eps_k += A[i][j] * A[i][j];
        }
        eps_k = sqrt(eps_k);
        iterations_number++;
    }
    for (int i = 0; i < n; i++)
        self_value[i][0] = A[i][i];
    return iterations_number;
}

int main() 
{
    matrix A(n, n), self_vec(n,n), self_value(n,1);
    double eps;
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
    ifstream fin("input.txt");
    fin >> eps;
    fin >> A;
    int iterations_number = turns_method(A, self_vec, self_value, eps);
    fout << "Эпсилон:" << to_string(eps) << endl << endl;
    fout << "Собственные векторы:\n" << endl;
    for (int i = 0; i < n; i++) {
        fout << "СВ#" << i+1 << endl;
        for (int j = 0; j < n; j++) {
            fout << self_vec[j][i] << endl ;
        }
        fout << endl;
    }
    fout << "Собственные значения:\n" << self_value  << "\nИтераций: ";
    fout << iterations_number;
}