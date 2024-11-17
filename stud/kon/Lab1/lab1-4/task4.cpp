#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

const double eps = 1e-4;

using namespace std;

void rotation_method(vector<vector<double>> matrix, vector<vector<double>> U, vector<vector<double>> &V, vector<double> &w, int n, int &k){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            if (i == j) V[i][j] = 1;
            else V[i][j] = 0;
        }

    double f = 0.0;
    vector<vector<double>> B(n, vector <double>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            f = f + matrix[i][j]*matrix[i][j];
    f = sqrt(f);

    double phi = 0.0;

    while (f > eps)
    {
        int p = 1, q = 2;
        double g = 0.0;       

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (matrix[i][j] > g){
                    g = matrix[i][j];
                    p = i;
                    q = j;
                }
                if (-matrix[i][j] > g){
                    g = -matrix[i][j];
                    p = i;
                    q = j;
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) U[i][j] = 1;
                else U[i][j] = 0;
            }
        }
        if (matrix[p][p] == matrix[q][q]) phi = M_PI/4;
        else phi = atan(2 * matrix[p][q]/(matrix[p][p] - matrix[q][q]))/2;

        U[p][p] = cos(phi);
        U[p][q] = -sin(phi);
        U[q][p] = sin(phi);
        U[q][q] = cos(phi);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((i == p) || (i == q) || (j == p) || (j == q)) 
                {
                    B[i][j] = 0;
                    for (int p = 0; p < n; p++)
                        B[i][j] = B[i][j] + U[p][i]*matrix[p][j];
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((i == p) || (i == q) || (j == p) || (j == q)) 
                {
                    matrix[i][j] = 0;
                    for (int p = 0; p < n; p++)
                        matrix[i][j] = matrix[i][j] + B[i][p]*U[p][j];
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((i == p) || (i == q) || (j == p) || (j == q)) 
                {
                    B[i][j] = 0;
                    for (int p = 0; p < n; p++)
                        B[i][j] = B[i][j] + V[i][p]*U[p][j];
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((i == p) || (i == q) || (j == p) || (j == q)) 
                    V[i][j] = B[i][j];                   
            }
        }

        f = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                f = f + matrix[i][j]*matrix[i][j];

        f = sqrt(f);
        ++k;
    }

    for (int i = 0; i < n; i++)
    {
        w[i] = matrix[i][i];
    } 
}

int main() {
    int n = 3;
    vector<vector<double>> matrix(n, vector <double>(n, 0)), Vectors(n, vector <double>(n, 0)), U(n, vector <double>(n, 0));
    vector<double> w(n, 0);
    int k = 0;
    ifstream in("matrix.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            in >> matrix[i][j];
    }

    rotation_method(matrix, U, Vectors, w, n, k);

    ofstream out("output.txt");

    out << "Собственные значения:" << endl;
    for (size_t i = 0; i < w.size(); ++i)
    {
        out << w[i] << endl;
    }

    out << "Собственные векторы:" << endl;
    for (const auto& row : Vectors) 
    {
        for (const auto& elem : row)
            out << elem << "\t";
        out << endl;
    }

    out << "Количество итераций: " << k << endl;

    return 0;
}