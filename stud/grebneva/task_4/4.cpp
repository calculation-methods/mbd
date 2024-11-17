#include <iostream>
#include <vector>
#include <fstream>

#define M_PI 3.14159265358979323846

using namespace std;

void rotation_method(vector<vector<double>> matrix, vector<vector<double>> U, vector<vector<double>> &V, 
                        vector<double> &w, const int n, const double e, int &k){

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            if (i == j) V[i][j] = 1;
            else V[i][j] = 0;
        }

    double f = 0.0;

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            f = f + matrix[i][j]*matrix[i][j];
    f = sqrt(f);

    double h = 0.0;
    vector<vector<double>> B(3, vector <double>(3, 0));

    while (f > e)
    {
        int l = 1, m = 2;
        double g = 0.0;       
        
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (matrix[i][j] > g){
                    g = matrix[i][j];
                    l = i;
                    m = j;
                }
                if (-matrix[i][j] > g){
                    g = -matrix[i][j];
                    l = i;
                    m = j;
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
        if (matrix[l][l] == matrix[m][m]) h = M_PI/4;
        else h = atan(2 * matrix[l][m]/(matrix[l][l] - matrix[m][m]))/2;

        U[l][l] = cos(h);
        U[l][m] = -sin(h);
        U[m][l] = sin(h);
        U[m][m] = cos(h);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if ((i == l) || (i == m) || (j == l) || (j == m)) 
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
                if ((i == l) || (i == m) || (j == l) || (j == m)) 
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
                if ((i == l) || (i == m) || (j == l) || (j == m)) 
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
                if ((i == l) || (i == m) || (j == l) || (j == m)) 
                    V[i][j] = B[i][j];                   
            }
        }

        f = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                f = f + matrix[i][j]*matrix[i][j];
                
        f = sqrt(f);
        k = k + 1;
    }

    for (int i = 0; i < n; i++)
    {
        w[i] = matrix[i][i];
    } 
}

int main() {
    vector<vector<double>> matrix(3, vector <double>(3, 0)), V(3, vector <double>(3, 0)), U(3, vector <double>(3, 0));
    double e;
    vector<double> w(3, 0);
    int n = matrix.size();

    ifstream fina("matrix.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fina >> matrix[i][j];
    }
    fina >> e; 

    int k = 0;

    rotation_method(matrix, U, V, w, n, e, k);

    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;

    fout << "Собственные значения:" << endl;
    for (size_t i = 0; i < w.size(); ++i)
    {
        fout << "w[" << i << "] = " << w[i] << endl;
    }

    fout << "Собственные векторы:" << endl;
    for (const auto& row : V) 
    {
        for (const auto& elem : row)
            fout << fixed << elem << "\t";
        fout << endl;
    }

    fout << "Количество итераций: " << k << endl;
    
    return 0;
}