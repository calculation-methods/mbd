#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace std;

void LU(vector <vector <double>> A, vector <vector <double>> &L, vector <vector <double>> &U, int n)
{
    U=A;
    for(int i = 0; i < n; i++)
        for(int j = i; j < n; j++)
            L[j][i]=U[j][i]/U[i][i];

    for(int k = 1; k < n; k++)
    {
        for(int i = k-1; i < n; i++)
            for(int j = i; j < n; j++)
                L[j][i]=U[j][i]/U[i][i];

        for(int i = k; i < n; i++)
            for(int j = k-1; j < n; j++)
                U[i][j]=U[i][j]-L[i][k-1]*U[k-1][j];
    }
}

void proisv(vector <vector <double>> A, vector <vector <double>> B,
            vector <vector <double>> &R, int n)
{
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                R[i][j] += A[i][k] * B[k][j];
}

void print_matrix(vector <vector <double>> A, int n, ofstream& fout)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            fout <<"\t"<< fixed << setprecision(2) << A[i][j] << "\t";
        }
        fout << endl;
    }
}

vector<double> first_slau(vector<vector<double>>& L, vector<double>& b, int n) {
    vector<double> z(n, 0);
    for (int i = 0; i < n; i++) {
        z[i] = b[i];
        for (int j = 0; j < i; j++)
            z[i] -= L[i][j] * z[j];
    }
    return z;
}

vector<double> second_slau(vector<vector<double>>& U, vector<double>& z, int n) {
    vector<double> x(n, 0);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = z[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }
    return x;
}

vector<vector<double>> getMinor(vector<vector<double>>& matrix, int row, int col) {
    vector<vector<double>> minor(matrix.size() - 1, vector<double>(matrix.size() - 1));
    for (int i = 0, r = 0; i < matrix.size(); ++i) {
        if (i == row) continue;
        for (int j = 0, c = 0; j < matrix.size(); ++j) {
            if (j == col) continue;
            minor[r][c++] = matrix[i][j];
        }
        ++r;
    }
    return minor;
}

int determinant(vector<vector<double>>& matrix) {
    int size = matrix.size();
    if (size == 1) return matrix[0][0];
    if (size == 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    int det = 0;
    for (int i = 0; i < size; ++i) {
        vector<vector<double>> minor = getMinor(matrix, 0, i);
        int sign = (i % 2 == 0) ? 1 : -1;
        det += sign * matrix[0][i] * determinant(minor);
    }
    return det;
}


vector<vector<double>> inverseMatrix(vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<vector<double>> augmentedMatrix(n, vector<double>(2 * n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmentedMatrix[i][j] = matrix[i][j];
        }
        augmentedMatrix[i][i + n] = 1;
    }
    for (int i = 0; i < n; ++i) {
        if (augmentedMatrix[i][i] == 0) {
            for (int j = i + 1; j < n; ++j) {
                if (augmentedMatrix[j][i] != 0) {
                    swap(augmentedMatrix[i], augmentedMatrix[j]);
                    break;
                }
            }
        }
        double divisor = augmentedMatrix[i][i];
        for (int j = i; j < 2 * n; ++j) {
            augmentedMatrix[i][j] /= divisor;
        }
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                double factor = augmentedMatrix[j][i];
                for (int k = i; k < 2 * n; ++k) {
                    augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k];
                }
            }
        }
    }
    vector<vector<double>> inverse(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inverse[i][j] = augmentedMatrix[i][j + n];
        }
    }
    return inverse;
}

int main()
{
    ofstream fout;
    fout.open("output.txt");
    int n = 4;
    vector <vector <double>> A (n,vector <double>(n, 0)), L (n,vector <double>(n, 0)), U(n,vector <double>(n, 0)), R(n,vector <double>(n, 0));
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            L[i].push_back(0);
            U[i].push_back(0);
            R[i].push_back(0);
        }
    }
    A = {
            {8, 8,  -5, -8},
            {8, -5, 9,  -8 },
            {5,  -4,  -6, -2},
            {8,  3, 6, 6}
    };
    vector<double> b = {13, 38, 14, -95};
    LU(A,L,U,n);
    vector <double> z1 = first_slau(L, b, n);
    vector<double> x1 = second_slau(U, z1, n);
    int det = determinant(A);
    fout << "Определитель = " << det << endl;
    vector<vector<double>> invMatrix = inverseMatrix(A);
    fout << "Обратная матрица:" << endl;
    print_matrix(invMatrix, n, fout);
    fout << "Вектор решений" << endl;
    for (int i = 0; i < n; i++){
        fout << x1[i] << endl;
    }
    fout << "Матрица U" << endl;
    print_matrix(U,n, fout);
    fout << "Матрица L" << endl;
    print_matrix(L,n, fout);
    proisv(L,U,R,n);
    fout << "Матрица L*U" << endl;
    print_matrix(R,n, fout);
    return 0;
}