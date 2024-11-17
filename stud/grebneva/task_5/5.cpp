#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

vector<vector<double>> transpose(const vector<vector<double>>& A) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<double>> AT(cols, vector<double>(rows, 0.0));
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

vector<vector<double>> multiply_m(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int colsB = B[0].size();
    vector<vector<double>> C(rowsA, vector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; i++) 
    {
        for (int j = 0; j < colsB; j++) 
        {
            for (int k = 0; k < colsA; k++) 
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

vector<vector<double>> multiply_s(const vector<vector<double>>& A, double scalar) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<double>> C(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            C[i][j] = A[i][j] * scalar;
        }
    }
    return C;
}

vector<vector<double>> division(const vector<vector<double>>& A, double B) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<double>> C(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            C[i][j] = A[i][j] / B;
        }
    }
    return C;
}

vector<vector<double>> difference(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<double>> C(rows, vector<double>(cols, 0.0));
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

double norm_vector(const vector<double>& b) {
    double norm = 0.0;
    for (int i = 0; i < b.size(); i++) 
    {
        norm += b[i] * b[i];
    }
    return sqrt(norm);
}

vector<vector<double>> Householder(const vector<vector<double>>& H, int index) {
    vector<vector<double>> H1 = H;
    int n = H1[0].size();
    vector<double> x1(n, 0.0);
    vector<double> b(n, 0.0);
    for (int k = 0; k < n; k++) 
    {
        b[k] = H1[k][index];
    }
    double norm = norm_vector(b);
    for (int i = 0; i < n; i++) 
    {
        if (i == index) {
            x1[i] = H1[i][index] + (H1[i][index] > 0 ? 1 : -1) * norm;
        } else if (i < index) {
            x1[i] = 0.0;
        } else {
            x1[i] = H1[i][index];
        }
    }
    vector<vector<double>> v1(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) 
    {
        v1[i][0] = x1[i];
    }
    vector<vector<double>> v1_T = transpose(v1);
    vector<vector<double>> E(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) 
    {
        E[i][i] = 1.0;
    }
    H1 = difference(E, multiply_s(division(multiply_m(v1, v1_T), multiply_m(v1_T, v1)[0][0]), 2.0));
    return H1;  
}

pair<vector<vector<double>>, vector<vector<double>>> QR(const vector<vector<double>>& A) {
    vector<vector<double>> A0 = A;
    vector<vector<vector<double>>> H;   
    int n = A.size();
    for (int i = 0; i < n - 1; i++) 
    {
        vector<vector<double>> H0 = Householder(A0, i);
        A0 = multiply_m(H0, A0);
        H.push_back(H0);
    }
    for (int i = 0; i < H.size() - 1; i++) 
    {
        H[i + 1] = multiply_m(H[i], H[i + 1]);
    }
    return make_pair(A0, H.back());
}

vector<double> eigenvalues(const vector<vector<double>>& A, const double e) {
    double m = 1.0;
    vector<vector<double>> A0 = A;
    while (m > e) 
    {
        vector<vector<double>> Q0 = QR(A0).second;
        vector<vector<double>> R0 = QR(A0).first;
        vector<double> a;
        A0 = multiply_m(R0, Q0);
        int n = A0.size();
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                if (i > j) a.push_back(A0[i][j]);
            }
        }
        m = norm_vector(a);
    }
    vector<double> L;
    int n = A0.size();
    for (int i = 0; i < n; i++) 
    {
        L.push_back(A0[i][i]);
    }
    return L;
}

int main() {
    vector<vector<double>> A(3, vector <double>(3, 0));
    double e;
    int n = A.size();

    ifstream fina("matrix.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fina >> A[i][j];
    }
    fina >> e;

    pair<vector<vector<double>>, vector<vector<double>>> QRresult = QR(A);
    vector<vector<double>> Q = QRresult.second;
    vector<vector<double>> R = QRresult.first;
    vector<double> w = eigenvalues(A, e);

    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;

    fout << "Матрица Q:" << endl;
    for (const auto& row : Q) 
    {
        for (const auto& elem : row)
            fout << fixed << elem << "\t";
        fout << endl;
    }
    fout << "Матрица R:" << endl;
    for (const auto& row : R) 
    {
        for (const auto& elem : row)
            fout << fixed << elem << "\t";
        fout << endl;
    }
    fout << "Собственные значения:" << endl;
    for (double val : w) {
        fout << val << " ";
    }
    fout << endl;

    return 0;
}