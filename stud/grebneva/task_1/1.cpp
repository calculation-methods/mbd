#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

vector<vector<double>> identity(int n) {
    vector<vector<double>> I(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i)
        I[i][i] = 1;
    return I;
}

void LU_decomposition(vector<vector<double>>& matrix, vector<int>& P, vector<vector<double>>& L, vector<vector<double>>& U) {
    int n = matrix.size();
    L = identity(n);
    U = matrix;

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

vector<double> forward_substitution(const vector<vector<double>>& L, const vector<double>& b) {
    int n = L.size();
    vector<double> y(n, 0);
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j)
            y[i] -= L[i][j] * y[j];
    }
    return y;
}

vector<double> backward_substitution(const vector<vector<double>>& U, const vector<double>& y) {
    int n = U.size();
    vector<double> x(n, 0);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }
    return x;
}

double calculate_determinant(const vector<vector<double>>& U) {
    int n = U.size();
    double det = 1;
    for (int i = 0; i < n; ++i)
        det *= U[i][i];
    return det;
}

vector<vector<double>> inverse_matrix(const vector<vector<double>>& L, const vector<vector<double>>& U) {
    int n = U.size();
    vector<vector<double>> inv_mat(n, vector<double>(n, 0));
    vector<double> E(n, 0);
    for (int i = 0; i < n; ++i) {
        E[i] = 1;
        vector<double> y = forward_substitution(L, E);
        vector<double> x = backward_substitution(U, y);
        for (int j = 0; j < n; ++j)
            inv_mat[j][i] = x[j];
        E[i] = 0;
    }
    return inv_mat;
}

vector<vector<double>> dot_product(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    vector<vector<double>> C(n, vector<double>(m, 0));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < p; ++k)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

int main() {
    vector<vector<double>> matrix(4, vector <double>(4, 0));

    vector<double> vec(4, 0);

     ifstream fina("matrix.txt"), finb("column.txt");
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix.size(); j++)
            fina >> matrix[i][j];
    }
    for (int i = 0; i < vec.size(); i++)
    {
        finb >> vec[i];
    }

    vector<int> P;
    vector<vector<double>> L, U;

    LU_decomposition(matrix, P, L, U);

    vector<double> Pb = vec;
    for (size_t i = 0; i < P.size(); ++i)
        Pb[i] = vec[P[i]];

    vector<double> y = forward_substitution(L, Pb);
    vector<double> x = backward_substitution(U, y);

    double det = calculate_determinant(U);

    vector<vector<double>> inv_matrix = inverse_matrix(L, U);


    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
   



    fout << "Решение системы:" << endl;
    for (size_t i = 0; i < x.size(); ++i)
        fout << "x[" << i << "] = " << x[i] << endl;

    fout << "Определитель матрицы: " << det << endl;
    fout << "Обратная матрица:" << endl;
    
    for (const auto& row : inv_matrix) {
        for (const auto& elem : row)
            fout << fixed << elem << "\t";
        fout << endl;
    }
    fout << "Матрица L:" << endl;
    for (const auto& row : L) {
        for (const auto& elem : row)
            fout << fixed << elem << "\t";
        fout << endl;
    }
    fout << "Матрица U:" << endl;
    for (const auto& row : U) {
        for (const auto& elem : row)
            fout << fixed << elem << "\t";
        fout << endl;
    }
    fout << "Матрица L*U:" << endl;
    for (const auto& row : dot_product(L, U)) {
        for (const auto& elem : row)
            fout << fixed << elem << "\t";
        fout << endl;
    }
    return 0;
}
