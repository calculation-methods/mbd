#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

void LU(vector<vector<double>>& matrix, vector<vector<double>>& L, vector<vector<double>>& U, int n) {
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

vector<vector<double>> result(vector<vector<double>>& matrix_1, vector<vector<double>>& matrix_2, vector<vector<double>>& L, vector<vector<double>>& U, int n) {

    LU(matrix_1, L, U, n);
    vector<vector<double>> res = matrix_2;
    int m = res[0].size();
    for (int k=0; k<m; k++)
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                res[i][k] -= res[j][k]*L[i][j];
    for (int k=0; k<m; k++) {
        for (int i=n-1; i>-1; i--) {
            for (int j=i+1; j<n; j++) {
                res[i][k] -= res[j][k]*U[i][j];
            }
            res[i][k] /= U[i][i];
        }
    }
    return res;
}

double determinant(const vector<vector<double>>& U) {
    int n = U.size();
    double det = 1;
    for (int i = 0; i < n; ++i)
        det *= U[i][i];
    return det;
}

int main(){
    int n = 4;
    vector<vector<double>> matrix(n, vector <double>(n, 0)), b(n, vector <double>(n, 0));
    ifstream in1("matrix.txt"), in2("b.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++){
            in1 >> matrix[i][j];
        }
    }
    for (int i = 0; i < n; i++)
    {
        in2 >> b[i][0];
    }
    vector<vector<double>> L(n, vector<double>(0)), U(n, vector<double>(0)), E(n, vector<double>(0));
    for (int i = 0; i < n; i++){
        for (int j=0; j<n; j++){
            L[i].push_back(0);
            U[i].push_back(0);
            if (i == j){
                E[i].push_back(1);
            }else E[i].push_back(0);
        }
    }
    LU(matrix, L, U, n);
    double det = 1;
    det = determinant(U);
    vector<vector<double>> res = result(matrix, b, L, U, n);
    vector<vector<double>> invert = result(matrix, E, L, U, n);
    ofstream out("output.txt");
    out << "Решение системы: " << endl;
    for (int i = 0; i < n; ++i)
        out << res[i][0] << endl;
    out << endl << "Определитель матрицы: " << det << endl;
    out << endl << "Обратная матрица: " << endl;
    for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			out << invert[i][j] << "\t";
		}
		out << endl;
	}
    return 0;
}