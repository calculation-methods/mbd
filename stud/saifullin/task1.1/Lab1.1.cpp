#include <iostream>
#include <vector>
#include <fstream>

using namespace std;


void LU(vector <vector <double>> A, vector <vector <double>> &L,
        vector <vector <double>> &U, int n) {
    U = A;

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

void show(vector <vector <double>> A, int n)
{
	ofstream fout("answer.txt");
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			fout << A[i][j] << "\t";
		}
		fout << endl;
	}
}

double determinant(vector <vector <double>> U, int n)
{
	double det = 1;
	for (int i=0; i<n; i++){
		det *= U[i][i];
	}
	return det;
}


vector <double> solve_L_matrix(const vector <vector <double>> &L, const vector <double> &b, int n) {
	vector <double> y(n, 0);
	for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j)
            y[i] -= L[i][j] * y[j];
    }
    return y;
}

vector <double> solve_U_matrix(const vector <vector <double>> &U, const vector <double> &y, int n){
	vector<double> x(n, 0);
	vector<double> E(n, 0);
	for (int i = 0; i < n; ++i) {
        E[i] = 1;
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }
    return x;
}
}

vector <double> solve_SLAU(vector <vector <double>> L, vector <vector <double>> U, vector <double> b, int n)
{
	vector<double> y = solve_L_matrix(L, b, n);
	vector<double> x = solve_U_matrix(U, y, n);
	return x;
}

vector <vector <double>> inverse_matrix(vector <vector <double>> &A, vector <vector <double>> &L, vector <vector <double>> &U, int n){
	vector<vector<double>> inverse_A(n, vector<double>(n, 0));
    vector<double> E(n, 0);
    for (int i = 0; i < n; ++i) {
        E[i] = 1;
        vector<double> y = solve_L_matrix(L, E, n);
        vector<double> x = solve_U_matrix(U, y, n);
        for (int j = 0; j < n; ++j)
            inverse_A[j][i] = x[j];
        E[i] = 0;
    }
    return inverse_A;
}


int main(){
	const int n=4;
	vector <vector <double>> A(n, vector <double>(n, 0));
	vector <double> b(n,0);
	ifstream fina("matrix.txt"), finb("column.txt");
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A.size(); j++)
            fina >> A[i][j];
    }
    for (int i = 0; i < b.size(); i++)
    {
        finb >> b[i];
    }

	vector<vector<double>> L(n, vector<double>(0)), U(n, vector<double>(0)), RES(n, vector<double>(0));
    for (int i = 0; i < n; i++){
        for (int j=0; j<n; j++){
            L[i].push_back(0);
            U[i].push_back(0);
            RES[i].push_back(0);
        }
    }

	ofstream fout("answer.txt");

	fout.precision(2);
	fout << fixed;
    LU(A,L,U,n);
	fout << "Original matrix" << endl;
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			fout << A[i][j] << "\t";
		}
		fout << endl;
	}
	fout << "U matrix" << endl;
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			fout << U[i][j] << "\t";
		}
		fout << endl;
	}
	fout << "L matrix" << endl;
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			fout << L[i][j] << "\t";
		}
		fout << endl;
	}
	proisv(L,U,RES,n);
	fout << "L*U matrix" << endl;
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			fout << RES[i][j] << "\t";
		}
		fout << endl;
	}
	fout << "Determinant:" << determinant(U, n) << endl;
	fout << "Inversed matrix:" << endl;
	vector <vector <double>> inversed = inverse_matrix(A, L, U, n);
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			fout << inversed[i][j] << "\t";
		}
		fout << endl;
	}
	fout << "Solution SLAU:" << endl;
	vector <double> x = solve_SLAU(L,U,b,n);
	for (size_t i = 0; i < x.size(); ++i)
        fout << "x[" << i << "] = " << x[i] << endl;
	return 0;
}
