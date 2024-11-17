#include<iostream>
#include<conio.h>
#include <fstream>
#include <vector>

using namespace std;
const int n = 4;

istream& operator>>(istream& stream, float a[n][n])
{
    for(int i= 0;i<n;i++)
    {
        for(int j=0;j<n;j++)
            stream >> a[i][j];
    }
    return stream;
}
ostream& operator<<(ostream& stream, float a[n][n])
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            stream << a[i][j] << " ";
        stream << endl;
    }
    return stream;
}
istream& operator>>(istream& stream, float a[n])
{
    for(int i=0;i<n;i++)
        stream >> a[i];
    return stream;
}
ostream& operator<<(ostream& stream, float a[n])
{
    for (int i = 0; i < n; i++)
    {
            stream << a[i] << endl;
    }
    return stream;
}

float det_from_U(float u[n][n]) 
{
    double det = 1;
    for (int i = 0; i < n; i++)
        det *= u[i][i];
    return det;
}

void solve_eq_for_L(float l[n][n], const float b[n], float y[n]) {
	for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j)
            y[i] -= l[i][j] * y[j];
    }
}

void solve_eq_for_U(float u[n][n], float z[n], float x[n]){
	float E[n];
	for (int i = 0; i < n; ++i) {
        E[i] = 1;
    for (int i = n - 1; i >= 0; --i) {
        x[i] = z[i];
        for (int j = i + 1; j < n; ++j)
            x[i] -= u[i][j] * x[j];
        x[i] /= u[i][i];
        }
    }
}

void inverse(float const a[n][n], float l[n][n], float u[n][n], float inversed[n][n])
{
    float E[n] = {0};
    for (int i = 0; i < n; ++i) {
        E[i] = 1;
        float Y_E[n] = {0}, X_E[n] = {0};
        solve_eq_for_L(l, E, Y_E);
        solve_eq_for_U(u, Y_E, X_E);
        for (int j = 0; j < n; ++j)
            inversed[j][i] = X_E[j];
        E[i] = 0;
    }
}


void LU_Decomposite(float const a[n][n], float u[n][n], float l[n][n]) {
    
	for (int i = 0; i  < n; ++i) {
        for (int j = 0; j  < n; ++j) {
            u[i][j] = a[i][j];
        }
    }

	for(int i = 0; i < n; i++)
		for(int j = i; j < n; j++)
			l[j][i]=l[j][i]/u[i][i];
	
	for(int k = 1; k < n; k++)
	{
		for(int i = k-1; i < n; i++)
			for(int j = i; j < n; j++)
				l[j][i]=u[j][i]/u[i][i];

		for(int i = k; i < n; i++)
			for(int j = k-1; j < n; j++)
				u[i][j]=u[i][j]-l[i][k-1]*u[k-1][j];
	}
}

int main()
{
    int i,k,j,p;
    float a[n][n],l[n][n]={0},u[n][n]={0},sum,b[n],z[n]={0},x[n]={0};

    ofstream file_ans("answer.txt");
    file_ans.precision(2);
    file_ans << fixed;
    
    ifstream file_A("A_matrix.txt"), file_b("b_vector.txt");
    file_A >> a;
    file_b >> b;

    // LU decomposition
    LU_Decomposite(a, u ,l);
    // Displaying LU matrix
    file_ans<<"LU matrices: "<< endl;
    file_ans << "L:" << endl << l;
    file_ans<<"U:" << endl << u << endl;

    //Finding Z; Lz=b
    solve_eq_for_L(l,b, z);
    //Finding X; Ux=Z
    solve_eq_for_U(u,z,x);
    //Finding inversed matrix
    float inversed[n][n];
    inverse(a, l, u, inversed);
    //Solution
    file_ans << "Решение системы:\n" << x;
    file_ans << "\nОпределитель:" << det_from_U(u) << endl;
    file_ans << "\nОбратная матрица:\n" << inversed;
    return 0;
}