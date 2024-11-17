#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

vector <double> progon_method(vector <vector <double>> A, vector <double> b, int n){
    // Прямой ход
    double y;
    vector <double> alph(n,0), Bett(n,0), x(n,0);
    y = A[0][0];
    alph[0] = -A[0][1] / y;
    Bett[0] = b[0] / y;
    for (int i = 1; i < n; i++) {
        y = A[i][i] + A[i][i - 1] * alph[i - 1];
        alph[i] = -A[i][i + 1] / y;
        Bett[i] = (b[i] - A[i][i - 1] * Bett[i - 1]) / y;
    }

    // Обратный ход
    for (int i = n - 1; i >= 0; i--) {
        x[i] = alph[i] * x[i + 1] + Bett[i];
    }

    return x;
}

int main(){
	const int n=5;
	vector <vector <double>> A(n, vector <double>(n, 0));
	vector <double> b(n,0);
	ifstream fina("matrix2.txt"), finb("column2.txt");
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A.size(); j++)
            fina >> A[i][j];
    }
    for (int i = 0; i < b.size(); i++)
    {
        finb >> b[i];
    }

    ofstream fout("answer2.txt");
    vector <double> x = progon_method(A, b, n);
    for (size_t i = 0; i < x.size(); ++i)
        fout << "x[" << i << "] = " << x[i] << endl;
	
    return 0;
}