#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <ccomplex>

using namespace std;

vector<vector<double>> transpose(vector<vector<double>>matrix) {
    int n = matrix.size();
    vector<vector<double>> result(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i][j] = matrix[j][i];
        }
    }
    return result;
}

double norma(vector <double> A){
    int n = A.size();
    double norm = 0;
    for (int i=0; i<n; i++){
        norm += A[i] * A[i];
    }

    return sqrt(norm);
}

double scalar_proisv(vector<double> first, vector<double> second){
    double sum = 0;
    int n = first.size();
    for (int i=0; i<n; i++){
        sum += first[i] * second[i];
    }
    return sum;
}


void normalize_matrix(vector<vector <double>>& A) {
    int n = A.size();
    for (int i = 0; i<n; i++){
        double norm = norma(A[i]);
        for (int j=0;j<n;j++){
            A[i][j] /= norm;
        }
    }
}

vector <double> substract_column(vector<double> first, vector<double> second){
    int n = first.size();
    vector <double> res(n,0);
    for (int i = 0; i<n; i++){
        res[i] = first[i] - second[i];
    }
    return res;
}

vector<vector <double>> gramSchmidt(vector<vector<double>> A) {
    int n = A.size();
    vector <vector <double>> transposed_matrix = transpose(A);
    vector <vector <double>> B(n, vector<double>(n,0));
    B[0] = transposed_matrix[0];
    for (int i=1; i<n; i++){
        vector <double> projection(n,0);
        for (int count=0; count<i; count++){
            double k = scalar_proisv(transposed_matrix[i], B[count]) / scalar_proisv(B[count], B[count]);
            for (int j = 0; j<n; j++){
                projection[j] += k * B[count][j];
            }
        }
        B[i] = substract_column(transposed_matrix[i], projection);
    }
    return B;
}

vector <vector <double>> proisv(vector <vector <double>> A, vector <vector <double>> B){
    int n = A.size();
    vector <vector <double>> R(n, vector<double>(n,0));
	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
			for(int k = 0; k < n; k++)
				R[i][j] += A[i][k] * B[k][j];
    return R;
}


void QR_decomp(vector<vector <double>> &A, vector<vector <double>> &Q, vector<vector <double>> &R){
    Q = gramSchmidt(A);
    normalize_matrix(Q);
    Q = transpose(Q);
    R = proisv(transpose(Q), A);
}

vector <complex<double>> sobs_values(vector<vector<double>> A, double eps)
{
    int n = A.size();
    vector <complex<double>> prev_sobs(n);
    vector<vector<double>> Q, R;
    vector <complex<double>> cur_sobs(n,0);
    while (true) {
        QR_decomp(A,Q,R);
        A = proisv(R,Q);
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && abs(A[i + 1][i]) > 1e-7) {
                double b = -(A[i][i] + A[i + 1][i + 1]);
                double c = A[i][i] * A[i + 1][i + 1] - A[i][i + 1] * A[i + 1][i];
                double discriminant = b * b - 4 * c;
                
                if (discriminant > 0) {
                    cur_sobs[i] = 0.5 * (-b - sqrt(discriminant));
                    cur_sobs[i + 1] = 0.5 * (-b + sqrt(discriminant));
                    i++;
                } else {
                    cur_sobs[i] = complex<double>(-b / 2, sqrt(-discriminant) / 2);
                    cur_sobs[i + 1] = complex<double>(-b / 2, -sqrt(-discriminant) / 2);
                    i++;
                }
            } else {
                cur_sobs[i] = A[i][i];
            }
        }
        bool ok = true;
        for (int i = 0; i < n; i++) {
            ok = ok && abs(cur_sobs[i] - prev_sobs[i]) < eps;
        }
        if (ok)
            break;
        prev_sobs = cur_sobs;
    }
    return prev_sobs;
}

int main() {
    int n = 3;
    ifstream fina("matrix5.txt");
    ofstream fout("answer5.txt");
    vector <vector <double>> A(n, vector <double>(n, 0));
    for (int i = 0; i < A.size(); i++){
        for (int j = 0; j < A.size(); j++)
            fina >> A[i][j];
    }
    vector<vector<double>> Q, R;
    QR_decomp(A,Q,R);

    fout << "Q matrix:" << endl;
    for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			fout << Q[i][j] << "\t";
		}
		fout << endl;
	}
    
    fout << "\nR matrix:" << endl;
    for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			fout << R[i][j] << "\t";
		}
		fout << endl;
	}
    double eps = 0.1;
    vector<complex<double>> sobstv = sobs_values(A, eps);
    fout << "\nSobsv_values:" << endl;
    for(int i = 0; i < n; i++){
        fout << sobstv[i] << "\t";
		fout << endl;
	}
    return 0;
}