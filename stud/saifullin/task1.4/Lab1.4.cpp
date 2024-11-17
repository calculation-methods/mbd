#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

vector <vector <double>> proisv(vector <vector <double>> A, vector <vector <double>> B, int n){
    vector <vector <double>> R(n, vector <double>(n,0));
	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
			for(int k = 0; k < n; k++)
				R[i][j] += A[i][k] * B[k][j];
    return R;
}


vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<vector<double>> result(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i][j] = matrix[j][i];
        }
    }

    return result;
}

void jacobiAlgorithm(vector<vector<double>>& A, vector<double>& sobs_values, vector<vector<double>>& sobs_vectors) {
    int n = A.size();
    sobs_vectors = vector<vector<double>>(n, vector<double>(n, 0));
    for (int i = 0; i < n; ++i) sobs_vectors[i][i] = 1;
    
    vector<vector<double>> B = A;
    double eps = 0.1;
    while (true) {
        double maxVal = 0;
        int p, q;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (fabs(B[i][j]) > maxVal) {
                    maxVal = fabs(B[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (maxVal < eps) break;
        double pi = acos(-1);
        double theta = pi / 4;
        if (abs(B[q][q] - B[p][p]) > 1e-7)
            theta = 0.5 * atan((2 * B[q][p]) / (B[q][q] - B[p][p]));
        double sinTheta = sin(theta);
        double cosTheta = cos(theta);

        vector<vector<double>> rotationMatrix(n, vector<double>(n, 0));
        for (int i = 0; i < n; ++i) rotationMatrix[i][i] = 1;
        rotationMatrix[p][p] = cosTheta;
        rotationMatrix[p][q] = sinTheta;
        rotationMatrix[q][p] = -sinTheta;
        rotationMatrix[q][q] = cosTheta;

        vector <vector <double>> res = proisv(transpose(rotationMatrix), B, n);
        res = proisv(res, rotationMatrix, n);
        B = res;
        sobs_vectors = proisv(sobs_vectors, rotationMatrix, n);
    }

    sobs_values.resize(n);
    for (int i = 0; i < n; ++i) {
        sobs_values[i] = B[i][i];
    }
}

int main() {
    int n = 3;
    ifstream fina("matrix4.txt");
    ofstream fout("answer4.txt");
    vector <vector <double>> A(n, vector <double>(n, 0));
    for (int i = 0; i < A.size(); i++)    {
        for (int j = 0; j < A.size(); j++)
            fina >> A[i][j];
    }
    vector<double> sobs_values;
    vector<vector<double>> sobs_vectors;

    jacobiAlgorithm(A, sobs_values, sobs_vectors);
    fout << "eps = 0.1\n" << endl;
    fout << "Sobstv values:" << endl;
    for (size_t i = 0; i < sobs_values.size(); ++i)
        fout << sobs_values[i] << endl;
    fout << endl;
    fout << "Sobstv vectors:" << endl;
    for(int i = 0; i < n; i++)	{
		for(int j = 0; j < n; j++){
			fout << sobs_vectors[i][j] << "\t";
		}
		fout << endl;
	}

    return 0;
}