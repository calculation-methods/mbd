#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

vector <double> method_iteration(vector <vector <double>> A, vector <double> b, int n, int &counter){

    vector <double> x(n,0);
    vector <double> xn(n,0);
    double eps = 0.001;
    for (int i = 0; i < n; i++) {
		x[i] = b[i] / A[i][i];
	}
    do {
		for (int i = 0; i < n; i++) {
			xn[i] = b[i] / A[i][i];
			for (int j = 0; j < n; j++) {
				if (i == j)
					continue;
				else {
					xn[i] -= A[i][j] / A[i][i] * x[j];
				}
			}
		}

		bool flag = true;
		for (int i = 0; i < n - 1; i++) {
			if (fabs(xn[i] - x[i]) > eps) {
				flag = false;
				break;
			}
		}

		for (int i = 0; i < n; i++) {
			x[i] = xn[i];
		}

		if (flag)
			break;
        counter++;
	} while (1);
    return x;
}

vector <double> method_zeidel(vector <vector <double>> A, vector <double> b, int n, int &counter){
    vector<double> x(n, 0);
    vector<double> x_prev(n, 0);
    double eps = 0.001;
    double diff;
    int max_iter = 1000;


    do {
        x_prev = x;
        for (int i = 0; i < n; i++) {
            double sum1 = 0;
            double sum2 = 0;
            for (int j = 0; j < i; j++) {
                sum1 += A[i][j] * x[j];
            }
            for (int j = i + 1; j < n; j++) {
                sum2 += A[i][j] * x_prev[j];
            }
            x[i] = (b[i] - sum1 - sum2) / A[i][i];
        }

        diff = 0;
        for (int i = 0; i < n; i++) {
            diff += abs(x[i] - x_prev[i]);
        }

        counter++;
    } while (diff > eps);

    return x;
}

int main() {
    const int n=4;
	vector <vector <double>> A(n, vector <double>(n, 0));
	vector <double> b(n,0);
	ifstream fina("matrix3.txt"), finb("column3.txt");
    for (int i = 0; i < A.size(); i++)
    {
        for (int j = 0; j < A.size(); j++)
            fina >> A[i][j];
    }
    for (int i = 0; i < b.size(); i++)
    {
        finb >> b[i];
    }

    int counter = 0;
    vector <double> x = method_iteration(A, b, n, counter);
    ofstream fout("answer3.txt");
    fout << "eps = 0.001\n" << endl;
    fout << "Method Iteration" << endl; 
    fout << "Count of iterations:" << counter << endl;
    for (size_t i = 0; i < x.size(); ++i)
        fout << "x[" << i << "] = " << x[i] << endl;
    counter = 0;
    x = method_zeidel(A,b, n, counter);
    fout << "\nMethod Zeidel" << endl; 
    fout << "Count of iterations:" << counter << endl;
    for (size_t i = 0; i < x.size(); ++i)
        fout << "x[" << i << "] = " << x[i] << endl;

    return 0;
}