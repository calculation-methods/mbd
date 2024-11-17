#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

vector<double> solve_linear_system(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();
    
    
    for (int i = 0; i < n; i++) {
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(A[k][i]) > abs(A[max_row][i])) {
                max_row = k;
            }
        }
        swap(A[i], A[max_row]);
        swap(b[i], b[max_row]);
        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    return x;
}

pair<vector<double>, vector<double>> method_least_squares(vector<double> x, vector<double> y) {
    vector <vector <double>> A(2, vector<double>(3,0));
    double sumx = 0, sumy = 0, sumx2 = 0, sumxy = 0;
    for (int i = 0; i < x.size(); ++i) {
        sumx += x[i];
        sumy += y[i];
        sumx2 += pow(x[i], 2);
        sumxy += x[i] * y[i];
    }
    A[0][0] = x.size();
    A[0][1] = sumx;
    A[0][2] = sumy;
    A[1][0] = sumx;
    A[1][1] = sumx2;
    A[1][2] = sumxy;

    vector <vector <double>> first_slice_A = {{A[0][0], A[0][1]}, {A[1][0], A[1][1]}};
    vector <double> second_slice_A = {A[0][2], A[1][2]};
    vector <double> roots1 = solve_linear_system(first_slice_A, second_slice_A);

    vector <double> y1 (x.size(), 0);
    for (int i = 0; i<y1.size(); i++){
        y1[i] = roots1[0] + x[i] * roots1[1];
    }

    vector <vector <double>> A2(3, vector<double>(4,0));
    sumx = 0, sumy = 0, sumx2 = 0, sumxy = 0;
    double sumx3 = 0, sumx4 = 0, sumx2y = 0;
    for (int i = 0; i < x.size(); ++i) {
        sumx += x[i];
        sumy += y[i];
        sumx2 += pow(x[i],2);
        sumx3 += pow(x[i],3);
        sumx4 += pow(x[i],4);
        sumxy += x[i] * y[i];
        sumx2y += pow(x[i],2) * y[i];
    }
    A2[0][0] = x.size();
    A2[0][1] = sumx;
    A2[0][2] = sumx2;
    A2[0][3] = sumy;
    A2[1][0] = sumx;
    A2[1][1] = sumx2;
    A2[1][2] = sumx3;
    A2[1][3] = sumxy;
    A2[2][0] = sumx2;
    A2[2][1] = sumx3;
    A2[2][2] = sumx4;
    A2[2][3] = sumx2y;

    vector <vector <double>> first_slice_A2 = {{A2[0][0], A2[0][1], A2[0][2]}, {A2[1][0], A2[1][1], A2[1][2]}, {A2[2][0], A2[2][1], A2[2][2]}};
    vector <double> second_slice_A2 = {A2[0][3], A2[1][3], A2[2][3]};

    vector <double> roots2 = solve_linear_system(first_slice_A2, second_slice_A2);

    vector <double> y2(x.size(), 0);

    for (int i = 0; i < y2.size(); i++){
        y2[i] = roots2[0] + x[i] * roots2[1] + pow(x[i],2) * roots2[2];
    }

    return make_pair(y1, y2);
}

int main() {
    ofstream fout("answer3.txt");
    vector<double> x = {-0.7, -0.4, -0.1, 0.2, 0.5, 0.8};
    vector<double> y = {-1.4754, -0.81152, -0.20017, 0.40136, 1.0236, 1.7273};

    vector <double> y1 = method_least_squares(x,y).first;
    vector <double> y2 = method_least_squares(x,y).second;

    double F1 = 0.0, F2 = 0.0;
    for (int i = 0; i < y.size(); ++i) {
        F1 += pow((y[i] - y1[i]), 2);
        F2 += pow((y[i] - y2[i]), 2);
    }

    fout << "First-degree polynomial" << endl;
    for (int i = 0; i < y1.size(); i++){
        fout << y1[i] << " ";
    }

    fout << endl << endl;
    fout << "Second-degree polynomial" << endl;
    for (int i = 0; i < y2.size(); i++){
        fout << y2[i] << " ";
    }
    fout << endl << endl;
    fout << "Sum of squares of errors for the first-degree polynomial: " << F1 << endl;
    fout << "Sum of squares of errors for the second-degree polynomial: " << F2 << endl;

    return 0;
}
