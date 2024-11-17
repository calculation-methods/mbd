#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int main() {
    int n = 5;
    ofstream fout;
    fout.open("output.txt");
    vector<vector<double>> A = {
            {-6, 5, 0, 0, 0},
            {-1, 13, 6, 0 ,0},
            {0, -9, -15, -4, 0},
            {0, 0, -1, -7, 1},
            {0, 0, 0, 9, -18}
    };
    vector<double> d = {51, 100, -12, 47, -90};
    vector<double> P (n, 0);
    vector<double> Q (n, 0);
    vector<double> x (n, 0);
    P[0] = - A[0][1] / A[0][0];
    Q[0] = d[0] / A[0][0];
    for (int i = 1; i < n; i++){
        P[i] = - A[i][i+1] / (A[i][i] + A[i][i-1] * P[i-1]);
        Q[i] = (d[i] - A[i][i-1] * Q[i-1]) / (A[i][i] + A[i][i-1] * P[i-1]);
    }
    for (int i = n - 1; i >= 0; i--){
        x[i] = P[i] * x[i+1] + Q[i];
    }
    for (int i = 0; i < n; i++){
        fout << x[i] << endl;
    }
    return 0;
}
