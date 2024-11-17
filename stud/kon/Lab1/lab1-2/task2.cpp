#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

vector<double> the_run_through_method(vector<vector<double>> matrix, vector<double> b, int n){
    vector<double> P(n, 0.0), Q(n, 0.0), x(n, 0.0);
    //a=matrix[i][0], b=matrix[i][1], c=matrix[i][2], d=b[i]
    P[0] = -matrix[0][2]/matrix[0][1];
    Q[0] = b[0]/matrix[0][1];
    for (int i = 1; i < n; i++)
    {
        P[i] = -matrix[i][2]/(matrix[i][1] + matrix[i][0]*P[i-1]);
        Q[i] = (b[i] - matrix[i][0]*Q[i-1])/(matrix[i][1] + matrix[i][0]*P[i-1]);
    }
    P[n-1] = 0;
    x[n-1] = Q[n-1];
    for (int i = n - 2; i >= 0; i--)
    {
        x[i] = P[i]*x[i+1] + Q[i];
    }
    return x;
}

int main(){
    int n = 5;
    vector<vector<double>> matrix(n, vector <double>(n, 0.0));
    vector<double> b(n, 0);
    ifstream in1("matrix.txt"), in2("b.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n-2; j++){
            in1 >> matrix[i][j];
        }
    }
    for (int i = 0; i < n; i++)
    {
        in2 >> b[i];
    }
    vector<double> res = the_run_through_method(matrix, b, n);

    ofstream out("output.txt");
  
    out << "Решение системы: " << endl;
    for (int i = 0; i < n; ++i)
        out << res[i] << endl;
    
    return 0;
}