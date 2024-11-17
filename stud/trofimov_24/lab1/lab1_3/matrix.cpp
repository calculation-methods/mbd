#include "matrix.h"
#include <iostream>
#include <vector>
using namespace std;


using matrix = vector<vector<double> >;

void print_matrix(const matrix& matrix1) {
    for(const auto& vect: matrix1) {
        for (auto x: vect)
            cout << x << " ";
        cout << endl;
    }
}




pair<matrix, int> simple_iter(matrix& A,matrix& res){
    int n = A.size();
    matrix B(n,std::vector<double> (1) );
    matrix x(n,std::vector<double> (1) );
    matrix x_prev(n,std::vector<double> (1) );

    x_prev = B;
    for (int i = 0; i < n; ++i) {
        B[i][0] = res[i][0]/A[i][i];
    }
    double eps0  = 1e-2;
    double eps = 0;
    bool flag = true;
    int k = 0;
    while (flag or eps  > eps0){
        flag = false;
        k+=1;
        for (int i = 0; i < n; ++i) {
            x[i] = B[i];
            for (int j = 0; j < n; ++j) {
                if (i < j)
                    x[i][0] += x_prev[j][0]*A[i][j]/A[i][i];
                else if (i!=j)
                    x[i][0] += x_prev[j][0]*A[i][j]/A[i][i];

            }
        }
        x_prev = x;
    };


    return make_pair(x,k);


}
double getEps( matrix& v1,  matrix& v2) {
    double eps = 0;
    for (int i = 0; i < v1.size(); i++)
        eps += pow(v1[i][0] - v2[i][0], 2);
    return sqrt(eps);
}
pair<matrix, int> zeidel(matrix& A,matrix& res){
    int n = A.size();
    matrix B(n,std::vector<double> (1) );
    matrix x(n,std::vector<double> (1) );
    matrix x_prev(n,std::vector<double> (1) );

    x = B;
    x_prev = x;
    for (int i = 0; i < n; ++i) {
        B[i][0] = res[i][0]/A[i][i];
    }
    double eps0  = 1e-2;
    double eps = 0;
    bool flag = true;
    int k = 0;
    while (flag or eps  > eps0){
        flag = false;
        k+=1;
        for (int i = 0; i < n; ++i) {
            x[i] = B[i];
            for (int j = 0; j < n; ++j) {
                if (i < j)
                    x[i][0] += x_prev[j][0]*A[i][j]/A[i][i];
                else if (i!=j)
                    x[i][0] += x_prev[j][0]*A[i][j]/A[i][i];

            }
        }
        eps = getEps(x, x_prev);

        x_prev = x;
    };


    return make_pair(x,k);


}



