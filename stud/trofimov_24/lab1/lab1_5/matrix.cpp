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
int sign(double a){
    return (a >= 0) ? 1 : -1;
}
matrix getE(int n){
    matrix E(n, vector<double>(n, 0));
    for(int i=0; i<n; i++)
        E[i][i] = 1;
    return E;
}
matrix minusMatrix(const matrix& m1, const matrix& m2) {
    int n = m1.size();
    int m = m1[0].size();

    matrix res(n, vector<double>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            res[i][j] = m1[i][j] - m2[i][j];
        }
    }

    return res;
}
matrix getTranspose(matrix m) {
    matrix res(m[0].size(), std::vector<double> (m.size()));

    for(int i= 0; i < m.size(); i++) {
        for(int j = 0; j < m[0].size(); j++) {
            res[j][i] = m[i][j];
        }
    }

    return res;
}

matrix mult(const matrix& matrix1, const matrix& matrix2) {
    int n1 = matrix1.size(), m1 = matrix1[0].size(), m2 = matrix2[0].size();
    matrix res(n1);
    for (int i=0; i<n1; i++)
        for (int j=0; j<m2; j++)
            res[i].push_back(0);

    for (int i=0; i<n1; i++) {
        for (int j=0; j<m2; j++) {
            double cntr = 0;
            for (int k=0; k<m1; k++)
                cntr += matrix1[i][k] * matrix2[k][j];
            res[i][j] = cntr;
        }
    }
    return res;
}



matrix hausehoulder(matrix A,int k){
    int n = A.size();
    matrix V(n,std::vector<double> (1) );
    for(int i=0; i<n; i++){
        if (i < k){
            V[i][0] = 0;

        }
        else if (i == k){
            double temp = 0;
            for (int j=k; j < n; j++){
                temp += A[j][k]*A[j][k];

            }
           V[i][0] = A[i][k] + sign(A[i][k]) * sqrt(temp);
        }
        else
            V[i][0] = A[i][k];
    }

    matrix top = mult(V,getTranspose(V));
    double down = mult(getTranspose(V),V)[0][0];
    matrix rightMatrix  = top;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            rightMatrix[i][j] = rightMatrix[i][j] * 2 / down;
        }
    }
    matrix H(n, vector<double>(n, 0));

    H = minusMatrix(getE(n),rightMatrix);
    return H;


}
pair<matrix ,matrix > QR(matrix A0){
    matrix H1 = hausehoulder(A0,0);
    matrix A1 = mult(H1,A0);
    matrix H2 = hausehoulder(A1,1);
    matrix A2 =mult(H2,A1);
    matrix Q = mult(H1,H2);
    matrix R = A2;
    return {Q,R};

}




double t(const matrix& A){
    double res = 0;
    int n = A.size();
    for(int i=0; i<n; i++)
        for(int j=0; j<i-1; j++)
            res+=pow(A[i][j],2);

    return sqrt(res);
}

matrix solve(matrix A,double eps0) {
    int n = A.size();
    while (t(A) > eps0){
        auto [Q,R] = QR(A);
        A = mult(R,Q);
    }

    matrix res(n,std::vector<double> (1,0) );
    for (int i = 0; i < n; ++i) {
        res[i][0] = A[i][i];
    }
    return res;

}







