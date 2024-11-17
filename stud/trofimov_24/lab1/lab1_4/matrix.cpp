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
pair<int,int> maxNoDiamElem(matrix A){
    pair<int,int> res = {0,1};
    int n= A.size();
    for (int i = 0; i <n ; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i!=j && abs( A[res.first][res.second]) < abs(A[i][j])){
                res = {i,j};
            }
        }
    }
    return res;
}


double getPhi(matrix A,int i,int j ){
    return  atan( 2 * A[i][j] / (A[i][i] - A[j][j]) )/2;
}
matrix getE(int n){
    matrix E(n, vector<double>(n, 0));
    for(int i=0; i<n; i++)
        E[i][i] = 1;
    return E;
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
matrix getU(const matrix& A){
    int n = A.size();
    matrix res = getE(n);
    auto [i, j]= maxNoDiamElem(A);
    double phi =  getPhi(A,i,j);


    res[i][i] = cos(phi);
    res[j][j] = cos(phi);
    res[i][j] = -sin(phi);
    res[j][i] = sin(phi);
    return res;
}
double t(matrix A){
    int n = A.size();
    double res = 0;
    for (int i = 0; i <n ; ++i) {
        for (int j = i+1; j < n; ++j) {
            res+=pow(A[i][j],2);
        }
    }
    return sqrt(res);
}

pair<matrix,matrix> solve(matrix A,double eps0) {
    int k = 0;
    int n = A.size();
    matrix U;
    matrix selfVals(n,std::vector<double> (1) );
    matrix selfVectors(n,std::vector<double> (1) );
    selfVectors = getE(n);
    while( t(A) > eps0){
        U = getU(A);
        selfVectors = mult(selfVectors,U);
        A = mult(mult(getTranspose(U) , A),U);
        k+=1;
    }
    for (int i = 0; i < n; ++i) {
        selfVals[i][0] = A[i][i] ;
    }
    return make_pair(selfVals,selfVectors);

}
matrix multiply_matrix(const matrix& matrix1, const matrix& matrix2) {
    int n1 = matrix1.size();
    int m1 = matrix1[0].size();
    int n2 = matrix2.size();
    int m2 = matrix2[0].size();

    if (m1 != n2) {
        cout << "Incorrect shapes of matrices" << endl;
        return matrix();
    }

    matrix res(n1, vector<double>(m2, 0));

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < m2; ++j) {
            double cntr = 0;
            for (int k = 0; k < m1; ++k) {
                cntr += matrix1[i][k] * matrix2[k][j];
            }
            res[i][j] = cntr;
        }
    }

    return res;
}






