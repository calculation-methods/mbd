#include "matrix.hpp"
#include <vector>
#include <cstdio>

#include <iostream>
using namespace std;


using matrix = std::vector<std::vector<double> >;

void print_matrix(const matrix& matrix1) {
    std::cout <<  "[ "<<std::endl;

    for(const auto& vect: matrix1) {
        for (auto x: vect)
            std::cout << x << " ";
        std::cout << std::endl;
    }
    std::cout <<  "]" << endl;
}
matrix getTranspose(matrix m) {
    matrix solution(m[0].size(), std::vector<double> (m.size()));

    for(int i= 0; i < m.size(); i++) {
        for(int j = 0; j < m[0].size(); j++) {
            solution[j][i] = m[i][j];
        }
    }

    return solution;
}
pair<matrix, matrix> LU(matrix& A) {
    int n = A.size();
    matrix L(A[0].size(), std::vector<double> (A.size()));

    matrix U = A;


    for(int i = 0; i < n; i++)
        for(int j = i; j < n; j++)
            L[j][i]=U[j][i]/U[i][i];

    for(int k = 1; k < n; k++)
    {
        for(int i = k-1; i < n; i++)
            for(int j = i; j < n; j++)
                L[j][i]=U[j][i]/U[i][i];

        for(int i = k; i < n; i++)
            for(int j = k-1; j < n; j++)
                U[i][j]=U[i][j]-L[i][k-1]*U[k-1][j];
    }
    return make_pair(L, U);
}

matrix solver(matrix& A, matrix& b)
{
    auto[L,U] = LU(A);
    int n = A.size();
    matrix y(n, std::vector<double> (1));
    matrix x(n, std::vector<double> (1));

    for (int i = 0; i < n; i++)
    {
        double temp = 0;

        for (int k = 0; k < i; k++)
        {
            temp += L[i][k] * y[k][0];
        }


        y[i][0] = (b[i][0] - temp) / L[i][i];
    }

    for (int i = n - 1; i >= 0; i--)
    {
        double temp = 0;

        for (int k = n - 1; k > i; k--)
        {
            temp += U[i][k] * x[k][0];
        }

        x[i][0] = (y[i][0] - temp) / U[i][i];
    }
    return x;
}


double getDeterminant(matrix m) {

    int dimension = m.size();

    if(dimension == 0) {
        return 1;
    }

    if(dimension == 1) {
        return m[0][0];
    }

    if(dimension == 2) {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    double result = 0;
    int sign = 1;
    for(int i = 0; i < dimension; i++) {

        matrix subVect(dimension - 1, std::vector<double> (dimension - 1));
        for(int j = 1; j < dimension; j++) {
            int z = 0;
            for(int n = 0; n < dimension; n++) {
                if(n != i) {
                    subVect[j-1][z] = m[j][n];
                    z++;
                }
            }
        }

        result = result + sign * m[0][i] * getDeterminant(subVect);
        sign = -sign;
    }

    return result;
}
matrix getCofactor(const matrix m) {


    matrix solution(m.size(), std::vector<double> (m.size()));
    matrix subVect(m.size() - 1, std::vector<double> (m.size() - 1));

    for(int i = 0; i < m.size(); i++) {
        for(int j = 0; j < m[0].size(); j++) {

            int p = 0;
            for(size_t x = 0; x < m.size(); x++) {
                if(x == i) {
                    continue;
                }
                int q = 0;

                for(size_t y = 0; y < m.size(); y++) {
                    if(y == j) {
                        continue;
                    }

                    subVect[p][q] = m[x][y];
                    q++;
                }
                p++;
            }

            solution[i][j] = pow(-1, i + j) * getDeterminant(subVect) + 0.0;
        }
    }

    return solution;
}

matrix getInverse(matrix& m) {
    double d = 1.0/getDeterminant(m);
    matrix solution(m.size(), std::vector<double> (m.size()));

    for(int i = 0; i < m.size(); i++) {
        for(int j = 0; j < m.size(); j++) {


            solution[i][j] = m[i][j];
        }
    }

    solution = getTranspose(getCofactor(solution));

    for(size_t i = 0; i < m.size(); i++) {
        for(size_t j = 0; j < m.size(); j++) {
            solution[i][j] *= d;
        }
    }

    return solution;
}
