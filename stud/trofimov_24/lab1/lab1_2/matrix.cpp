#include <vector>
#include <iostream>

using namespace std;


using matrix = vector<vector<double> >;
void print_matrix(const matrix& matrix1) {
    std::cout <<  "[ "<<std::endl;

    for(const auto& vect: matrix1) {
        for (auto x: vect)
            std::cout << x << " ";
        std::cout << std::endl;
    }
    std::cout <<  "]" << endl;

}

matrix solver(matrix& A,matrix& res){
    int n = A.size();
    double a = 0;
    double b = A[0][0];
    double c = A[0][1];
    double d = res[0][0];
    matrix P(n,std::vector<double> (1) );
    matrix Q(n,std::vector<double> (1) );
    matrix roots(n,std::vector<double> (1) );
    P[0][0] = -c/b;
    Q[0][0] = d/b;
    for (int i = 1; i < n; ++i) {
        a = A[i][i-1];
        b = A[i][i];

        if ((i+1) < n){
            c = A[i][i+1];

        }else{
            c = 0;
        }
        d = res[i][0];

        P[i][0] = -c/(b + a*P[i-1][0]);
        Q[i][0] = (d - a*Q[i-1][0])/(b + a*P[i-1][0]);

    }


    roots[n-1][0] =  Q[n-1][0];
    for (int i = n-2; i >=0; --i) {
        roots[i][0] = P[i][0]*roots[i+1][0] + Q[i][0];
    }
    return roots;



}



