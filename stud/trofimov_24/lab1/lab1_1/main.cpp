#include "matrix.hpp"
#include <vector>
#include <cstdio>

#include <iostream>
using namespace std;

int main() {

    matrix A{
            {-7,-2,-1,-4},
            {-4,6,0,-4},
            {-8,-2,-9,-3},
            {0,0,-7,1},

    };
    matrix b {
            {-12},
            {22},
            {51},
            {49}
    };
    auto [L, U] = LU(A);

    std::cout << std::endl << "L =" << endl;
    print_matrix(L);

    std::cout << std::endl << std::endl;
    std::cout << "U =" << std::endl;
    print_matrix(U);

    std::cout << std::endl << std::endl;
    std::cout << "Determinant = " << getDeterminant(A) << std::endl;

    std::cout << std::endl << std::endl << "Roots =" << std::endl;

    matrix res = solver(A,b);
    print_matrix(res);


    matrix AT = getInverse(A);
    std::cout << std::endl << std::endl << "Reversed matrix =" << endl;

    print_matrix(AT);

    return 0;
}
