#include <vector>
#include <iostream>

#include "matrix.h"

using namespace std;
int main() {

    matrix A{
            {-11,9,0,0,0},
            {-9,17,6,0,0},
            {0,5,20,8,0},
            {0,0-6,-20,7},
            {0,0,0,2,8},


    };
    matrix b {
            {-117},
            {97},
            {-6},
            {59},
            {-86}
    };
    matrix res = solver(A,b);
    std::cout << "Roots = " << std::endl;

    print_matrix(res);

    return 0;
}
