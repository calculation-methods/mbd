#include <iostream>
#include "matrix.h"



int main() {

    matrix A{
            {-25,4,-4,9},
            {-9,21,5,-6},
            {9,2,19,-7},
            {-7,4,-7,25}


    };
    matrix b {
            {86},
            {29},
            {28},
            {68}
    };

    std::cout<<"Simple iters \nMatrix:";
    auto [inter_res,k1] = zeidel(A,b);
    print_matrix(inter_res);
    std::cout<<"inter count: "<<k1<<std::endl;

    std::cout<<std::endl;

    std::cout<<"Zeidel method\nMatrix:";
    auto [zeidel_res,k2] = zeidel(A,b);
    print_matrix(zeidel_res);
    std::cout<<"inter count: "<<k2<<std::endl;
    auto [s,k0] = simple_iter(A,b);
    return 0;
}
