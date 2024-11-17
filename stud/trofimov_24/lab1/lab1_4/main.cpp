#include "matrix.h"
#include <iostream>
#include <vector>




int main() {

    matrix A{
            {-8,-4,8},
            {-4,-3,9},
            {8,9,-5}


    };


    auto [selfVals,selfVectors] = solve(A,0.01);
    std::cout<<"self values: "<<std::endl;
    print_matrix(selfVals);
    std::cout<<std::endl;

    std::cout<<"self vectors: "<<std::endl;
    print_matrix(selfVectors);
    return 0;
}
