#include "matrix.h"
#include <iostream>
#include <vector>

1.
begin try
declare @var int = 2
print @Var
        END TRY
        begin catch
print @Var
        END CATCH
2.
if 1 = 1begin
declare @var int = 2
print @Var
        END
print @Var
int main() {

    matrix A{
            {-3,1,-1},
            {6,9,-4},
            {5,-4,-8}


    };


    matrix res = solve(A,0.01);

    std::cout << "lambda :" << std::endl;
    print_matrix(res);


    return 0;
}
