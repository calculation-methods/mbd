#include <vector>
#include <iostream>
#include "solve.h"


using namespace std;
//The left-hand first derivative: 1.679
//The right-hand first derivative: 2.402
//
//Second derivative: 3.615

int main() {
    double x0 = 1.4;
    vector<double> x = {1.0	,1.2   	,1.4    ,	1.6  , 	1.8   };
    vector<double>y = {2.0,	2.1344,	2.4702	,2.9506,	3.5486};

    int n = x.size();
    vector<double> derivative1 = der1(x,y);

    vector<double> derivative2  = der2(x,y);

    for (int i = 0; i < n - 1; ++i) {
        if (x[i] == x0) {
            cout << "The left-hand first derivative: " << derivative1[i - 1] << endl;
            cout << "The right-hand first derivative: " << derivative1[i] << endl;
            break;
        } else if (x[i] < x0 && x0 < x[i + 1]) {
            cout << "First derivative: " << derivative1[i] << endl;
        }
    }

    cout << endl;

    for (int i = 0; i < n - 2; ++i) {
        if (x[i] <= x0 && x0 <= x[i + 1]) {
            cout << "Second derivative: " << derivative2[i] << endl;
            break;
        }
    }

    return 0;
}