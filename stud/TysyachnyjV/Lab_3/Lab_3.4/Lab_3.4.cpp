#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

int main(){
    int n = 5;
    double X[n] = {0., 1., 2., 3., 4.};
    double Y[n] = {0.0, 0.5, 1.7321, 3.0, 3.4641};
    double x = 2.0;

    int i;
    for (int j = 0; j < n - 1; ++j){
        if (X[j] <= x and x <= X[j + 1]){
            i = j;
        }
    }

    // первая производная
    double dy = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i]) + ((Y[i + 2] - Y[i + 1]) / (X[i + 2] - X[i + 1]) - (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])) / (X[i + 2] - X[i]) * (2 * x - X[i] - X[i + 1]);
    
    // вторая производная
    double ddy = 2 * ((Y[i + 2] - Y[i + 1]) / (X[i + 2] - X[i + 1]) - (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])) / (X[i + 2] - X[i]);
    cout << dy << " " << ddy;
}