#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

// исходная функция
double f(double x){
    return pow(x + 2, 0.5) - 2 * cos(x);
}

// первая производная исходной функции
double df(double x){
    return 0.5 * pow(x + 2, -0.5) + 2 * sin(x);
}

// вторая производная исходной функции
double ddf(double x){
    return -0.25 * pow(x + 2, -1.5) + 2 * cos(x);
}

// эквивалентная функция
double phi(double x){
    return acos(pow(x + 2, 0.5) / 2);
}

// первая производная эквивалентнаой функции
double dphi(double x){
    return -0.5 / pow(4 - x * x, 0.5);
}

int main(){

    // метод Ньютона
    double x0 = 1;
    double eps = 0.001;
    if (f(x0) * ddf(x0) > 0){
        cout << "Начальная точка подходит" << "\n";
        
        int k = 0;
        double x[2] = {x0 - eps - 1, x0};
        while (abs(x[1] - x[0]) >= eps){
            x[0] = x[1];
            x[1] -= f(x[1]) / df(x[1]);
            k += 1;
        }
        cout << x[1] << " " << k << "\n";
    }
    else{
        cout << "Начальная точка не подходит" << "\n";
    }

    // метод простых итераций
    // начальный интервал, q и эквивалентная функция подобраны аналитически, исходя из условия сходимости
    int k = 0;
    double q = abs(dphi(1));
    double x[2] = {0., (0. + 1.) / 2};
    while (q / (1 - q) * abs(x[1] - x[0]) > eps){
        x[0] = x[1];
        x[1] = phi(x[1]);
        k += 1;
    }
    cout << x[1] << " " << k << "\n";

}