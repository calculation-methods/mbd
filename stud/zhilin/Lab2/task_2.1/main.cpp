#include <bits/stdc++.h>

using namespace std;


double func(double x){
    return pow(x, 3)  + pow(x, 2) - 2 * x - 1;
}


pair<double, int> newton_method(double x_0, double EPS){
    /*
    Функция f(x): x^3 + x^2 - 2x - 1
    Первая производная: 3x^2 + 2x - 2
    Вторая производная: 6x + 2

    Условие сходимости:
    f(x_0)*f''(x_0) > 0

    Тогда для заданной функции подходит начальное значение:
    x_0 = 2 (выбрано случайно)
    */

    // double dfunc(double x){
    //     return 3 * pow(x, 2) + 2 * x - 2;
    // }

    // double eps(double val1, double val2){
    //     return abs(val1 - val2);
    // }

    auto d_func = [](double x) {
        return 3 * pow(x, 2) + 2 * x - 2;
    };

    auto eps = [](double val1, double val2) {
        return abs(val1 - val2);
    };

    if (func(x_0) == 0)
        return make_pair(x_0, 0);
    
    int k = 1;
    double x_next = x_0 - func(x_0)/d_func(x_0), x_curr = x_0;
    
    while (eps(x_curr, x_next) >= EPS){
        k += 1;
        x_curr = x_next;
        x_next = x_next - func(x_next)/d_func(x_next);
    }
    return make_pair(x_next, k);
}



pair<double, int> simple_iterations_method(double x_0, double a, double b, double q, double EPS){
   
    /*
    Функция f(x): x^3 + x^2 - 2x - 1
    Уравнение с выделенным членом (x = phi(x)): x = (x^3 + x^2 - 1)/2
    phi(x) = (x^3 + x^2 - 1)/2
    phi'(x) = 1.5*x^2 + x

    Условие сходимости:
    1) phi(x)∈[a, b] ∀x∈[a, b]
    2) ∃ q: |phi'(x)|<=q<1 ∀x∈(a, b)

    Тогда для заданной функции подходят параметры:
    q = 0.8
    [a, b] = [-1/3, (-1 + (1 + 1.5*4*q)**0.5)/3]
    x_0 = -0.25
    */

    auto phi = [](double x) {
        return (pow(x, 3) + pow(x, 2) - 1)/2;
    };

    auto eps = [](double val1, double val2, double q) {
        return q*abs(val1 - val2)/(1-q);
    };

    if (func(x_0) == 0)
        return make_pair(x_0, 0);

    int k = 0;
    double x_next = x_0 + 2 * EPS, x_curr = x_0;
    while (eps(x_curr, x_next, q) >= EPS){
        k += 1;
        x_curr = x_next;
        x_next = phi(x_next);
    }
    return make_pair(x_next, k);
}


int main(){
    double deviation = 1e-9, q = 0.8, result;
    int number_of_iterations;

    tie(result, number_of_iterations) = newton_method(2, deviation);
    cout << "\nNewton's method\n";
    cout << "Result (x_k): " <<  result << endl << "Number of iterations (k): " << number_of_iterations << endl;
    cout << "The value of the function at the given root x_k: " << func(result) << endl << endl;

    tie(result, number_of_iterations) = simple_iterations_method(-0.25, -1/3, (-1 + sqrt(1 + 1.5*4*q))/3, q, deviation);
    cout << "\nThe method of simple iterations\n";
    cout << "Result (x_k): " <<  result << endl << "Number of iterations (k): " << number_of_iterations << endl;
    cout << "The value of the function at the given root x_k: " << func(result) << endl << endl;

    return 1;
}
