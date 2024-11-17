#include <bits/stdc++.h>

using namespace std;


// Заданная функция
double f(double x){
    return log(x+1) - 2*pow(x, 2) + 1;
}

// Первая производная заданной функции
double df(double x){
    return 1/(x+1) - 4*x;
}

// Уравнение с выделенным членом
double phi(double x) {
    return sqrt((log(x+1)+1)/2);
};

// Реализация метода Ньютона (по методичке)
pair<double, int> newton(double x0, double EPS){
    /*
        x0 - начальное приближение (корня)
        EPS - точность

        k - количество итераций до достижения необходимой точности
        prev - текущее значение приближения (корня)
        last - следующеее значение приближения (корня)
    */

    int k = 1;
    double last = x0 - f(x0)/df(x0), prev = x0; // задание начальных значений
    
    while (EPS <= abs(last - prev)){ // подсчет ошибки
        k += 1;
        prev = last;
        last = last - f(last)/df(last); // поиск следующего значения через функцию и производную
    }
    return make_pair(last, k);
}


// реализация метода простых итераций (по методичке)
pair<double, int> iteration_method(double x0, double q, double EPS){   
    /*
        x0 - начальное приближение (корня)
        q - значение для подсчета ошибки
        EPS - точность

        k - количество итераций до достижения необходимой точности
        prev - текущее значение приближения (корня)
        last - следующеее значение приближения (корня)
    */
    int k = 1;
    double last = x0*2 + 1, prev = x0; // задание начальных значений

    while (EPS <= q*abs(last - prev)/(1-q)){ // подсчет ошибки
        k += 1;
        prev = last;
        last = phi(last); // поиск следующего значения через уравнение с выделенным членом
    }
    return make_pair(last, k);
}


int main(){
    double x_res;
    int iters_count;

    tie(x_res, iters_count) = newton(2, 1e-5);
    cout << "Result: " <<  x_res << endl << "Iterations (newton): " << iters_count << endl;

    tie(x_res, iters_count) = iteration_method(0.5, 0.8, 1e-5);
    cout << "Iterations (simple iterations): " << iters_count << endl << endl;
}
