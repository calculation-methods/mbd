#include <bits/stdc++.h>

using namespace std;
using float_vect = vector<double>;

// x1^2 + x2^2 - 9 = 0
auto f1 = [](float_vect x) {
    return pow(x[0], 2) + pow(x[1], 2) - 9;
};

// x1 - e^(x2) + 3 = 0
auto f2 = [](float_vect x) {
    return x[0] - exp(x[1]) + 3;
};

// Решение системы методом Ньютона
pair<float_vect, int> nm(float_vect x_0, double deviation) {

    // функция для подсчёта ошибки (по методичке)
    auto eps = [](float_vect& val1, float_vect& val2) {
        double d = 0.0;
        for (int i = 0; i < val1.size(); ++i) {
            d = max(d, abs(val1[i] - val2[i]));
        }
        return d;
    };

    // поиск определителя матрицы 
    auto det = [](float_vect& x, const vector<vector<function<double(float_vect)>>>& m) {
        return m[0][0](x) * m[1][1](x) - m[0][1](x) * m[1][0](x);
    };

    // Частные производные заданных функций
    auto f1x1 = [](const float_vect& x) { return 2 * x[0]; };
    auto f1x2 = [](const float_vect& x) { return 2 * x[1]; };
    auto f2x1 = [](const float_vect& x) { return 1; };
    auto f2x2 = [](const float_vect& x) { return -exp(x[1]); };

    // Формирование матриц из методички (J это якобиан)
    vector<vector<function<double(float_vect)>>> A1 = {{f1, f1x2}, {f2, f2x2}};

    vector<vector<function<double(float_vect)>>> A2 = {{f1x1, f1}, {f2x1, f2}};

    vector<vector<function<double(float_vect)>>> J = {{f1x1, f1x2}, {f2x1, f2x2}};

    float_vect x_next = { x_0[0] - det(x_0, A1)/det(x_0, J), x_0[1] - det(x_0, A2)/det(x_0, J) }, x_curr = x_0; // задание начальних значений (в форме вектора)
    int k = 1;

    while (deviation <= eps(x_curr, x_next)){
        k += 1;
        x_curr = x_next;
        x_next = { x_next[0] - det(x_next, A1)/det(x_next, J), x_next[1] - det(x_next, A2)/det(x_next, J) };
    }
    return make_pair(x_next, k);
}

// Решение системы методом простых итераций
pair<float_vect, int> sim(float_vect x_0, double q, double deviation) {

    // функция для подсчёта ошибки (по методичке)
    auto eps = [](float_vect& val1, float_vect& val2, double q) {
        double d = 0.0;
        for (int i = 0; i < val1.size(); ++i) {
            d = max(d, abs(val1[i] - val2[i]));
        }
        return d*q/(1-q);
    };

    float_vect x_next = x_0, x_curr = {x_0[0] + 5, x_0[1]*3 + 5}; // задание начальних значений (в форме вектора)
    int k = 1;

    while (deviation <= eps(x_curr, x_next, q)){
        k += 1;
        x_curr = x_next;
        x_next = { sqrt(9-pow(x_next[1], 2)), log(x_next[0]+3) }; // поиск вектора следующих значений
    }
    return make_pair(x_next, k);
}


int main(){
    float_vect r;
    int iter_count;
    // Вывод метода Ньютона
    tie(r, iter_count) = nm({1.0, 1.0}, 1e-5);
    cout << "x1 = "  << r[0] << endl << "x2 = " << r[1] << endl << "Iterations (newton): " << iter_count << endl;
    // Вывод простых итераций
    tie(r, iter_count) = sim({1.0, 1.0}, 0.7, 1e-5);
    cout << "Iterations (simple iterations): " << iter_count << endl << endl;
}
