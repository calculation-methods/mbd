#include <bits/stdc++.h>
using namespace std;

double first_method(double x, vector<double>& x_v, int n) {
    vector<double> k_vect(x_v.size());

    for (int i = 0; i < n; i++)
        k_vect[i] = asin(x_v[i]);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; ++j)
            if (i != j)
                k_vect[i] /= x_v[i] - x_v[j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; ++j)
            if (i != j)
                k_vect[i] *= x - x_v[j];

    double answ = 0;
    for(int i=0; i<n; i++)
        answ += k_vect[i];
    return answ;
}

double second_method(double x, const vector<double>& x_v, int n) {
    vector<double> k_vect(x_v.size());

    for (int i = 0; i < n; i++)
        k_vect[i] = asin(x_v[i]);

    for (int i = 1; i < n; i++)
        for (int j = n - 1; j > i-1; --j) 
            k_vect[j] = (k_vect[j] - k_vect[j - 1]) / (x_v[j] - x_v[j - i]);

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; ++j) {
            k_vect[i] *= x - x_v[j];
        }
    }

    double answ = 0;
    for(int i=0; i<n; i++)
        answ += k_vect[i];
    return answ;
}

int main() {
    vector<double> x_vect = {-0.4, 0.0, 0.2, 0.5};
    double t = 0.1;

    cout << "\nМногочлен Лагранжа\n" << "   Ответ " << first_method(t, x_vect, 4) << endl << "   Погрешность " << abs(first_method(t, x_vect, 4) - asin(t)) << endl;
    cout << "\nМногочлен Ньютона\n" << "   Ответ " << second_method(t, x_vect, 4) << endl << "   Погрешность " << abs(second_method(t, x_vect, 4) - asin(t)) << endl;

    return 0;
}
