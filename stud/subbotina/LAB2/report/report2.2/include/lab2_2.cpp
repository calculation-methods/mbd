#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <locale.h>
#include "matrix.h"

using namespace std;

const double a = 1.0;
const int n = 2;
const double start_delta = 0.2;
const double min_delta = 0.01;
const double search_step = 0.01;

vector<double> vector_minus(const vector<double>& a, const vector<double>& b) {
    vector<double> minus = a;
    for (unsigned i = 0; i < minus.size(); ++i) {
        minus[i] -= b[i];
    }
    return minus;
}

vector<double> vector_plus(const vector<double>& a, const vector<double>& b) {
    vector<double> plus = a;
    for (unsigned i = 0; i < plus.size(); ++i) {
        plus[i] += b[i];
    }
    return plus;
}

double norm_of_vectors(const vector<double>& x1, const vector<double>& x2) {
    double norm = 0.0;
    for (unsigned i = 0; i < x1.size(); ++i) {
        norm = max(abs(x1[i] - x2[i]), norm);
    }
    return norm;
}

double f1(const vector<double>& x) {
    return a * x[0] * x[0] - x[0] + x[1]*x[1] - 1.0;
}

double df1_dx1(const vector<double>& x) {
    return 2.0 * a * x[0] - 1.0;
}

double df1_dx2(const vector<double>& x) {
    return 2.0*x[1];
}


double f2(const vector<double>& x) {
    return x[1] - tan(x[0]);
}


double df2_dx2(const vector<double>& x) {
    return 1.0;
}

double df2_dx1(const vector<double>& x) {
    return - 1.0 / (cos(x[0]) * cos(x[0]));
}

double fitta2(const vector<double>& x) {
    return sqrt(1 - a * x[0]*x[0] + x[0]);
}

double dfitta2_dx1(const vector<double>& x) {
    return (2 * a *x[0]-1) * sqrt(1 - a * x[0] * x[0] + x[0]) / (2 * a *(x[0] * x[0] - x[0]-1));
}

double dfitta2_dx2(const vector<double>& x) {
    return 0.0;
}

double fitta1(const vector<double>& x) {
    return atan(x[1]);
}

double dfitta1_dx1(const vector<double>& x) {
    return 0.0;
}

double dfitta1_dx2(const vector<double>& x) {
    return 1.0 / (x[1] * x[1] + 1);
}

void set_dfitta_matrix(Matrix& A, const vector<double>& x) {
    A = Matrix(n, n);
    A[0][0] = dfitta1_dx1(x);
    A[0][1] = dfitta1_dx2(x);
    A[1][0] = dfitta2_dx1(x);
    A[1][1] = dfitta2_dx2(x);
}

vector<double> nex_step(const vector<double>& x) {
    vector<double> ans(n);
    ans[0] = fitta1(x);
    ans[1] = fitta2(x);
    return ans;
}

int newton_method(const vector<double>& x0, vector<double>& x, double alfa) {
	int itter = 0;
	vector<double> x_i;
	double det1, det2, detJ;
    vector<double> det_n(n);
	int k = 0;
    x = x0;
    do {
        fill(det_n.begin(), det_n.end(), 0);
		x_i.swap(x);
		det1 = f1(x_i) * df2_dx2(x_i) - f2(x_i) * df1_dx2(x_i);
		det2 = df1_dx1(x_i) * f2(x_i) - df2_dx1(x_i) * f1(x_i);
		detJ = df1_dx1(x_i) * df2_dx2(x_i) - df2_dx1(x_i) * df1_dx2(x_i);
		det_n[0] = - det1 / detJ;
		det_n[1] = - det2 / detJ;
        x = vector_plus(x_i, det_n);
        ++itter;
    } while (norm_of_vectors(x, x_i) > alfa);
	return itter;
}

double find_q(const vector<double>& x0) {
    double delta = start_delta * 2.0;
    vector<double> ans_x = x0;
    Matrix Dfitta;
    double ans;
    do {
        delta /= 2.0;
        for (unsigned i = 0; i < x0.size(); ++i) {
            double maximum = 0.0;
            vector<double> x = x0;
            for (double v = x0[i] - delta; v <= x0[i] + delta; v += search_step) {
                x[i] = v;
                set_dfitta_matrix(Dfitta, x);
                double norm = Dfitta.get_norm();
                if (norm > maximum) {
                    ans_x[i] = v;
                    maximum = norm;
                }
            }
        }
        set_dfitta_matrix(Dfitta, ans_x);
        ans = Dfitta.get_norm();
    } while (ans >= 1.0 && delta >= min_delta);
    return ans;
}

int itteration_method(const vector<double>& x0, vector<double>& x, double alfa, double& q) {
    int itter = 0;
    vector<double> x_i;
    vector<double> fx, dx;
    q = find_q(x0);
    if (q < 1.0) {
        alfa *= (1.0 - q);
        alfa /= q;
    }
    x = x0;
    do {
        x_i.swap(x);
        x = nex_step(x_i);
        ++itter;
    } while (norm_of_vectors(x, x_i) > alfa);
    return itter;
}

void print_solution(const vector<double>& x_n, int itter_n, const vector<double>& x_i, int itter_i, const double q) {
    cout << "============================================================" << endl;
    cout << "|                      ANSWER:                             |" << endl;
    cout << "============================================================" << endl;
    cout << "|                   NEWTON METHOD:                         |" << endl;
    cout << "============================================================" << endl;
    cout << "x = (" << x_n[0];
    for (int i = 1; i < n; ++i) {
        cout << ", " << x_n[i];
    }
    cout << ")" << endl;
    cout << "Itterations: " << itter_n << endl;
    cout << "============================================================" << endl;
    cout << "|               SIMPLE ITTERATIONS METHOD:                 |" << endl;
    cout << "============================================================" << endl;
    if (q >= 1.0) {
        cout << "Sufficient condition not done!" << endl;
    }
    else {
        cout << "Sufficient condition done with q: " << q << endl;
    }
    cout << "x = (" << x_i[0];
    for (int i = 1; i < n; ++i) {
        cout << ", " << x_i[i];
    }
    cout << ")" << endl;
    cout << "Itterations: " << itter_i << endl;
    cout << "============================================================" << endl;
}

int main() {
    setlocale(0, "");
    vector<double> x0(n), x_n(n), x_i(n, 0);
    double alfa, q;
    int itter_n, itter_i = 0;
    cout << "Введите точность вычислений: ";
    cin >> alfa;
    cout << "Введите начальную точку: \n";
    for (int i = 0; i < n; ++i) {
        cin >> x0[i];
    }

    itter_n = newton_method(x0, x_n, alfa);
    itter_i = itteration_method(x0, x_i, alfa, q);
    cout << "Метод Ньютона: \n";
    cout << "x = (" << x_n[0];
    for (int i = 1; i < n; ++i) {
        cout << ", " << x_n[i];
    }
    cout << ")" << endl;
    cout << "Количество итераций: " << itter_n << endl;

    cout << "Метод простой итерации: \n";
    cout << "x = (" << x_i[0];
    for (int i = 1; i < n; ++i) {
        cout << ", " << x_i[i];
    }
    cout << ")" << endl;
    cout << "Количество итераций: " << itter_i << endl;

    return 0;
}