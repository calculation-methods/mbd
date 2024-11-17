#include <cmath>
#include <iostream>
#include <vector>
#include <tuple>
#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <iomanip>

using namespace std;
using tddd = tuple<double, double, double>;
/* f(x, y, z) */
using func = function<double(double, double, double)>;
using vect = vector<tddd>;
using vec = vector<double>;

const double PI = acos(-1.0);
ofstream fout("answer.txt");


double exact_y (double x) {
    return sin(x) + 2  - sin(x)*log((1+sin(x))/(1-sin(x)));
}

double g(double x, double y, double z) {
    return tan(x)*z - 2*y;
}

double f(double x, double y, double z) {
    (void)x;
    (void)y;
    return z;
}

double px(double x) {
    return -tan(x);
}

double qx(double x) {
    return 2.0;
}

double fx(double x) {
    (void)x;
    return 0.0;
}

template <class T>
class Trid {
private:  
    const T EPS = 1e-6;

    int n;
    vector<T> a, b, c;
public:
    Trid(const int &size) : n(size), a(n), b(n), c(n) {}
    Trid(vector<double> &_a, vector<double> &_b, vector<double> &_c) : n(_a.size()), a(_a), b(_b), c(_c) {};

    vector<T> Solve(const vector<T> &d) {
        vector<T> p(n);
        p[0] = -c[0] / b[0];
        vector<T> q(n);
        q[0] = d[0] / b[0];
        for (int i = 1; i < n; ++i) {
            p[i] = -c[i] / (b[i] + a[i]*p[i-1]);
            q[i] = (d[i] - a[i]*q[i-1])/(b[i] + a[i]*p[i-1]);
        }
        vector<T> x(n);
        x.back() = q.back();
        for (int i = n - 2; i >= 0; --i) {
            x[i] = p[i] * x[i+1] +q[i];
        }
        return x;
    }

    friend istream & operator >> (istream &in, Trid<T> &t) {
        in >> t.b[0] >> t.c[0];
        for (int i = 1; i < t.n - 1; ++i) {
            in >> t.a[i] >> t.b[i] >> t.c[i];
        }
        in >> t.a.back() >> t.b.back();
        return in;
    }

    ~Trid() = default;
};


const double EPS = 1e-9;

bool leq(double a, double b) {
    return (a < b) || (abs(b - a) < EPS);
}


vect runge_solve(double l, double r, double y0, double z0, double h) {
        vect res;
        double xk = l;
        double yk = y0;
        double zk = z0;
        res.push_back(make_tuple(xk, yk, zk));
        while (leq(xk + h, r)) {
            double K1 = h * f(xk, yk, zk);
            double L1 = h * g(xk, yk, zk);
            double K2 = h * f(xk + 0.5 * h, yk + 0.5 * K1, zk + 0.5 * L1);
            double L2 = h * g(xk + 0.5 * h, yk + 0.5 * K1, zk + 0.5 * L1);
            double K3 = h * f(xk + 0.5 * h, yk + 0.5 * K2, zk + 0.5 * L2);
            double L3 = h * g(xk + 0.5 * h, yk + 0.5 * K2, zk + 0.5 * L2);
            double K4 = h * f(xk + h, yk + K3, zk + L3);
            double L4 = h * g(xk + h, yk + K3, zk + L3);
            double dy = (K1 + 2.0 * K2 + 2.0 * K3 + K4) / 6.0;
            double dz = (L1 + 2.0 * L2 + 2.0 * L3 + L4) / 6.0;
            xk += h;
            yk += dy;
            zk += dz;
            res.push_back(make_tuple(xk, yk, zk));
        }
        return res;
    }

double runge_romberg_max(const vect & y_2h, const vect & y_h, double p) {
    double coef = 1.0 / (pow(2, p) - 1.0);
    double res = 0.0;
    for (size_t i = 0; i < y_2h.size(); ++i) {
        res = max(res, coef * abs(get<1>(y_2h[i]) - get<1>(y_h[2 * i])));
    }
    return res;
}

double get_eta_next(double eta_prev, double eta, const vect sol_prev, const vect sol, double delta, double gamma, double  y1) {
    double yb_prev = get<1>(sol_prev.back());
    double zb_prev = get<2>(sol_prev.back());
    double phi_prev = delta * yb_prev + gamma * zb_prev - y1; //phi = D * y + g * z - y1
    double yb = get<1>(sol.back());
    double zb = get<2>(sol.back());
    double phi = delta * yb + gamma * zb - y1;
    return eta - (eta - eta_prev) / (phi - phi_prev) * phi; // Метод секущих для решения уравнения phi(eta) = 0;
}


vect shooting_solve(double a, double b, double alpha, double beta, double y0, double delta, double gamma, double y1, double h, double eps) { //четвыртый порядок
    double eta_prev = 0.9;
    double eta = 0.8;
    while (1) {
        vect sol_prev = runge_solve(a, b, eta_prev, y0,h),
        sol = runge_solve(a, b, eta, y0, h);

        double eta_next = get_eta_next(eta_prev, eta, sol_prev, sol, delta, gamma, y1);
        if (abs(eta_next - eta) < eps) {
            return sol;
        } else {
            eta_prev = eta;
            eta = eta_next;
        }
    }
}

class fin_dif {
private:

    double a, b;
    func p, q, f;
    double alpha, beta, y0;
    double delta, gamma, y1;

public:
    fin_dif(const double _a, const double _b,
            const func _p, const func _q, const func _f,
            const double _alpha, const double _beta, const double _y0,
            const double _delta, const double _gamma, const double _y1)
            : a(_a), b(_b), p(_p), q(_q), f(_f),
              alpha(_alpha), beta(_beta), y0(_y0),
              delta(_delta), gamma(_gamma), y1(_y1) {}


};
using tridiag = Trid<double>;
vect fin_dif_solve(double a, double b,
        double alpha, double beta, double y0,
        double delta, double gamma, double y1, double h) {
    size_t n = (b - a) / h;
    vec xk(n + 1);
    for (size_t i = 0; i <= n; ++i) {
        xk[i] = a + h * i; //Разностная сетка
    }
    vec A(n + 1);
    vec B(n + 1);
    vec C(n + 1);
    vec D(n + 1);
    B[0] = (alpha - beta/h);
    C[0] = beta/h;
    D[0] = y0;
    A.back() = -gamma/h;
    B.back() = delta + gamma/h;
    D.back() = y1;
    for (size_t i = 1; i < n; ++i) { // Составляем систему трехдиагональная матрица
        A[i] = 1.0 - px(xk[i]) * h * 0.5;
        B[i] = -2.0 + h * h * qx(xk[i]);
        C[i] = 1.0 + px(xk[i]) * h * 0.5;
        D[i] = h * h * fx(xk[i]);
    }
    tridiag sys_eq(A, B, C);
    vec yk = sys_eq.Solve(D);
    vect res;
    for (size_t i = 0; i <= n; ++i) {
        res.push_back(make_tuple(xk[i], yk[i], NAN));
    }
    return res;
}

double runge_romberg(double y1, double y2, int64_t p) {
    return abs((y1 - y2) / (pow(2, p) - 1));
}

void print_data(vector<tddd> &sol_h1, vector<tddd> &sol_h2, int64_t p) {
    fout << "      x      |" << "      y\t   |" << "   exact y   |" << " y - exact y | runge-romberg\n-------------------------------------------------------------------------\n";
    for (int i = 0; i < sol_h1.size(); ++i) {
        double ex_y = exact_y(get<0>(sol_h1[i]));
        fout << fixed << setprecision(9) << " " << get<0>(sol_h1[i]) << " | " << get<1>(sol_h1[i]) << " | " << ex_y << " | " << abs(ex_y - get<1>(sol_h1[i])) << " | " << runge_romberg(get<1>(sol_h1[i]), get<1>(sol_h2[2*i]), p) << endl;
    }
    fout << endl;
}

int main() {
    cout.precision(6);
    cout << fixed;
    double h = 0.1, eps = 0.001;

    double a = 0, b = PI/6;
    double alpha = 1, beta = 0, y0 = 2;
    double delta = 1, gamma = 0, y1 = 2.5 - 0.5*log(3);
    fout << "Метод стрельбы:" << endl;
    vector<tddd> sol_shooting_h1 = shooting_solve(a, b, alpha, beta, y0, delta, gamma, y1, h, eps), 
    sol_shooting_h2 = shooting_solve(a, b, alpha, beta, y0, delta, gamma, y1,h / 2, eps);
    print_data(sol_shooting_h1, sol_shooting_h2, 4);
    fout << "Погрешность вычислений:" << endl;
    double shooting_err = runge_romberg_max(sol_shooting_h1, sol_shooting_h2, 4);
    fout << shooting_err << endl << endl;
    
    vector<tddd> sol_fin_dif_h1 = fin_dif_solve(a, b, alpha, beta, y0, delta, gamma, y1, h),
    sol_fin_dif_h2 = fin_dif_solve(a, b, alpha, beta, y0, delta, gamma, y1,h / 2);
    fout << "Конечно-разностный метод:" << endl;
    print_data(sol_fin_dif_h1, sol_fin_dif_h2, 2);
    fout << "Погрешность вычислений:" << endl;
    double fin_dif_err = runge_romberg_max(sol_fin_dif_h1, sol_fin_dif_h2, 2);
    fout << fin_dif_err << endl;
}