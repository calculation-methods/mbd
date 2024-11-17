#include <functional>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <iomanip>

using namespace std;
using tddd = tuple<double, double, double>;

/* f(x, y, z) */
using func = function<double(double, double, double)>;
using vect = vector<tddd>;
using vec = vector<double>;

ofstream fout("answer.txt");
const double EPS = 1e-9;


bool leq(double a, double b) {
    return (a < b) || (abs(b - a) < EPS);
}


double g(double x, double y, double z) {
    return (-x/z);
}

double f(double x, double y, double z) {
    (void)x;
    (void)y;
    return z;
}

vect solve_euler(double l, double r, double y0, double z0, double h){
    vect res;
    double xk = l;
    double yk = y0;
    double zk = z0;
    res.push_back(make_tuple(xk, yk, zk));
    while (leq(xk + h, r)) {
        double dy = h * f(xk, yk, zk);
        double dz = h * g(xk, yk, zk);
        xk += h;
        yk += dy;
        zk += dz;
        res.push_back(make_tuple(xk, yk, zk));
    }
    return res;
}
vect solve_runge(double l, double r, double y0, double z0, double h) { // 4ый Порядок
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

double calc_tuple(func f, tddd xyz) {
    return f(get<0>(xyz), get<1>(xyz), get<2>(xyz));
}

vect solve_adams(double l, double r, double y0, double z0, double h) { //4ый порядок
    if (l + 3.0 * h > r) {
        throw invalid_argument("h is too big"); //Многошаговый метод: решение зависит не от данных в одном узле, а от нескольких
    } // Через интеграл
    vect res = solve_runge(l, l + 3.0 * h, y0, z0, h);// Первые точки через рунге
    size_t cnt = res.size();
    double xk = get<0>(res.back());
    double yk = get<1>(res.back());
    double zk = get<2>(res.back());
    while (leq(xk + h, r)) {
        /* Предиктор */
        double dy = (h / 24.0) * (55.0 * calc_tuple(f, res[cnt - 1])
                                    - 59.0 * calc_tuple(f, res[cnt - 2])
                                    + 37.0 * calc_tuple(f, res[cnt - 3])
                                    - 9.0 * calc_tuple(f, res[cnt - 4]));
        double dz = (h / 24.0) * (55.0 * calc_tuple(g, res[cnt - 1])
                                    - 59.0 * calc_tuple(g, res[cnt - 2])
                                    + 37.0 * calc_tuple(g, res[cnt - 3])
                                    - 9.0 * calc_tuple(g, res[cnt - 4]));
        double xk1 = xk + h;
        double yk1 = yk + dy;
        double zk1 = zk + dz;
        res.push_back(make_tuple(xk1, yk1, zk1));
        ++cnt;
        /* Корректор */
        dy = (h / 24.0) * (9.0 * calc_tuple(f, res[cnt - 1])
                            + 19.0 * calc_tuple(f, res[cnt - 2])
                            - 5.0 * calc_tuple(f, res[cnt - 3])
                            + 1.0 * calc_tuple(f, res[cnt - 4]));
        dz = (h / 24.0) * (9.0 * calc_tuple(g, res[cnt - 1])
                            + 19.0 * calc_tuple(g, res[cnt - 2])
                            - 5.0 * calc_tuple(g, res[cnt - 3])
                            + 1.0 * calc_tuple(g, res[cnt - 4]));
        xk += h;
        yk += dy;
        zk += dz;
        res.pop_back();
        res.push_back(make_tuple(xk, yk, zk));
    }
    return res;
}

double max_runge_romberg(const vect & y_2h, const vect & y_h, double p) {
    double coef = 1.0 / (pow(2, p) - 1.0);
    double res = 0.0;
    for (size_t i = 0; i < y_2h.size(); ++i) {
        res = max(res, coef * abs(get<1>(y_2h[i]) - get<1>(y_h[2 * i])));
    }
    return res;
}


double y(double x) {
    return 1 +log(abs(x));
}

double runge_romberg(double y1, double y2, int64_t p) {
    return abs((y1 - y2) / (pow(2, p) - 1));
}


void print_data(vector<tddd> &sol_h1, vector<tddd> &sol_h2, int64_t p) {
    fout << "      x      |" << "      y\t   |" << "   exact y   |" << " y - exact y | runge-romberg\n-------------------------------------------------------------------------\n";
    for (int i = 0; i < sol_h1.size(); ++i) {
        double exact_y = y(get<0>(sol_h1[i]));
        fout << fixed << setprecision(9) << " " << get<0>(sol_h1[i]) << " | " << get<1>(sol_h1[i]) << " | " << exact_y << " | " << abs(exact_y - get<1>(sol_h1[i])) << " | " << runge_romberg(get<1>(sol_h1[i]), get<1>(sol_h2[2*i]), p) << endl;
    }
    fout << endl;
}

int main() {
    double l = 1, r = 2, y0 = 1, z0 = 1, h = 0.1;

    vector<tddd> sol_euler_h1 = solve_euler(l, r, y0, z0, h), sol_euler_h2 = solve_euler(l, r, y0, z0, h/2);
    fout << "Метод Эйлера:" << endl;
    print_data(sol_euler_h1, sol_euler_h2, 1);
    fout << max_runge_romberg(sol_euler_h1, sol_euler_h2, 1);
    fout << endl;
    fout << endl;

    vector<tddd> sol_runge_h1 = solve_runge(l, r, y0, z0, h), sol_runge_h2 = solve_runge(l, r, y0, z0, h/2);
    fout << "Метод Рунге-Кутты:" << endl;
    print_data(sol_runge_h1, sol_runge_h2, 4);
    fout << max_runge_romberg(sol_runge_h1, sol_runge_h2, 4);
    fout << endl;
    fout << endl;
    
    vector<tddd> sol_adams_h1 = solve_adams(l, r, y0, z0, h), sol_adams_h2 = solve_adams(l, r, y0, z0, h/2);
    fout << "Метод Адамса:" << endl;
    print_data(sol_adams_h1, sol_adams_h2, 4);
    fout << max_runge_romberg(sol_adams_h1, sol_adams_h2, 4);
    fout << endl;
    fout << endl;
}