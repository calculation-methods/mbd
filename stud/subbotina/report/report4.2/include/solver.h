#include <cmath>
#include "tridiag.cpp"
#include <functional>
#include <iostream>
#include <vector>
#include <tuple>


using tddd = std::tuple<double, double, double>;

void print_data(const std::vector<tddd>& v) {
    std::cout << "x = [";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << std::get<0>(v[i]);
    }
    std::cout << "]\n";
    std::cout << "y = [";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << std::get<1>(v[i]);
    }
    std::cout << "]\n";
}

using func = std::function<double(double, double, double)>;
using vect = std::vector<tddd>;
using vec = std::vector<double>;

const double EPS = 1e-9;

bool leq(double a, double b) {
    return ((a < b) || (std::abs(b - a) < EPS));
}

class euler {
private:
    double l, r;
    func f, g;
    double y0, z0;

public:
    euler(const double _l, const double _r,
        const func _f, const func _g,
        const double _y0, const double _z0) : l(_l), r(_r), f(_f), g(_g), y0(_y0), z0(_z0) {}

    vect solve(double h) {
        vect res;
        double xk = l;
        double yk = y0;
        double zk = z0;
        res.push_back(std::make_tuple(xk, yk, zk));
        while (leq(xk + h, r)) {
            double dy = h * f(xk, yk, zk);
            double dz = h * g(xk, yk, zk);
            xk += h;
            yk += dy;
            zk += dz;
            res.push_back(std::make_tuple(xk, yk, zk));
        }
        return res;
    }
};

class runge {
private:
    double l, r;
    func f, g;
    double y0, z0;

public:
    runge(const double _l, const double _r,
        const func _f, const func _g,
        const double _y0, const double _z0) : l(_l), r(_r), f(_f), g(_g), y0(_y0), z0(_z0) {}

    vect solve(double h) {
        vect res;
        double xk = l;
        double yk = y0;
        double zk = z0;
        res.push_back(std::make_tuple(xk, yk, zk));
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
            res.push_back(std::make_tuple(xk, yk, zk));
        }
        return res;
    }
};

class adams {
private:
    double l, r;
    func f, g;
    double y0, z0;

public:
    adams(const double _l, const double _r,
        const func _f, const func _g,
        const double _y0, const double _z0) : l(_l), r(_r), f(_f), g(_g), y0(_y0), z0(_z0) {}

    double calc_tuple(func f, tddd xyz) {
        return f(std::get<0>(xyz), std::get<1>(xyz), std::get<2>(xyz));
    }

    vect solve(double h) {
        if (l + 3.0 * h > r) {
            throw std::invalid_argument("h is too big");
        }
        runge first_points(l, l + 3.0 * h, f, g, y0, z0);
        vect res = first_points.solve(h);
        size_t cnt = res.size();
        double xk = std::get<0>(res.back());
        double yk = std::get<1>(res.back());
        double zk = std::get<2>(res.back());
        while (leq(xk + h, r)) {
           
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
            res.push_back(std::make_tuple(xk1, yk1, zk1));
            ++cnt;
           
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
            res.push_back(std::make_tuple(xk, yk, zk));
        }
        return res;
    }
};

double runge_romberg(const vect& y_2h, const vect& y_h, double p) {
    double coef = 1.0 / (std::pow(2, p) - 1.0);
    double res = 0.0;
    for (size_t i = 0; i < y_2h.size(); ++i) {
        res = std::max(res, coef * std::abs(std::get<1>(y_2h[i]) - std::get<1>(y_h[2 * i])));
    }
    return res;
}



class shooting {
private:
    double a, b;
    func f, g;
    double alpha, beta, y0;
    double delta, gamma, y1;

public:
    shooting(const double _a, const double _b,
        const func _f, const func _g,
        const double _alpha, const double _beta, const double _y0,
        const double _delta, const double _gamma, const double _y1)
        : a(_a), b(_b), f(_f), g(_g),
        alpha(_alpha), beta(_beta), y0(_y0),
        delta(_delta), gamma(_gamma), y1(_y1) {}

    double get_start_cond(double eta) {
        return (y0 - alpha * eta) / beta;
    }

    double get_eta_next(double eta_prev, double eta, const vect sol_prev, const vect sol) {
        double yb_prev = std::get<1>(sol_prev.back());
        double zb_prev = std::get<2>(sol_prev.back());
        double phi_prev = delta * yb_prev + gamma * zb_prev - y1; 
        double yb = std::get<1>(sol.back());
        double zb = std::get<2>(sol.back());
        double phi = delta * yb + gamma * zb - y1;
        return eta - (eta - eta_prev) / (phi - phi_prev) * phi;
    }

    vect solve(double h, double eps) { 
        double eta_prev = 1.0;
        double eta = 0.8;
        while (1) {
            double runge_z0_prev = get_start_cond(eta_prev);
            runge de_solver_prev(a, b, f, g, eta_prev, runge_z0_prev);
            vect sol_prev = de_solver_prev.solve(h);

            double runge_z0 = get_start_cond(eta);
            runge de_solver(a, b, f, g, eta, runge_z0);
            vect sol = de_solver.solve(h);

            double eta_next = get_eta_next(eta_prev, eta, sol_prev, sol);
            if (std::abs(eta_next - eta) < eps) {
                return sol;
            }
            else {
                eta_prev = eta;
                eta = eta_next;
            }
        }
    }
};

class fin_dif {
private:
    using fx = std::function<double(double)>;
    using tridiag = Tridiag<double>;

    double a, b;
    fx p, q, f;
    double alpha, beta, y0;
    double delta, gamma, y1;

public:
    fin_dif(const double _a, const double _b,
        const fx _p, const fx _q, const fx _f,
        const double _alpha, const double _beta, const double _y0,
        const double _delta, const double _gamma, const double _y1)
        : a(_a), b(_b), p(_p), q(_q), f(_f),
        alpha(_alpha), beta(_beta), y0(_y0),
        delta(_delta), gamma(_gamma), y1(_y1) {}

    vect solve(double h) {
        size_t n = (b - a) / h;
        vec xk(n + 1);
        for (size_t i = 0; i <= n; ++i) {
            xk[i] = a + h * i;
        }
        vec a(n + 1);
        vec b(n + 1);
        vec c(n + 1);
        vec d(n + 1);
        b[0] = (alpha - beta / h);
        c[0] = beta / h;
        d[0] = y0;
        a.back() = -gamma / h;
        b.back() = delta + gamma / h;
        d.back() = y1;
        for (size_t i = 1; i < n; ++i) { 
            a[i] = 1.0 - p(xk[i]) * h * 0.5;
            b[i] = -2.0 + h * h * q(xk[i]);
            c[i] = 1.0 + p(xk[i]) * h * 0.5;
            d[i] = h * h * f(xk[i]);
        }
        tridiag sys_eq(a, b, c);
        vec yk = sys_eq.Solve(d);
        vect res;
        for (size_t i = 0; i <= n; ++i) {
            res.push_back(std::make_tuple(xk[i], yk[i], NAN));
        }
        return res;
    }
};

