#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;


class Polynom {
    private:
        vector<double> _poly;
    public:
    Polynom(int n) {
        _poly.resize(n);
    }        
    Polynom() {}
    Polynom(const vector<double> &v){
        _poly.resize(v.size());
        for (int i = 0; i < v.size(); ++i){
            _poly[i] = v[i];
        }
    }

    int size() {
        return _poly.size();
    }

    double & operator[](int i) {
        return _poly[i];
    }

    void push_back(double k){
        _poly.push_back(k);
    }

    friend Polynom operator+(Polynom &lhs, Polynom &rhs) {
        int n_max = max(lhs.size(), rhs.size());
        int n_min = min(lhs.size(), rhs.size());
        Polynom res(n_max);
        for (int i = 0; i < n_min; ++i) {
            res[i] = lhs[i] + rhs[i];
        }
        for (int i = n_min; i < n_max; ++i) {
            if (lhs.size() > rhs.size()) {
                res[i] = lhs[i];
            } else {
                res[i] = rhs[i];
            }
        }
        return res;
    }

    friend Polynom operator*(Polynom &lhs, Polynom &rhs) {
        Polynom res(lhs.size() + rhs.size() - 1);
        for (int i = 0; i < rhs.size(); ++i) {
            for (int j = 0; j < lhs.size(); ++j) {
                res[i + j] += lhs[j] * rhs[i];
            }
        }
        return res;
    }

    friend Polynom operator*(Polynom &lhs, double rhs) {
        Polynom res(lhs.size());
        for (int i = 0; i < lhs.size(); ++i) {
            res[i] = lhs[i] * rhs;
        }
        return res;
    }
};

ostream& operator<<(ostream& stream, Polynom poly)
{    
    for (int i = 0; i < poly.size(); ++i) {
        if (i != poly.size() - 1)
            stream << poly[i] << "x^" << i << " + ";
        else
            stream << poly[i] << "x^" << i << "\n";
    }
    return stream;
}

Polynom lagrange_polynom(vector<double> &x, vector<double> &y) {
    int n = x.size();
    Polynom L(n);
    for (int i = 0; i < n; ++i) {
        Polynom li;
        li.push_back(1);
        double c = 1;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                Polynom d({(-1) * x[j], 1});
                li = li*d;
                c *= (x[i] - x[j]);
            }
        }
        L = L + li* (y[i]/c);
    }
    return L;
}

Polynom newton_polynom(vector<double> &x, vector<double> &y) {
    int n = x.size();
    vector<vector<double>> table(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        table[i][0] = y[i];
    }
    for (int j = 1; j < n; ++j) {
        for (int i = 0; i < n - j; ++i) {
            table[i][j] = (table[i][j - 1] - table[i + 1][j - 1]) / (x[i] - x[i + j]);
        }
    }
    Polynom P(n);
    Polynom k;
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            k.push_back(1);
        } else {
            Polynom d({(-1) * x[i - 1], 1});
            k = k * d;
        }
        P = (P + k * table[0][i]);
    }
    return P;
}

double f(double x) {
    return 1/tan(x) + x;
}

int main() {
    double pi = 2*acos(0.0);
    double X = 3*pi/16;
    vector<double> x_a = {pi/8, 2*pi/8, 3*pi/8, 4*pi/8};
    vector<double> y_a;
    for (int i = 0; i < x_a.size(); ++i) {
        y_a.push_back(f(x_a[i]));
    }
    
    ofstream fout("answer.txt");
    fout << "A:\n";
    fout << "Lagrange polynom:\n";
    Polynom polynom_l_a = lagrange_polynom(x_a, y_a);
    fout << polynom_l_a;

    double res_a = 0;
    for (int i = 0; i < polynom_l_a.size(); ++i) {
        res_a += polynom_l_a[i] * pow(X, i);
    }
    fout << "f(x*) = " << res_a << "\n";
    fout << "error = " << abs(f(X) - res_a) << "\n";


    fout << "\nNewton polynom\n";
    Polynom polynom_n_a = newton_polynom(x_a, y_a);
    fout << polynom_n_a;
    res_a = 0;
    for (int i = 0; i < polynom_n_a.size(); ++i) {
        res_a += polynom_n_a[i] * pow(X, i);
    }
    fout << "f(x*) = " << res_a << "\n";
    fout << "error = " << abs(f(X) - res_a) << "\n\n";




    fout << "B:\n";
    
    vector<double> x_b = {pi/8, pi/3, 3*pi/8, pi/2};
    vector<double> y_b;
    for (int i = 0; i < x_b.size(); ++i) {
        y_b.push_back(f(x_b[i]));
    }

    fout << "Lagrange polynom:\n";
    Polynom polynom_l_b = lagrange_polynom(x_a, y_a);
    fout << polynom_l_b;

    double res_b = 0;
    for (int i = 0; i < polynom_l_b.size(); ++i) {
        res_b += polynom_l_b[i] * pow(X, i);
    }
    fout << "f(x*) = " << res_b << "\n";
    fout << "error = " << abs(f(X) - res_b) << "\n";


    fout << "\nNewton polynom:\n";
    Polynom polynom_n_b = newton_polynom(x_b, y_b);
    fout << polynom_n_b;
    res_b = 0;
    for (int i = 0; i < polynom_n_b.size(); ++i) {
        res_b += polynom_n_b[i] * pow(X, i);
    }
    fout << "f(x*) = " << res_b << "\n";
    fout << "error = " << abs(f(X) - res_b) << "\n";
    return 0;
}