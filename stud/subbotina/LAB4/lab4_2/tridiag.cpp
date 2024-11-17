#include <iostream>
#include <vector>

template <class T>
class Tridiag {
private:
    const T EPS = 1e-6;

    int n;
    std::vector<T> a, b, c;
public:
    Tridiag(const int& size) : n(size), a(n), b(n), c(n) {}
    Tridiag(std::vector<double>& _a, std::vector<double>& _b, std::vector<double>& _c) : n(_a.size()), a(_a), b(_b), c(_c) {};

    friend std::istream& operator >> (std::istream& in, Tridiag<T>& t) {
        in >> t.b[0] >> t.c[0];
        for (int i = 1; i < t.n - 1; ++i) {
            in >> t.a[i] >> t.b[i] >> t.c[i];
        }
        in >> t.a.back() >> t.b.back();
        return in;
    }

    std::vector<T> Solve(const std::vector<T>& d) {
        std::vector<T> p(n);
        p[0] = -c[0] / b[0];
        std::vector<T> q(n);
        q[0] = d[0] / b[0];
        for (int i = 1; i < n; ++i) {
            p[i] = -c[i] / (b[i] + a[i] * p[i - 1]);
            q[i] = (d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1]);
        }
        std::vector<T> x(n);
        x.back() = q.back();
        for (int i = n - 2; i >= 0; --i) {
            x[i] = p[i] * x[i + 1] + q[i];
        }
        return x;
    }
    ~Tridiag() = default;
};
