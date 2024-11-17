#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

using namespace std;

double function(double x, double y, double z) {
    return -4*x*z - (4*x*x + 2) * y;
}

double accurate_function(double x) {
    return (1+x) * exp(-x*x);
}

double p(double x) {
    return 4*x;
}

double q(double x) {
    return (4*x*x + 2);
}

vector<double> Runge_Kutty(double a, double b, double h, double y0, double z0) {
    int N = ((b - a) / h) + 1;
    vector<double> x(N), y(N), z(N);
    vector<double> K(4,0), L(4,0);

    x[0] = a;
    y[0] = y0;
    z[0] = z0;

    for (int i = 1; i < N; ++i) {
        x[i] = a + i * h;
        for (int j = 1; j < 4; ++j) {
            K[0] = h * z[i - 1];
            L[0] = h * function(x[i - 1], y[i - 1], z[i - 1]);
            K[j] = h * (z[i - 1] + L[j - 1] / 2);
            L[j] = h * function(x[i - 1] + h / 2, y[i - 1] + K[j - 1] / 2, z[i - 1] + L[j - 1] / 2);
        }
        double deltay = (K[0] + 2 * K[1] + 2 * K[2] + K[3]) / 6;
        double deltaz = (L[0] + 2 * L[1] + 2 * L[2] + L[3]) / 6;
        y[i] = y[i - 1] + deltay;
        z[i] = z[i - 1] + deltaz;
    }
    return y;
}

vector<double> shooting_method(double a, double b, double h, double e, double y0, double y1) {
    double nu1 = 1.0, nu2 = 0.8;
    double f1, f2;

    f1 = Runge_Kutty(a, b, h, y0, nu1).back() - y1;
    f2 = Runge_Kutty(a, b, h, y0, nu2).back() - y1;

    while (std::abs(f2) > e) {
        double temp = nu2;
        nu2 = nu2 - f2 * (nu2 - nu1) / (f2 - f1);
        nu1 = temp;
        f1 = f2;
        f2 = Runge_Kutty(a, b, h, y0, nu2).back() - y1;
    }
    return Runge_Kutty(a, b, h, y0, nu2);
}

vector<double> finite_difference_method(double a, double b, double h, double alfa, double beta, double delta, double gamma, double y0, double y1) {
    int N = ((b - a) / h); 
    vector<double> x(N), A, B, C, D, P(N), Q(N), ans(N);

    x[0] = a;
    x[1] = a + h;
    A.push_back(0);
    B.push_back(-2 + h * h * q(x[1]));
    C.push_back(1 + p(x[1]) * h / 2);
    D.push_back(-(1 - (p(x[1]) * h) / 2) * y0);
    for (int i = 2; i < N; ++i) {
        x[i] = a + i * h;
        A.push_back(1 - p(x[i]) * h / 2);
        B.push_back(-2 + h * h * q(x[i]));
        C.push_back(1 + p(x[i]) * h / 2);
        D.push_back(0);
    }
    A.push_back(1 - p(x[N - 2]) * h / 2);
    B.push_back(-2 + h * h * q(x[N - 2]));
    C.push_back(0);
    D.push_back(-(1 + (p(x[N - 2]) * h) / 2) * y1);

    P[0] = (-C[0] / B[0]);
    Q[0] = (D[0] / B[0]);
    for (int i = 1; i <= N; ++i) {
        P[i] = (-C[i] / (B[i] + A[i] * P[i - 1]));
        Q[i] = ((D[i] - A[i] * Q[i - 1]) / (B[i] + A[i] * P[i - 1]));
    }

    ans[N-1] = Q[N-1];
    for (int i = N - 2; i > 0; --i)
        ans[i] = P[i] * ans[i + 1] + Q[i];
    ans[0] = y0;
    ans[N] = y1;
    return ans;
}

pair<vector<double>, vector<double>> RRR_method(double a, double b, double h, double e, double y0, double y1, double alfa, double beta, double gamma, double delta) {
    vector<double> shooting_method1, finite_difference_method1;
    vector<double> shooting_method_norm = shooting_method(a, b, h, e, y0, y1);
    vector<double> shooting_method_half = shooting_method(a, b, h / 2, e, y0, y1);
    vector<double> finite_difference_method_norm = finite_difference_method(a, b, h, alfa, beta, delta, gamma, y0, y1);
    vector<double> finite_difference_method_half = finite_difference_method(a, b, h / 2, alfa, beta, delta, gamma, y0, y1);
    int N = static_cast<int>((b - a) / h);

    shooting_method1.resize(N + 1);
    finite_difference_method1.resize(N + 1);

    for (int i = 0; i <= N; ++i) {
        shooting_method1[i] = shooting_method_norm[i] + (shooting_method_half[2 * i] - shooting_method_norm[i]) / (1 - 0.5 * 0.5);
        finite_difference_method1[i] = finite_difference_method_norm[i] + (finite_difference_method_half[2 * i] - finite_difference_method_norm[i]) / (1 - 0.5 * 0.5);
    }
    return make_pair(shooting_method1, finite_difference_method1);
}



int main() {
    double a = 0, b = 2, h = 0.2, e = 0.001, alfa = 0, beta = 1, delta = -1, gamma = 4;
    double y0 = 1, y1 = 3 * exp(-4);
    ofstream fout("answer2.txt");
    vector<double> x, y, y_exact, shooting, finite_difference;
    vector<double> shooting_method_result, finite_difference_method_result;

    int N = ((b - a) / h);
    x.resize(N + 1);
    y.resize(N + 1);
    y_exact.resize(N + 1);
    fout.precision(3);
    fout << fixed;
    for (int i = 0; i <= N; ++i) {
        x[i] = a + i * h;
        y_exact[i] = accurate_function(x[i]);
    }

    shooting_method_result = shooting_method(a, b, h, e, y0, y1);
    finite_difference_method_result = finite_difference_method(a, b, h, alfa, beta, delta, gamma, y0, y1);
    tie(shooting, finite_difference) = RRR_method(a, b, h, e, y0, y1, alfa, beta, delta, gamma);

    fout << "X:" << endl;
    for (int i = 0; i <= N; ++i)
        fout << x[i] << "\t";
    fout << endl;

    fout << "Accurate value Y:"<< endl;
    for (int i = 0; i <= N; ++i)
        fout << y_exact[i] << "\t";
    fout << endl << endl;

    fout << "Shooting method:"<< endl;
    for (int i = 0; i <= N; ++i)
        fout << shooting_method_result[i] << "\t";
    fout << endl;

    fout << "Finite_difference_method:"<< endl;
    for (int i = 0; i <= N; ++i)
        fout << finite_difference_method_result[i] << "\t";
    fout << endl << endl;

    fout << "With Runge-Romberg-Richardson:" << endl;
    fout << "Shooting method:"<< endl;
    for (int i = 0; i <= N; ++i)
        fout << shooting[i] << "\t";
    fout << endl;

    fout << "Finite_difference_method:"<< endl;
    for (int i = 0; i <= N; ++i)
        fout << finite_difference[i] << "\t";
    fout << endl << endl;

    fout << "Difference from accurate value" << endl;
    fout << "Shooting method:"<< endl;
    for (int i = 0; i <= N; ++i)
        fout << abs(shooting[i] - y_exact[i]) << "\t";
    fout << endl;

    fout << "Finite_difference_method:"<< endl;
    for (int i = 0; i <= N; ++i)
        fout << abs(finite_difference[i] - y_exact[i]) << "\t";
    fout << endl;

    return 0;
}
