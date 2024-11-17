#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

double exact_solution(double x) {
    return (-0.9783 * cos(2 * x) + 0.4776 * sin(2 * x)) / sin(x);
}

std::vector<double> f(double x, double y, double y_prime) {
    double dy = y_prime;
    double dy_prime = -2 * y_prime * (cos(x) / sin(x)) - 3 * y;
    return {dy, dy_prime};
}

void euler_method(double h, const std::vector<double>& x, std::vector<double>& y, std::vector<double>& y_prime) {
    int n = x.size();
    for (int i = 1; i < n; ++i) {
        std::vector<double> f_val = f(x[i-1], y[i-1], y_prime[i-1]);
        y[i] = y[i-1] + h * f_val[0];
        y_prime[i] = y_prime[i-1] + h * f_val[1];
    }
}

void runge_kutta_method(double h, const std::vector<double>& x, std::vector<double>& y, std::vector<double>& y_prime) {
    int n = x.size();
    for (int i = 1; i < n; ++i) {
        std::vector<double> k1 = f(x[i-1], y[i-1], y_prime[i-1]);
        std::vector<double> k2 = f(x[i-1] + h/2, y[i-1] + h/2 * k1[0], y_prime[i-1] + h/2 * k1[1]);
        std::vector<double> k3 = f(x[i-1] + h/2, y[i-1] + h/2 * k2[0], y_prime[i-1] + h/2 * k2[1]);
        std::vector<double> k4 = f(x[i-1] + h, y[i-1] + h * k3[0], y_prime[i-1] + h * k3[1]);
        
        y[i] = y[i-1] + h/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
        y_prime[i] = y_prime[i-1] + h/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
    }
}

void adams_bashforth_moulton_method(double h, const std::vector<double>& x, std::vector<double>& y, std::vector<double>& y_prime) {
    int n = x.size();
    std::vector<double> f0, f1, f2, f3;
    
    for (int i = 1; i <= 3; ++i) {
        std::vector<double> k1 = f(x[i-1], y[i-1], y_prime[i-1]);
        std::vector<double> k2 = f(x[i-1] + h/2, y[i-1] + h/2 * k1[0], y_prime[i-1] + h/2 * k1[1]);
        std::vector<double> k3 = f(x[i-1] + h/2, y[i-1] + h/2 * k2[0], y_prime[i-1] + h/2 * k2[1]);
        std::vector<double> k4 = f(x[i-1] + h, y[i-1] + h * k3[0], y_prime[i-1] + h * k3[1]);
        
        y[i] = y[i-1] + h/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
        y_prime[i] = y_prime[i-1] + h/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
    }
    
    for (int i = 4; i < n; ++i) {
        f0 = f(x[i-1], y[i-1], y_prime[i-1]);
        f1 = f(x[i-2], y[i-2], y_prime[i-2]);
        f2 = f(x[i-3], y[i-3], y_prime[i-3]);
        f3 = f(x[i-4], y[i-4], y_prime[i-4]);
        
        double yp = y[i-1] + h/24 * (55*f0[0] - 59*f1[0] + 37*f2[0] - 9*f3[0]);
        double y_prime_p = y_prime[i-1] + h/24 * (55*f0[1] - 59*f1[1] + 37*f2[1] - 9*f3[1]);
        
        std::vector<double> fp = f(x[i], yp, y_prime_p);
        y[i] = y[i-1] + h/24 * (9*fp[0] + 19*f0[0] - 5*f1[0] + f2[0]);
        y_prime[i] = y_prime[i-1] + h/24 * (9*fp[1] + 19*f0[1] - 5*f1[1] + f2[1]);
    }
}

double runge_romberg(double y_h1, double y_h2, int p) {
    return (y_h1 - y_h2) / (std::pow(2, p) - 1);
}

int main() {
    double h = 0.1;
    double h2 = h / 2;
    double a = 1.0;
    double b = 2.0;
    int n = static_cast<int>((b - a) / h) + 1;
    int n2 = static_cast<int>((b - a) / h2) + 1;
    
    std::vector<double> x(n), y_euler(n), y_prime_euler(n);
    std::vector<double> y_rk(n), y_prime_rk(n);
    std::vector<double> y_adams(n), y_prime_adams(n);
    
    std::vector<double> x2(n2), y_euler2(n2), y_prime_euler2(n2);
    std::vector<double> y_rk2(n2), y_prime_rk2(n2);
    std::vector<double> y_adams2(n2), y_prime_adams2(n2);
    
    x[0] = x2[0] = a;
    y_euler[0] = y_rk[0] = y_adams[0] = y_euler2[0] = y_rk2[0] = y_adams2[0] = 1.0;
    y_prime_euler[0] = y_prime_rk[0] = y_prime_adams[0] = y_prime_euler2[0] = y_prime_rk2[0] = y_prime_adams2[0] = 1.0;
    
    for (int i = 1; i < n; ++i) {
        x[i] = x[i-1] + h;
    }
    
    for (int i = 1; i < n2; ++i) {
        x2[i] = x2[i-1] + h2;
    }
    
    euler_method(h, x, y_euler, y_prime_euler);
    runge_kutta_method(h, x, y_rk, y_prime_rk);
    adams_bashforth_moulton_method(h, x, y_adams, y_prime_adams);
    
    euler_method(h2, x2, y_euler2, y_prime_euler2);
    runge_kutta_method(h2, x2, y_rk2, y_prime_rk2);
    adams_bashforth_moulton_method(h2, x2, y_adams2, y_prime_adams2);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "x\tEuler\t\tRK4\t\tAdams\t\tExact\t\tError (Euler)\tError (RK4)\tError (Adams)\n";
    
    for (int i = 0; i < n; ++i) {
        double exact = exact_solution(x[i]);
        std::cout << x[i] << "\t" 
                  << y_euler[i] << "\t" << y_rk[i] << "\t" << y_adams[i] << "\t" << exact << "\t"
                  << std::abs(y_euler[i] - exact) << "\t" << std::abs(y_rk[i] - exact) << "\t" << std::abs(y_adams[i] - exact) << "\n";
    }
    
    std::cout << "\nRunge-Romberg Error Estimation for Euler's Method\n";
    
    std::cout << "x\tEuler (h=0.1)\tEuler (h=0.05)\tError Estimate (Euler)\n";
    for (int i = 0; i < n; ++i) {
        double y_h2 = y_euler2[i * 2];
        double error_estimate = runge_romberg(y_euler[i], y_h2, 1);
        std::cout << x[i] << "\t" << y_euler[i] << "\t\t" << y_h2 << "\t\t" << error_estimate << "\n";
    }
    
    std::cout << "\nRunge-Romberg Error Estimation for RK4 Method\n";
    
    std::cout << "x\tRK4 (h=0.1)\tRK4 (h=0.05)\tError Estimate (RK4)\n";
    for (int i = 0; i < n; ++i) {
        double y_h2 = y_rk2[i * 2];
        double error_estimate = runge_romberg(y_rk[i], y_h2, 4);
        std::cout << x[i] << "\t" << y_rk[i] << "\t\t" << y_h2 << "\t\t" << error_estimate << "\n";
    }
    
    std::cout << "\nRunge-Romberg Error Estimation for Adams Method\n";
    
    std::cout << "x\tAdams (h=0.1)\tAdams (h=0.05)\tError Estimate (Adams)\n";
    for (int i = 0; i < n; ++i) {
        double y_h2 = y_adams2[i * 2];
        double error_estimate = runge_romberg(y_adams[i], y_h2, 4);
        std::cout << x[i] << "\t" << y_adams[i] << "\t\t" << y_h2 << "\t\t" << error_estimate << "\n";
    }

    
    return 0;
}
