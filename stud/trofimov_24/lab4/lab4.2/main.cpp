#include <iostream>
#include <vector>
#include <iomanip>
#include "shooting.h"
#include "finite_diff.h"
#include "funcs.h"


using namespace std;




int main() {
    cout << fixed << setprecision(6);
    int N = 100;
    double x_end = 1.0;

    double s_guess = -1;
    auto shooting = Shooting();
    auto sol_shooting = shooting.result(s_guess, x_end, N);

    auto shooting_2N = Shooting();
    auto sol_shooting_2N = shooting_2N.result(s_guess, x_end, 2 * N);

    auto fd = FiniteDifference();
    auto sol_fd = fd.result(N);

    auto fd_2N = FiniteDifference();
    auto sol_fd_2N = fd_2N.result(2 * N);

    vector<double> x(N + 1);
    vector<double> exact(N + 1);
    for (int i = 0; i <= N; ++i) {
        x[i] = i * 1.0 / N;
        exact[i] = f(x[i]);
    }

    double error_fd = rungeRomberg(sol_fd_2N, sol_fd, N);
    vector<double> y_shooting(N + 1);
    vector<double> y_shooting_2N(2 * N + 1);
    for (int i = 0; i <= N; ++i) {
        y_shooting[i] = sol_shooting[i][0];
    }
    for (int i = 0; i <= 2 * N; ++i) {
        y_shooting_2N[i] = sol_shooting_2N[i][0];
    }
    double error_shooting = rungeRomberg(y_shooting_2N, y_shooting, N);

    cout << "Shooting method:\n";
    cout << "x = " << x[N] << ", y = " << sol_shooting[N][0] << ", exact y = " << exact[N] << "\n";

    cout << "Shooting method error (Runge-Romberg method): " << error_shooting << "\n";

    cout << "\nFinite difference method:\n";
    cout << "x = " << x[N] << ", y = " << sol_fd[N] << ", exact y = " << exact[N];

    cout << "\nError of the finite difference method (Runge-Romberg method):  " << error_fd << "\n";

    return 0;
}