#include <iostream>
#include <vector>
#include "math.h"
#include "runge_kutt.h"
#include "funs.h"

#include "adams.h"
#include "euler.h"

using namespace std;





void error(const vector<double>& numeric, double x0, double h, int steps) {
    double maxError = 0.0;
    for (int i = 0; i <= steps; ++i) {
        double exact = F(x0 + i * h);
        double error = fabs(numeric[i] - exact);
        if (error > maxError) {
            maxError = error;
        }
    }
    cout << "Error estimation by comparison method: " << maxError << '\n';
}

int main() {
    double h = 0.1;
    double x0 = 1.0;
    double y10 = 2 + exp(1);
    double y20 = 1;
    int steps = (int)((2.0 - 1.0) / h);
    Euler euler = Euler();
    vector<double> eulerResult = euler.result(h, x0, y10, y20, steps);
    error(eulerResult, x0, h, steps);
    RungeKutt rungeKutt = RungeKutt();

    vector<double> rungeKuttaResult = rungeKutt.result(h, x0, y10, y20, steps);
    error(rungeKuttaResult, x0, h, steps);

    Adams adams = Adams();

    vector<double> adamsResult = adams.result(h, x0, y10, y20, steps);
    error(adamsResult, x0, h, steps);

    return 0;
}