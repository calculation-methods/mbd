#include <bits/stdc++.h>

using namespace std;


double rectangular_method(function<double(double)> f, double x_start, double x_end, double step){
    double x = x_start, res = 0;
    while (x < x_end){
        res += f((2*x + step)/2);
        x += step;
    }
    return step*res;
}


double trapeze_method(function<double(double)> f, double x_start, double x_end, double step){
    double x = x_start+step, res = f(x_start)/2 + f(x_end)/2;
    while (x < x_end){
        res += f(x);
        x += step;
    }
    return step * res;
}


double simpson_method(function<double(double)> f, double x_start, double x_end, double step){
    double x = x_start + step, res = f(x_start) + f(x_end);
    bool flag = true;
    while (x < x_end){
        res += f(x) * ((flag) ? 4 : 2);
        x += step;
        flag = !flag;
    }
    return step * res / 3;
}


double RRR_estimation(double F_1, double F_2, double step_1, double step_2, double p){
    return F_1 + (F_1 - F_2)/(pow((step_2/step_1), p) - 1);
}


int main() {
    auto y = [](double x) { return x / (x*x + 9); };
    double x_0 = 0, x_k = 2;
    vector<double> precision = {0.5, 0.25, 0.000001};

    // auto y = [](double x) { return x / ((3*x + 4)*(3*x + 4)); };
    // double x_0 = -1, x_k = 1;
    // vector<double> precision = {0.5, 0.25, 0.000001};

    vector<pair<string, function<double(function<double(double)>, double, double, double)>>> methods = {
        {"Rectangular", rectangular_method},
        {"Trapeze", trapeze_method},
        {"Simpson", simpson_method}
    };

    for (auto& method : methods) {
        cout << method.first << " method" << endl;
        vector<double> F;
        for (auto h : precision) {
            F.push_back(method.second(y, x_0, x_k, h));
            cout << "\tIntegral = " << F.back() << "\t\t(Step = " << h << ")" << endl;
        }
        cout << "\tIntegral = " << RRR_estimation(F[0], F[1], precision[0], precision[1], 2) << "\t\t(Runge-Romberg-Richardson estimation, step 1 = " << precision[0] << ", step 2 = " << precision[1] << ")" << endl;
        cout << "\n";
    }
    return 0;
}