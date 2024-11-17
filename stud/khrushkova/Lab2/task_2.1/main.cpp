#include <bits/stdc++.h>

using namespace std;


double f(const double x){
    return pow(x, 6) - 5 * pow(x, 3) - 2;
}

double df (const double x) {
    return 6 * pow(x, 5) - 15 * pow(x, 2);
}

double eps (const double val1, const double val2, double q) {
    if (q == -1)
        return abs(val1 - val2);
    return q*abs(val1 - val2)/(1-q);
}

double fi(double x) {
    return pow(5*pow(x, 3)+2, 1.0/6);
}

pair<double, int> newton(double start_value, double EPS){
    double current_val = start_value - f(start_value)/df(start_value), previous_val = start_value;
    int counter=0;

    while (eps(previous_val, current_val, -1) >= EPS){
        counter++;
        previous_val = current_val;
        current_val = current_val - f(current_val)/df(current_val);
    }
    return {current_val, counter};
}

pair<double, int> simple_iter(double start_value, double q, double EPS){
    double current_val = start_value, previous_val = start_value*5;
    int counter=0;

    while (eps(previous_val, current_val, q) >= EPS){
        counter++;
        previous_val = current_val;
        current_val = fi(current_val);
    }
    return {current_val, counter};
}


int main(){
    double epsillon = 0.00001, res;
    int counter;

    tie(res, counter) = newton(2, epsillon);
    cout << endl << "Newton method" << endl << "Result = " << res << endl << "Iteration count = " << counter << endl << endl;

    tie(res, counter) = simple_iter(1.7, 0.6, epsillon);
    cout << "Simple iterations method" << endl << "Result = " << res << endl << "Iteration count = " << counter << endl << endl;

    return 1;
}
