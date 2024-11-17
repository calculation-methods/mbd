#include <iostream>
#include <vector>

using namespace std;

vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
vector<double> y = {1.0, 2.6931, 4.0986, 5.3863, 6.6094};
double X_star = 3.0;

double first_derivative(vector<double>& x, vector<double>& y, double X_star) {
    int n = x.size();
    double h = x[1] - x[0];

    int idx = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] < X_star)
            idx = i;
        else
            break;
    }

    double first_derivative = (y[idx + 1] - y[idx]) / h;

    return first_derivative;
}

double second_derivative(vector<double>& x, vector<double>& y, double X_star) {
    int n = x.size();
    double h = x[1] - x[0];

    int idx = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] < X_star)
            idx = i;
        else
            break;
    }

    double second_derivative = (y[idx + 2] - 2 * y[idx + 1] + y[idx]) / (h * h);

    return second_derivative;
}

int main() {
    double first = first_derivative(x, y, X_star);
    double second = second_derivative(x, y, X_star);

    cout << "First derivative at x = " << X_star << ": " << first << endl;
    cout << "Second derivative at x = " << X_star << ": " << second << endl;

    return 0;
}
