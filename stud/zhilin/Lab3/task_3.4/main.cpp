#include <bits/stdc++.h>


using namespace std;


int main() {
    double x_marked = 0.2;
    vector<double> x = {0.0, 0.1, 0.2, 0.3, 0.4}, y = {1.0, 1.1052, 1.2214, 1.3499, 1.4918};

    int n = x.size();
    vector<double> derivative_first;
    for (int i = 0; i < n - 1; ++i) {
        derivative_first.push_back((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
    }

    vector<double> derivative_second;
    for (int i = 0; i < n - 2; ++i) {
        derivative_second.push_back(2 * ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i])) / (x[i + 2] - x[i]));
    }

    for (int i = 0; i < n - 1; ++i) {
        if (x[i] == x_marked) {
            cout << "The left-hand first derivative: " << derivative_first[i - 1] << endl;
            cout << "The right-hand first derivative: " << derivative_first[i] << endl;
            break;
        } else if (x[i] < x_marked && x_marked < x[i + 1]) {
            cout << "First derivative: " << derivative_first[i] << endl;
        }
    }

    cout << endl;

    for (int i = 0; i < n - 2; ++i) {
        if (x[i] <= x_marked && x_marked <= x[i + 1]) {
            cout << "Second derivative: " << derivative_second[i] << endl;
            break;
        }
    }

    return 0;
}
