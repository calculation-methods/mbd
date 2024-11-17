#include <bits/stdc++.h>
using namespace std;

// Функция из лабораторной 1.2
vector<vector<double>> tridiagonal(vector<vector<double>>& coefficients, vector<vector<double>>& results) {
    int n = coefficients.size();
    double a = 0, b = coefficients[0][0], c = coefficients[0][1], d = results[0][0];
    vector<double> P(n, 0), Q(n, 0);

    P[0] = -c/b;
    Q[0] = d/b;
    for (int i=1; i < n-1; i++){
        a = coefficients[i][i-1];
        b = coefficients[i][i];
        c = coefficients[i][i+1];
        d = results[i][0];

        P[i] = -c/(b + a*P[i-1]);
        Q[i] = (d - a*Q[i-1])/(b + a*P[i-1]);
    }

    a = coefficients[n-1][n-2];
    b = coefficients[n-1][n-1];
    c = 0;
    d = results[n-1][0];

    Q[n-1] = (d - a * Q[n-2]) / (b + a * P[n-2]);

    vector<vector<double>> result(n);
    for(int i=0; i<n; i++)
        result[i].push_back(0);

    result[n-1][0] = Q[n-1];
    for (int i = n-2; i > -1; i--)
        result[i][0] = P[i]*result[i+1][0] + Q[i];

    return result;
}

int main() {
    double star_x = 1.5;
    vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}, y = {0.0, 2.6180, 0.90690, 1.5708, 1.3090}, h = {0.0};

    for (int i = 0; i < 4; ++i)
        h.push_back(x[i + 1] - x[i]);
    
    vector<vector<double>> X = {{2 * (h[1] + h[2]), h[2], 0}}, Y = {};
    for (int i=3; i<4; i++)
        X.push_back({h[i - 1], 2 * (h[i - 1] + h[i]), h[i]});
    for (int i=0; i<3; i++)
        Y.push_back({3 * ((y[i + 2] - y[i + 1]) / h[i + 2] - (y[i + 1] - y[i]) / h[i + 1])});
    X.push_back({0, h[3], 2 * (h[3] + h[4])});
    
    vector<double> a(y.begin(), y.end() - 1), b, c = {0}, d;
    auto result = tridiagonal(X, Y);

    for (auto val : result)
        c.push_back(val[0]);
    
    for (int i = 1; i < 4; ++i)
        b.push_back((y[i] - y[i - 1]) / h[i] - h[i] * (c[i] + 2 * c[i - 1]) / 3);
    b.push_back((y[4] - y[3]) / h[4] - 2 * h[4] * c[3] / 3);
    
    for (int i = 0; i < 3; ++i)
        d.push_back((c[i + 1] - c[i]) / (3 * h[i + 1]));
    d.push_back(-c[3] / (3 * h[4]));
    
    for (int i = 0; i < 4; ++i)
        if (x[i] <= star_x && star_x <= x[i + 1]) {
            double res = a[i] + b[i]*(star_x-x[i]) + c[i]*(star_x-x[i])*(star_x-x[i]) + d[i]*(star_x-x[i])*(star_x-x[i])*(star_x-x[i]);
            cout << "Result: " << res << endl;
            break;
        }
    return 0;
}
