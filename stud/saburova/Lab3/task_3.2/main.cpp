#include <bits/stdc++.h>
using namespace std;

vector<vector<double>> multiple_matrix(vector<vector<double>>& matrix1, vector<vector<double>>& matrix2) {
    int n1 = matrix1.size(), m1 = matrix1[0].size(), m2 = matrix2[0].size();
    vector<vector<double>> res(n1);
    for (int i=0; i<n1; i++)
        for (int j=0; j<m2; j++)
            res[i].push_back(0);

    for (int i=0; i<n1; i++) {
        for (int j=0; j<m2; j++) {
            double cntr = 0;
            for (int k=0; k<m1; k++)
               cntr += matrix1[i][k] * matrix2[k][j];
            res[i][j] = cntr;
        }
    }
    return res;
}

vector<vector<double>> tridiagonal_algorithm(vector<vector<double>>& coefficients, vector<vector<double>>& results) {
    double a, b, c, d;
    a = 0;
    b = coefficients[0][0];
    c = coefficients[0][1];
    d = results[0][0];
    vector<double> P(coefficients[0].size(), 0), Q(coefficients[0].size(), 0);

    P[0] = -c/b;
    Q[0] = d/b;
    for (int i=1; i < coefficients.size() - 1; i++){
        a = coefficients[i][i-1];
        b = coefficients[i][i];
        c = coefficients[i][i+1];
        d = results[i][0];

        P[i] = -c/(b + a*P[i-1]);
        Q[i] = (d - a*Q[i-1])/(b + a*P[i-1]);
    }

    a = coefficients[coefficients.size()-1][coefficients[0].size()-2];
    b = coefficients[coefficients.size()-1][coefficients[0].size()-1];
    c = 0;
    d = results[results.size()-1][0];

    Q[Q.size()-1] = (d - a * Q[Q.size()-2]) / (b + a * P[P.size()-2]);

    vector<vector<double>> decision(results.size());
    for(int i=0; i<decision.size(); i++)
        decision[i].push_back(0);

    decision[decision.size()-1][0] = Q[Q.size()-1];
    for (int i = decision.size()-2; i > -1; i--)
        decision[i][0] = P[i]*decision[i+1][0] + Q[i];

    return decision;
}

int main() {
    double star_x = 3.0;
    vector<double> x = {0, 1.7, 3.4, 5.1, 6.8}, y = {0.0, 3.0038, 5.2439, 7.3583, 9.4077}, h = {0.0};
    for (int i = 0; i < 4; ++i)
        h.push_back(x[i + 1] - x[i]);
    vector<vector<double>> Xdata = {{2 * (h[1] + h[2]), h[2], 0}}, Ydata = {};
    for (int i=3; i<4; i++)
        Xdata.push_back({h[i - 1], 2 * (h[i - 1] + h[i]), h[i]});
    for (int i=0; i<4-1; i++)
        Ydata.push_back({3 * ((y[i + 2] - y[i + 1]) / h[i + 2] - (y[i + 1] - y[i]) / h[i + 1])});
    Xdata.push_back({0, h[4 - 1], 2 * (h[4 - 1] + h[4])});
    vector<vector<double>> X(Xdata), Y(Ydata);
    vector<double> a(y.begin(), y.end() - 1), b, c = {0}, d;
    auto result = tridiagonal_algorithm(X, Y);
    for (auto val : result)
        c.push_back(val[0]);
    for (int i = 1; i < 4; ++i)
        b.push_back((y[i] - y[i - 1]) / h[i] - h[i] * (c[i] + 2 * c[i - 1]) / 3);
    b.push_back((y[4] - y[4 - 1]) / h[4] - 2 * h[4] * c[4 - 1] / 3);
    for (int i = 0; i < 4 - 1; ++i)
        d.push_back((c[i + 1] - c[i]) / (3 * h[i + 1]));
    d.push_back(-c[4 - 1] / (3 * h[4]));
    for (int i = 0; i < 4; ++i)
        if (x[i] <= star_x && star_x <= x[i + 1]) {
            double res = a[i] + b[i]*(star_x-x[i]) + c[i]*(star_x-x[i])*(star_x-x[i]) + d[i]*(star_x-x[i])*(star_x-x[i])*(star_x-x[i]);
            cout << "Result: " << res << endl;
            break;
        }
    return 0;
}
