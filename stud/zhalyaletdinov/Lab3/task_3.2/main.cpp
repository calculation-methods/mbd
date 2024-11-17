#include <bits/stdc++.h>

using namespace std;

vector<vector<double>> get_res(vector<vector<double>>& coefficients, vector<vector<double>>& results) {
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
    vector<double> x = {-0.4, -0.1, 0.2, 0.5, 0.8}, y = {-0.41152, -0.10017, 0.20136, 0.52360, 0.92730}, h = {0.0};
    double t = 0.1;

    for (int i = 0; i < 4; ++i)
        h.push_back(x[i + 1] - x[i]);

    vector<vector<double>> X_content = {{2 * (h[1] + h[2]), h[2], 0}}, Y_content = {};

    for (int i=3; i<4; i++)
        X_content.push_back({h[i - 1], 2 * (h[i - 1] + h[i]), h[i]});

    for (int i=0; i<4-1; i++)
        Y_content.push_back({3 * ((y[i + 2] - y[i + 1]) / h[i + 2] - (y[i + 1] - y[i]) / h[i + 1])});

    X_content.push_back({0, h[3], 2 * (h[3] + h[4])});

    vector<vector<double>> X(X_content), Y(Y_content);

    vector<double> k_a(y.begin(), y.end() - 1);
    vector<double> k_c = {0};
    auto result = get_res(X, Y);
    for (auto val : result) {
        k_c.push_back(val[0]);
    }
    vector<double> k_b;
    for (int i = 1; i < 4; ++i) {
        k_b.push_back((y[i] - y[i - 1]) / h[i] - h[i] * (k_c[i] + 2 * k_c[i - 1]) / 3);
    }
    k_b.push_back((y[4] - y[3]) / h[4] - 2 * h[4] * k_c[3] / 3);

    vector<double> k_d;
    for (int i = 0; i < 3; ++i) {
        k_d.push_back((k_c[i + 1] - k_c[i]) / (3 * h[i + 1]));
    }
    k_d.push_back(-k_c[3] / (3 * h[4]));

    bool marker = false;
    for (int i = 0; i < 4; ++i) {
        if (x[i] <= t && t <= x[i + 1]) {
            double res = k_a[i] + k_b[i]*(t-x[i]) + k_c[i]*(t-x[i])*(t-x[i]) + k_d[i]*(t-x[i])*(t-x[i])*(t-x[i]);
            cout << "Ответ " << res << endl;
            marker = true;
            break;
        }
    }

    return 0;
}
