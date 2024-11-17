#include <bits/stdc++.h>

using namespace std;
using matrix = vector<vector<double> >;

matrix multiple_matrix(matrix& matrix1, matrix& matrix2) {
    int n1 = matrix1.size(), m1 = matrix1[0].size(), m2 = matrix2[0].size();
    matrix res(n1);
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

matrix tridiagonal_algorithm(matrix& coefficients, matrix& results) {
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

    matrix decision(results.size());
    for(int i=0; i<decision.size(); i++)
        decision[i].push_back(0);

    decision[decision.size()-1][0] = Q[Q.size()-1];
    for (int i = decision.size()-2; i > -1; i--)
        decision[i][0] = P[i]*decision[i+1][0] + Q[i];

    return decision;
}


void print_matrix(matrix& matrix1) {
    for(const auto& vect: matrix1) {
        for (auto x: vect)
            cout << x << " ";
        cout << endl;
    }
}

int main() {
    double x_marked = 0.1;
    vector<double> x = {-0.4, -0.1, 0.2, 0.5, 0.8};
    vector<double> y = {1.9823, 1.6710, 1.3694, 1.0472, 0.64350};

    vector<double> h = {0.0};
    for (int i = 0; i < 4; ++i) {
        h.push_back(x[i + 1] - x[i]);
    }
    int n = x.size() - 1;

    vector<vector<double>> matr_data = {{2 * (h[1] + h[2]), h[2], 0}}, root_data = {};

    for (int i=3; i<n; i++)
        matr_data.push_back({h[i - 1], 2 * (h[i - 1] + h[i]), h[i]});

    for (int i=0; i<n-1; i++)
        root_data.push_back({3 * ((y[i + 2] - y[i + 1]) / h[i + 2] - (y[i + 1] - y[i]) / h[i + 1])});

    matr_data.push_back({0, h[n - 1], 2 * (h[n - 1] + h[n])});

    matrix matr(matr_data);
    matrix root(root_data);

    vector<double> coeff_a(y.begin(), y.end() - 1);
    vector<double> coeff_c = {0};
    auto result = tridiagonal_algorithm(matr, root);
    for (auto val : result) {
        coeff_c.push_back(val[0]);
    }
    vector<double> coeff_b;
    for (int i = 1; i < n; ++i) {
        coeff_b.push_back((y[i] - y[i - 1]) / h[i] - h[i] * (coeff_c[i] + 2 * coeff_c[i - 1]) / 3);
    }
    coeff_b.push_back((y[n] - y[n - 1]) / h[n] - 2 * h[n] * coeff_c[n - 1] / 3);

    vector<double> coeff_d;
    for (int i = 0; i < n - 1; ++i) {
        coeff_d.push_back((coeff_c[i + 1] - coeff_c[i]) / (3 * h[i + 1]));
    }
    coeff_d.push_back(-coeff_c[n - 1] / (3 * h[n]));

    bool flag = false;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= x_marked && x_marked <= x[i + 1]) {
            double res = coeff_a[i] + coeff_b[i]*(x_marked-x[i]) + coeff_c[i]*(x_marked-x[i])*(x_marked-x[i]) + coeff_d[i]*(x_marked-x[i])*(x_marked-x[i])*(x_marked-x[i]);
            cout << "Result = " << res << endl;
            flag = true;
            break;
        }
    }
    
    if (flag)
        cout << "Incorrect value" << endl;

    return 0;
}
