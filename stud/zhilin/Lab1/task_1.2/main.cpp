#include <bits/stdc++.h>

using namespace std;
using matrix = vector<vector<double> >;


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
    matrix coefficient_matrix{
        {8, -4, 0, 0, 0},
        {-2, 12, -7, 0, 0},
        {0, 2, -9, 1, 0},
        {0, 0, -8, 17, -4},
        {0, 0, 0, -7, 13}
    };

    matrix equation_roots = {
        {32},
        {15},
        {-10},
        {133},
        {-76}
    };

    matrix res_matrix = tridiagonal_algorithm(coefficient_matrix, equation_roots);

    cout << "Desicion" << endl;
    for (int i=0; i<res_matrix.size(); i++)
        printf("x%d = %f \n", i+1, res_matrix[i][0]);

    return 0;
}