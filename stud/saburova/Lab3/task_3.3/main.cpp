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

pair<vector<vector<double>>, vector<vector<double>>> lu_decomposition(vector<vector<double>>& coefficients, vector<vector<double>>& results) {
    int n1=coefficients.size(), m1=coefficients[0].size(), m2 = results[0].size();
    vector<vector<double>> L(n1), U = coefficients;
    for (int i=0; i<n1; i++)
        for (int j=0; j<m1; j++)
            L[i].push_back(0);

    for (int k=0; k<n1; k++) {
        if (U[k][k] == 0) {
            for (int i=k+1; i<n1; i++) {
                if (U[i][k] != 0) {
                    swap(U[k], U[i]);
                    swap(L[k], L[i]);
                    swap(coefficients[k], coefficients[i]);
                    swap(results[k], results[i]);
                    break;
                }
            }
        }
        L[k][k] = 1;
        for (int i=k+1; i<n1; i++) {
            L[i][k] = U[i][k]/U[k][k];
            if (U[i][k] == 0)
                continue;
            for(int j=k; j<m1; j++)
                U[i][j] -= L[i][k]*U[k][j];

        }
    }

    return make_pair(L, U);
}

vector<vector<double>> calculate_decisions(vector<vector<double>>& coefficients, vector<vector<double>>& results) {
    auto [L, U] = lu_decomposition(coefficients, results);
    vector<vector<double>> res = results;

    for (int k=0; k<res[0].size(); k++)
        for (int i=0; i<res.size(); i++)
            for (int j=0; j<i; j++)
                res[i][k] -= res[j][k]*L[i][j];
    for (int k=0; k<res[0].size(); k++) {
        for (int i=coefficients.size()-1; i>-1; i--) {
            for (int j=i+1; j<results.size(); j++) {
                res[i][k] -= res[j][k]*U[i][j];
            }
            res[i][k] /= U[i][i];
        }
    }

    return res;
}

int main() {
    vector<double> x = {0.0, 1.7, 3.4, 5.1, 6.8, 8.5}, y = {0.0, 3.0038, 5.2439, 7.3583, 9.4077, 11.415};
    double element_one = 0, element_two = 0, element_three = 0, element_four = 0, element_five = 0, element_six = 0, element_seven = 0;
    for (int i=0; i<6; i++){
        element_one += x[i];
        element_two += x[i]*x[i];
        element_three += x[i]*x[i]*x[i];
        element_four += x[i]*x[i]*x[i]*x[i];
        element_five += y[i];
        element_six += y[i]*x[i];
        element_seven += y[i]*x[i]*x[i];
    }

    vector<vector<double>> X = {
        {6.0, element_one},
        {element_one, element_two}
    };
    vector<vector<double>> Y = {
        {element_five},
        {element_six}
    };
    vector<vector<double>> coeffs = calculate_decisions(X, Y);
    cout << "Coefficents " << coeffs[0][0] << " " << coeffs[1][0] << endl;
    double loss = 0;
    for (int i = 0; i < 6; i++)
        loss += pow(coeffs[0][0] + coeffs[1][0] * x[i] - y[i], 2);
    cout << "Loss = " << loss << endl << endl;

    X = {
        {6, element_one, element_two},
        {element_one, element_two, element_three},
        {element_two, element_three, element_four}
    };
    Y = {
        {element_five},
        {element_six},
        {element_seven}
    };
    coeffs = calculate_decisions(X, Y);
    cout << "Coefficents " << coeffs[0][0] << " " << coeffs[1][0] << " " << coeffs[2][0] << endl;
    loss = 0;
    for (int i = 0; i < 6; i++)
        loss += pow(coeffs[0][0] + coeffs[1][0] * x[i] + coeffs[2][0] * pow(x[i], 2) - y[i], 2);
    cout << "Loss = " << loss << endl;

    return 0;
}