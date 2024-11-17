#include <bits/stdc++.h>

using namespace std;
using matrix = vector<vector<double> >;


matrix multiply_matrix(const matrix& matrix1, const matrix& matrix2) {
    int n1 = matrix1.size();
    int m1 = matrix1[0].size();
    int n2 = matrix2.size();
    int m2 = matrix2[0].size();

    if (m1 != n2) {
        cout << "Incorrect shapes of matrices" << endl;
        return matrix();
    }

    matrix res(n1, vector<double>(m2, 0));

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < m2; ++j) {
            double cntr = 0;
            for (int k = 0; k < m1; ++k) {
                cntr += matrix1[i][k] * matrix2[k][j];
            }
            res[i][j] = cntr;
        }
    }

    return res;
}


matrix plus_matrix(const matrix& matrix1, const matrix& matrix2) {
    int n = matrix1.size();
    int m = matrix1[0].size();

    matrix res(n, vector<double>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            res[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return res;
}

matrix minus_matrix(const matrix& matrix1, const matrix& matrix2) {
    int n = matrix1.size();
    int m = matrix1[0].size();

    matrix res(n, vector<double>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            res[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    return res;
}


void preprocessing(matrix& coefficients, matrix& results) {
    double n = coefficients.size(), m = coefficients[0].size();
    for(int i=0; i < n; i++) {
        double a = coefficients[i][i];
        if (a == 0)
            continue;
        results[i][0] /= a;
        for (int j=0; j < m; j++)
            if (i == j)
                coefficients[i][j] = 0;
            else
                coefficients[i][j] /= -a;
    }
}


double get_current_eps(const matrix& vect1, const matrix& vect2) {
    double eps = 0;
    for(int i=0; i<vect1.size(); i++)
        eps += pow(vect1[i][0] - vect2[i][0], 2);
    return sqrt(eps);
}


void print_matrix(const matrix& matrix1) {
    for(const auto& vect: matrix1) {
        cout << '\t';
        for (auto x: vect)
            cout << x << " ";
        cout << endl;
    }
}


pair<int, matrix> simple_iterations(matrix& a, matrix& b, double EPS0) {
    matrix x_previous = b;
    matrix x_current;
    int k = 0;
    while (get_current_eps(x_current = plus_matrix(b, multiply_matrix(a, x_previous)), x_previous) > EPS0) {
        k += 1;
        x_previous = x_current;
    }
    return make_pair(k, x_previous);
}


pair<int, matrix> seidel_method(const matrix& a, const matrix& b, double EPS0) {
    matrix x_previous = b;
    matrix x_current = x_previous;
    bool flag = true;
    int k = 0;
    double eps = 0;
    while (flag or eps > EPS0) {
        k += 1;
        flag = false;
        for(int i = 0; i < b.size(); i++) {
            x_current[i][0] = 0;
            for(int j=0; j < b.size(); j++) {
                if (i == j)
                    x_current[i][0] += b[i][0];
                else if (i < j)
                    x_current[i][0] += a[i][j]*x_previous[j][0];
                else
                    x_current[i][0] += a[i][j]*x_current[j][0];
            }
        }
        eps = get_current_eps(x_current, x_previous);
        x_previous = x_current;
    }
    return make_pair(k, x_previous);
}


int main() {
    matrix coefficient_matrix = {
        {12, -3, -1, 3},
        {5, 20, 9, 1},
        {6, -3, -21, -7},
        {8, -7, 3, -27}
    };

    matrix equation_roots = {
        {-31},
        {90},
        {119},
        {71}
    };

    preprocessing(coefficient_matrix, equation_roots);

    int last_iteration;
    matrix res_matrix;

    tie(last_iteration, res_matrix) = simple_iterations(coefficient_matrix, equation_roots, 0.001);
    cout << "Simple iterations method:" << endl;
    cout << "\tNumber of iterations = " << last_iteration << endl;
    cout << "\tSolution=" << endl;
    print_matrix(res_matrix);
    cout << endl << endl;



    tie(last_iteration, res_matrix) = seidel_method(coefficient_matrix, equation_roots, 0.001);
    cout << "Seidel's method:" << endl;
    cout << "\tNumber of iterations = " << last_iteration << endl;
    cout << "\tSolution = " << endl;
    print_matrix(res_matrix);

    return 0;
}
