#include <bits/stdc++.h>

using namespace std;


auto f1 = [](vector<double> x) {
    return pow(x[0], 2) + pow(x[1], 2) - 16;
};

auto f2 = [](vector<double> x) {
    return x[0] - exp(x[1]) + 4;
};

pair<vector<double>, int> newton_method(vector<double> x_0, double EPS) {
    /*
        Функция f1(x1, x2): x1^2 + x2^2 - 16
        Функция f2(x1, x2): x1 - e^(x2) + 4

        Первая производная f1 по x1: 2*x1
        Первая производная f1 по x2: 2*x2

        Первая производная f2 по x1: 1
        Первая производная f2 по x2: -e^(x2)

        Начальные значения (подбираются по графикам):
        x1 = -3.5
        x2 = -2
    */
    auto eps = [](const vector<double>& vect1, const vector<double>& vect2) {
        double max_diff = 0.0;
        for (size_t i = 0; i < vect1.size(); ++i) {
            max_diff = max(max_diff, abs(vect1[i] - vect2[i]));
        }
        return max_diff;
    };

    auto det = [](const vector<double>& x, const vector<vector<function<double(vector<double>)>>>& matrix) {
        return matrix[0][0](x) * matrix[1][1](x) - matrix[0][1](x) * matrix[1][0](x);
    };

    auto df1_x1 = [](const vector<double>& x) { return 2 * x[0]; };
    auto df1_x2 = [](const vector<double>& x) { return 2 * x[1]; };
    auto df2_x1 = [](const vector<double>& x) { return 1; };
    auto df2_x2 = [](const vector<double>& x) { return -exp(x[1]); };

    vector<vector<function<double(vector<double>)>>> A1 = {
        {f1, df1_x2},
        {f2, df2_x2}
    };

    vector<vector<function<double(vector<double>)>>> A2 = {
        {df1_x1, f1},
        {df2_x1, f2}
    };

    vector<vector<function<double(vector<double>)>>> J = {
        {df1_x1, df1_x2},
        {df2_x1, df2_x2}
    };

    vector<vector<vector<function<double(vector<double>)>>>> A = {A1, A2};

    vector<double> x_next, x_curr = x_0;
    x_next = {
        x_0[0] - det(x_0, A[0])/det(x_0, J),
        x_0[1] - det(x_0, A[1])/det(x_0, J)
    };
    int k = 1;

    while (eps(x_curr, x_next) >= EPS){
        k += 1;
        x_curr = x_next;
        x_next = {
            x_next[0] - det(x_next, A[0])/det(x_next, J),
            x_next[1] - det(x_next, A[1])/det(x_next, J)
        };
    }
    return make_pair(x_next, k);
}





pair<vector<double>, int> simple_iterations_method(vector<double> x_0, double q, double EPS) {
    /*
        Функция f1(x1, x2): x1^2 + x2^2 - 16
        Функция f2(x1, x2): x1 - e^(x2) + 4

        Уравнение с выделенным членом (x1 = phi1(x1, x2)): x1 = e^(x2) - 4
        Уравнение с выделенным членом (x2 = phi2(x1, x2)): x2 = sqrt(16 - x1^2)

        phi1(x1, x2) = e^(x2) - 4
        phi1_dx1(x1, x2) = 0
        phi1_dx2(x1, x2) = e^(x2)

        phi2(x1, x2) = sqrt(16 - x1^2)
        phi1_dx1(x1, x2) = -x1/sqrt(16 - x1^2)
        phi1_dx2(x1, x2) = 0
    */
   
    auto eps = [](const vector<double>& vect1, const vector<double>& vect2, double q) {
        double max_diff = 0.0;
        for (size_t i = 0; i < vect1.size(); ++i) {
            max_diff = max(max_diff, abs(vect1[i] - vect2[i]));
        }
        return q*max_diff/(1-q);
    };

    auto phi1 = [](const vector<double>& x) {
        return exp(x[1]) - 4;
    };

    auto phi2 = [](const vector<double>& x) {
        return -sqrt(16 - pow(x[0], 2));
    };

    vector<double> x_next = x_0, x_curr = {x_0[0]*3, x_0[1]*3};
    int k = 1;

    while (eps(x_curr, x_next, q) >= EPS){
        k += 1;
        x_curr = x_next;
        x_next = {
            phi1(x_next),
            phi2(x_next)
        };
    }
    return make_pair(x_next, k);
}


int main(){
    double deviation = 1e-9;
    vector<double> result, x_0 = {-3.5, -2};
    int number_of_iterations;

    tie(result, number_of_iterations) = newton_method(x_0, deviation);
    cout << "\nNewton's method\n";
    cout << "Result (x_k): ["  << result[0] << ", " << result[1] << "]" << endl << "Number of iterations (k): " << number_of_iterations << endl;
    cout << "The value of the function at the given root x_k: " << f1(result) << " " << f2(result) << endl << endl;

    tie(result, number_of_iterations) = simple_iterations_method(x_0, 0.9, deviation);
    cout << "\nhe method of simple iterations\n";
    cout << "Result (x_k): ["  << result[0] << ", " << result[1] << "]" << endl << "Number of iterations (k): " << number_of_iterations << endl;
    cout << "The value of the function at the given root x_k: " << f1(result) << " " << f2(result) << endl << endl;

    return 1;
}
