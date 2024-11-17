#include <bits/stdc++.h>

using namespace std;


double f1(double x1, double x2) {
    return x1 - sqrt(x2+3) + 1;
}

double f2(double x1, double x2) {
    return 3*x1*x1 - x2 + x2*x2 - 3;
}

double df1x1(double x1, double x2){
    return 1; 
}

double df1x2(double x1, double x2){ 
    return 1/(2*sqrt(x2+3)); 
}

double df2x1(double x1, double x2){
    return 6*x1 + x2*x2; 
}

double df2x2(double x1, double x2){
    return 3*x1*x1 + 2*x2;
}

double fi1(double x1, double x2) {
    return sqrt(x2+3) - 1;
}

double fi2(double x1, double x2) {
    // return 3*x1*x1 + x2*x2 - 3;
    return sqrt(x2-3*x1*x1+3);
}

double eps(const vector<double>& vect1, const vector<double>& vect2, double q) {
    double d = 0.0;
    for(int i = 0; i < vect1.size(); i++)
        d = max(d, abs(vect1[i] - vect2[i]));
    if (q == -1)
        return d;
    return q*d/(1-q);
}

double determinant(double x1, double x2, vector<vector<function<double(double, double)>>>& matrix) {
    return matrix[0][0](x1, x2) * matrix[1][1](x1, x2) - matrix[0][1](x1, x2) * matrix[1][0](x1, x2);
}


tuple<double, double, int> newton(double start_value_1, double start_value_2, double EPS) {   
    vector<vector<function<double(double, double)>>> J = {{df1x1, df1x2}, {df2x1, df2x2}};
    vector<vector<function<double(double, double)>>> A_1 = {{f1, df1x2}, {f2, df2x2}};
    vector<vector<function<double(double, double)>>> A_2 = {{df1x1, f1}, {df2x1, f2}};

    int counter = 0;
    double x_next_1, x_next_2, x_curr_1 = start_value_1, x_curr_2 = start_value_2;
    x_next_1 = start_value_1 - determinant(start_value_1, start_value_2, A_1)/determinant(start_value_1, start_value_2, J);
    x_next_2 = start_value_2 - determinant(start_value_1, start_value_2, A_2)/determinant(start_value_1, start_value_2, J);

    while (eps({x_curr_1, x_curr_2}, {x_next_1, x_next_2}, -1) >= EPS){
        counter += 1;
        
        x_curr_1 = x_next_1;
        x_curr_2 = x_next_2;

        x_next_1 = x_next_1 - determinant(x_next_1, x_next_2, A_1)/determinant(x_next_1, x_next_2, J);
        x_next_2 = x_next_2 - determinant(x_next_1, x_next_2, A_2)/determinant(x_next_1, x_next_2, J);
    }

    return {x_next_1, x_next_2, counter};
}


tuple<double, double, int> simple_iter(double start_value_1, double start_value_2, double q, double EPS) {
    int counter = 0; 
    double x_next_1 = start_value_1, x_next_2 = start_value_2, x_curr_1 = start_value_1*5, x_curr_2 = start_value_2*5;

    while (eps({x_curr_1, x_curr_2}, {x_next_1, x_next_2}, q) >= EPS and counter < 300){
        counter += 1;

        x_curr_1 = x_next_1;
        x_curr_2 = x_next_2;

        x_next_1 = fi1(x_next_1, x_next_2);
        x_next_2 = fi2(x_next_1, x_next_2);
    }

    return {x_next_1, x_next_2, counter};
}


int main(){
    double epsillon = 0.00001, res_1, res_2;
    int counter;

    tie(res_1, res_2, counter) = newton(-2.0, 0.0, epsillon);
    cout << endl << "Newton method" << endl << "x1 = " << res_1 << endl << "x2 = " << res_2 << endl << "Iteration count = " << counter << endl << endl;

    tie(res_1, res_2, counter) = simple_iter(-2.0, 0.0, 0.7, epsillon);
    cout << "Simple iterations method" << endl << "x1 = " << res_1 << endl << "x2 = " << res_2 << endl << "Iteration count = " << counter << endl << endl;

    return 1;
}
