#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

pair<double, double> derivative(vector<double>& x, vector<double>& y, double X, int n) {
    int index = -1;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= X)
            index = i;
        else
            break;
    }
    double h = x[1] - x[0];
    double derivative_1 = (y[index + 1] - y[index - 1]) / (2 * h);
    double derivative_2 = (y[index + 1] - 2 * y[index] + y[index - 1]) / (h * h);

    return make_pair(derivative_1, derivative_2);
}

int main(){
    int n = 5;
    vector<double> x = {-1.0, -0.4, 0.2, 0.6, 1.0};
    vector<double> y = {-1.4142, -0.55838, 0.27870, 0.84008, 1.4142};
    double X = 0.2;
    ofstream fout("output.txt");
    fout << "Первая производная:" << derivative(x, y, X, n).first << endl;
    fout << "Вторая производная:" << derivative(x, y, X, n).second << endl;
    return 0;
}