#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

double Diff1(vector<double>& x, vector<double>& y,double x0, int i){
    double res;
    res = (y[i+1] - y[i])/(x[i+1] - x[i]) + (((y[i+2] - y[i+1])/(x[i+2] - x[i+1]) - (y[i+1] - y[i])/(x[i+1] - x[i]))/(x[i+2] - x[i])) * (2*x0 - x[i] - x[i+1]);
    return res;
}

double Diff2(vector<double>& x, vector<double>& y,double x0, int i){
    double res;
    res = 2 * (((y[i+2] - y[i+1])/(x[i+2] - x[i+1]) - (y[i+1] - y[i])/(x[i+1] - x[i]))/(x[i+2] - x[i]));
    return res;
}

int main() {
    ofstream fout("output.txt");
    vector<double> x = {-0.2, 0, 0.2, 0.4, 0.6};
    vector<double> y = {-0.40136, 0.0, 0.40136, 0.81152, 1.2435};
    double x0 = 0.2;
    int p = 0;
    for (int i = 0; i < 5; i++){
        if (x0 >= x[i] && x0 <= x[i+1]){
            p = i;
            break;
        }
    }
    double res1 = Diff1(x, y, x0, p);
    double res2 = Diff2(x, y, x0, p);
    fout << "Первая производная в точке Х: " << res1 << endl;
    fout << "Вторая производная в точке Х: " << res2 << endl;
    return 0;
}
