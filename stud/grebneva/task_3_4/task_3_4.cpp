#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

pair<double, double> derivatives(const vector<double> x, const vector<double> y, const int n, const double starX){
    vector<vector<double>> d(n, vector<double>(n, 0));
    double f = 0., g = 0.;
    double v = 0., w = 0.;

    for (int i = 0; i < n; ++i) {
        d[i][0] = y[i];
    }
    for (int j = 1; j < n; ++j) {
        for (int i = 0; i < n-j; ++i) {
            d[i][j] = (d[i][j-1] - d[i+1][j-1])/(x[i] - x[i+j]);
        }
    }

    for (int j = 1; j < n; ++j) {
        f = 0.;
        for (int k = 0; k < j; ++k) {
            g = d[0][j];
            for (int i = 0; i < j; ++i) {
                if ( i != k) g = g*(starX - x[i]);
            }
            f = f + g;
        }
        v = v + f;
    }

    for (int j = 3; j <= n; ++j) {
        f = 0.;
        for (int k = 1; k <= (j-1)*(j-2); ++k) {
            g = d[0][j - 1];
            for (int i = 1; i < j; ++i) {
                if (( i != 1 + (k-1)/(j-2)) && (i != 1 + (k + (k-1)/(j-1)) % (j-1))) 
                    g = g * (starX - x[i-1]);
            }
            f = f + g;
        }
        w = w + f;
    }

    return make_pair(v, w);
}

int main() {
    int n = 5;
    vector<double> x(n, 0), y(n, 0); 
    double starX = 0.;

    ifstream fin("input.txt");
    for (int i = 0; i < n; ++i){
        fin >> x[i];     
    }
    for (int i = 0; i < n; ++i){
        fin >> y[i];
    }
    fin >> starX;

    ofstream fout("answer.txt");
    fout.precision(4);
    fout << fixed;

    fout << "Первая производная:" << derivatives(x, y, n, starX).first << endl;
    fout << "Вторая производная:" << derivatives(x, y, n, starX).second << endl;
    
    return 0;
}