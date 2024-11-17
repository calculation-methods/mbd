#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>


int input_coeff(  std::vector<double> &a, 
                  std::vector<double> &b, 
                  std::vector<double> &c,
                  std::vector<double> &d,
                  std::string filename ) {
    std::ifstream idescr(filename);
        int n;
        idescr >> n;
        a.resize(n);
        for(int i = 1; i < n; i++){
            idescr >> a[i];
        }
        b.resize(n);
        for(int i = 0; i < n; i++){
            idescr >> b[i];
        }
        c.resize(n);
        for(int i = 0; i < n - 1; i++){
            idescr >> c[i];
        }
        d.resize(n);
        for(int i = 0; i < n; i++){
            idescr >> d[i];
        }
    idescr.close();
    return n;
}


int main(int argc, char **argv){
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
    std::vector<double> d;
    int n = input_coeff(a, b, c, d, argv[1]);

    // Прямой ход
    std::vector<double> p(n);
    std::vector<double> q(n);
    p[0] = -c[0] / b[0];
    q[0] = d[0] / b[0];
    for(int i = 1; i < n; i++){
        p[i] = -c[i] / (b[i] + a[i] * p[i-1]);
        q[i] = (d[i] - a[i] * q[i-1]) / (b[i] + a[i] * p[i-1]);
    }
    // Обратный ход
    std::vector<double> x(n);
    x[n-1] = q[n-1];
    for(int i = n - 2; i >= 0; i--){
        x[i] = p[i] * x[i+1] + q[i];
    }

    // Вывод решения
    std::cout << "\tРешение системы" << std::endl;
    for(int i = 0; i < n; i++){
            std::cout << std::round(x[i]*100)/100.0  << '\t';
    }
    std::cout << std::endl;
    return 0;
}