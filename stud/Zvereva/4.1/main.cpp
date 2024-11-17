#include <iostream>
#include <cmath>
#include <vector>

double dfunc(double x, double y, double dy) {
    return -dy * tan(x) - y * cos(x) * cos(x);
}

double realFunc(double x) {
    return cos(sin(x)) + sin(cos(x));
}

double K(double x, double y, double dy, double h, int i);

double L(double x, double y, double dy, double h, int i){
    if (i == 0){
        return h * dy;
    }
    else if (i == 3){
        return h * (dy + K(x, y, dy, h, i - 1));
    }
    else{
        return h * (dy + K(x, y, dy, h, i - 1) / 2);
    }
}

double K(double x, double y, double dy, double h, int i){
    if (i == 0){
        return h * dfunc(x, y, dy);
    }
    else if (i == 3){
        return h * dfunc(x + h, y + L(x, y, dy, h, i - 1), dy + K(x, y, dy, h, i-1));
    }
    else{
        return h * dfunc(x + h / 2, y + L(x, y, dy, h, i - 1) / 2, dy + K(x, y, dy, h, i-1) / 2);
    }
}

double deltaDY(double x, double y, double dy, double h){
    double d = 0;
    for (int i = 0; i < 4; ++i){
        if (i == 0 || i == 3){
            d += K(x, y, dy, h, i);
        }
    else{
            d += 2 * K(x, y, dy, h, i);
        }
    }
    return d / 6;
}

double deltaY(double x, double y, double dy, double h){
    double d = 0;
    for (int i = 0; i < 4; ++i){
        if (i == 0 || i == 3){
            d += L(x, y, dy, h, i);
        }
    else{
            d += 2 *L(x, y, dy, h, i);
        }
    }
    return d / 6;
}

std::vector<double> Eiler(double l, double r, double h, double y0, double dy0) {
    double x = l;
    int n = static_cast<int>((r - l) / h);
    double dy = dy0;
    double y = y0;
    std::vector<double> res(n);
    res[0] = y0;
    for(int i = 0; i < n-1; ++i) {
        double dy1 = dy + h * dfunc(x, y, dy);
        y = y + h * dy;
        res[i+1] = y;
        dy = dy1;
        x += h;
    }
    return res;
}

std::vector<std::vector<double>> Runge(double l, double r, double h, double y0, double dy0) {
    double x = l;
    int n = static_cast<int>((r - l) / h);
    double dy = dy0;
    double y = y0;
    std::vector<double> res(n);
    std::vector<double> resDY(n);
    res[0] = y0;
    resDY[0] = dy0;
    for(int i = 0; i < n-1; ++i) {
        double dy1 = dy + deltaDY(x, y, dy, h);
        y = y +  deltaY(x, y,dy, h);
        res[i+1] = y;
        resDY[i+1] = dy1;
        dy = dy1;
        x += h;
    }
    return {res, resDY};
}

std::vector<double> Adams(double l, double r, double h, double y0, double dy0) {
    int n = static_cast<int>((r - l) / h);
    std::vector<double> res(n);
    std::vector<double> resDY(n);
    std::vector<double> x(n);
    auto runge = Runge(l, r, h, y0, dy0);
    for(int i = 0; i < 4; ++i) {
        x[i] = l + h * i;
        res[i] = runge[0][i];
        resDY[i] = runge[1][i];
    }
    for (int i = 4; i < n; ++i){
        res[i] = res[i - 1] + h / 24 * (55 * resDY[i - 1] - 59 * resDY[i - 2] + 37 * resDY[i - 3] - 9 * resDY[i - 4]);
        resDY[i] = resDY[i - 1] + h / 24 * (55 * dfunc(x[i - 1], res[i - 1], resDY[i-1]) - 59 * dfunc(x[i - 2], res[i - 2], resDY[i-2]) + 37 * dfunc(x[i - 3], res[i - 3], resDY[i-3]) - 9 * dfunc(x[i - 4], res[i - 4], resDY[i-4]));
        x[i] = x[i - 1] + h;
    }
    return res;
}

std::vector<double> RungeRomberg(std::vector<double> Y1, std::vector<double> Y2, int n, int p){
    std::vector<double> R(n);
    for (int i = 0; i < n; ++i){
        R[i] = (Y1[i * 2] - Y2[i]) / (pow(2, p) - 1);
    }
    return R;
}

std::vector<double> Loss(std::vector<double> Yt, std::vector<double> Y, int n){
    std::vector<double> eps(n);
    for (int i = 0; i < n; ++i){
        eps[i] = abs(Yt[i] - Y[i]);
    }
    return eps;
}



int main() {
    double h = 0.1;
    double l = 0;
    double r = 1;
    double y0 = 0;
    double dy0 = 1;
    int n = static_cast<int>((r - l) / h);

    std::vector<double> resEiler = Eiler(l, r, h, y0, dy0);
    std::vector<double> resEiler2 = Eiler(l, r, 2*h, y0, dy0);
    std::vector<double> resRunge = Runge(l, r, h, y0, dy0)[0];
    std::vector<double> resRunge2 = Runge(l, r, 2*h, y0, dy0)[0];
    std::vector<double> resAdams = Adams(l, r, h, y0, dy0);
    std::vector<double> resAdams2 = Adams(l, r, 2*h, y0, dy0);

    double num = l;
    int iter = 0;
    std::vector<double> realY(n);
    while (num < r-h) {
        realY[iter] = realFunc(num);
        std::cout << "Eiler: " << resEiler[iter] << " Real: " << realFunc(num) << " Runge: "<<resRunge[iter] << " Adams: "<<resAdams[iter] << std::endl;
        num+=h;
        iter+=1;
    }
    std::cout << "================================================" << std::endl;
    std::vector<double> rEiler = RungeRomberg(resEiler, resEiler2, n / 2, 2);
    std::vector<double> rRunge = RungeRomberg(resRunge, resRunge2, n / 2, 5);
    std::vector<double> rAdams = RungeRomberg(resAdams, resAdams2, n / 2, 5);
    std::vector<double> lEiler = Loss(realY, resEiler, n);
    std::vector<double> lRunge = Loss(realY, resRunge, n);
    std::vector<double> lAdams = Loss(realY, resAdams, n);
    for(int i = 0; i < rEiler.size(); ++i) {
        std::cout << "Accuracy by Runge-Romberg. Eiler: " << rEiler[i] << " Runge: "<<rRunge[i] << " Adams: "<<rAdams[i] << std::endl;
    }
    std::cout << "================================================" << std::endl;

    for(int i = 0; i < lEiler.size(); ++i) {
        std::cout << "Error magnitude with the real function. Eiler: " << lEiler[i] << " Runge: "<<lRunge[i] << " Adams: "<<lAdams[i] << std::endl;
    }

    return 0;
}
