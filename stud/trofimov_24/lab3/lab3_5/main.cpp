#include <functional>
#include <iostream>
#include "solve.h"

using namespace std;



double RRR_estimation(double F_1, double F_2, double step_1, double step_2, double p){
    return F_1 + (F_1 - F_2)/(pow((step_2/step_1), p) - 1);
}


int main() {
    auto y = [](double x) { return 16 - pow(x,2); };
    double x_0 = -2, x_k = 2;
    vector<double> precision = {1, 0.5, 0.000001};




    cout << " rectangular " << endl;
    vector<double> F1;
    for (auto h : precision) {
        F1.push_back(rectangular(y, x_0, x_k, h));
        cout << "\tIntegral = " << F1.back() << "\t\t(Step = " << h << ")" << endl;
    }
    cout << "\tIntegral = " << RRR_estimation(F1[0], F1[1], precision[0], precision[1], 2) << "\t\t(Runge-Romberg-Richardson estimation, step 1 = " << precision[0] << ", step 2 = " << precision[1] << ")" << endl;
    cout << "\n";


    cout << " trapeze " << endl;
    vector<double> F2;
    for (auto h : precision) {
        F2.push_back(trapeze(y, x_0, x_k, h));
        cout << "\tIntegral = " << F2.back() << "\t\t(Step = " << h << ")" << endl;
    }
    cout << "\tIntegral = " << RRR_estimation(F2[0], F2[1], precision[0], precision[1], 2) << "\t\t(Runge-Romberg-Richardson estimation, step 1 = " << precision[0] << ", step 2 = " << precision[1] << ")" << endl;
    cout << "\n";


    cout << " simpson " << endl;
    vector<double> F3;
    for (auto h : precision) {
        F3.push_back(simpson(y, x_0, x_k, h));
        cout << "\tIntegral = " << F3.back() << "\t\t(Step = " << h << ")" << endl;
    }
    cout << "\tIntegral = " << RRR_estimation(F3[0], F3[1], precision[0], precision[1], 2) << "\t\t(Runge-Romberg-Richardson estimation, step 1 = " << precision[0] << ", step 2 = " << precision[1] << ")" << endl;
    cout << "\n";
    return 0;
}