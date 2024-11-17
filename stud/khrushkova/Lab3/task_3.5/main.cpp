#include <bits/stdc++.h>
using namespace std;

double my_function(double x) {
    return sqrt(x*x + 9); 
}

int main() {
    double start_x = 1, end_x = 5, step_1 = 1.0, step_2 = 0.5;
    double F1, F2;

    cout << "Rectangle" << endl;
    double x = start_x, t = 0;
    while (x < end_x){
        t += my_function((2*x + step_1)/2);
        x += step_1;
    }
    F1 = step_1*t;
    x = start_x, t = 0;
    while (x < end_x){
        t += my_function((2*x + step_1)/2);
        x += step_2;
    }
    F2 = step_2*t;
    cout << "1 result = " << F1 << "\t2 result = " << F2 << "\t3 result: " << F1 + (F1 - F2)/(pow((step_2/step_1), 2) - 1) << endl << endl << "Trapeze" << endl;
    
    x = start_x+step_1, t = my_function(start_x)/2 + my_function(end_x)/2;
    while (x < end_x){
        t += my_function(x);
        x += step_1;
    }
    F1 = step_1 * t;
    x = start_x+step_2, t = my_function(start_x)/2 + my_function(end_x)/2;
    while (x < end_x){
        t += my_function(x);
        x += step_2;
    }
    F2 = step_2 * t;
    cout << "1 result = " << F1 << "\t2 result = " << F2 << "\t3 result: " << F1 + (F1 - F2)/(pow((step_2/step_1), 2) - 1) << endl << endl << "Simpson" << endl;    
    
    x = start_x + step_1, t = my_function(start_x) + my_function(end_x);
    bool flag = true;
    while (x < end_x){
        t += my_function(x) * ((flag) ? 4 : 2);
        x += step_1;
        flag = !flag;
    }
    F1 = step_1 * t / 3;
    x = start_x + step_2, t = my_function(start_x) + my_function(end_x);
    flag = true;
    while (x < end_x){
        t += my_function(x) * ((flag) ? 4 : 2);
        x += step_2;
        flag = !flag;
    }
    F2 = step_2 * t / 3;
    cout << "1 result = " << F1 << "\t2 result = " << F2 << "\t3 result: " << F1 + (F1 - F2)/(pow((step_2/step_1), 2) - 1) << endl << endl;
    return 0;
}