//
// Created by владислав трофимов on 11.05.2024.
//

#include "solve.h"
using namespace std;

vector<double> der1(vector<double> x,vector<double> y){
    vector<double> res;
    int n = x.size();
    for (int i = 0; i < n - 1; ++i) {
        res.push_back((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
    }
    return res;
};

vector<double> der2(vector<double> x,vector<double> y){
    vector<double> res;
    int n = x.size();
    vector<double> derivative_second;
    for (int i = 0; i < n - 2; ++i) {
        res.push_back(2 * ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i])) / (x[i + 2] - x[i]));
    }
    return res;
};
