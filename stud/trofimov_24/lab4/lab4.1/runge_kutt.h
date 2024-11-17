//
// Created by bigboy on 25.05.2024.
//

#ifndef LAB4_1_RUNGE_KUTT_H
#define LAB4_1_RUNGE_KUTT_H
#include <vector>


using namespace std;


class RungeKutt {
public:
    vector<double> result(double h, double x0, double y10, double y20, int steps);
};


#endif //LAB4_1_RUNGE_KUTT_H
