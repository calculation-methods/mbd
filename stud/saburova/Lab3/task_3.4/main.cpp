#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<double> x = {-0.2, 0.0, 0.2, 0.4, 0.6}, y = {1.5722, 1.5708, 1.5694, 1.5593, 1.5273}, dy, ddy;
    double star_x = 0.2;
    for (int i = 0; i < 5; i++) 
        dy.push_back((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
    for (int i = 0; i < 4; i++)
        ddy.push_back(2 * ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i])) / (x[i + 2] - x[i]));
    for (int i = 0; i < 5; i++)
        if (x[i] == star_x) {
            cout << "Left derivative = " << dy[i - 1] << "\tRight derivative = " << dy[i];
            break;
        } else if (x[i] < star_x && star_x < x[i + 1]) 
            cout << "First derivative = " << dy[i];
    for (int i = 0; i < 4; i++)
        if (x[i] <= star_x && star_x <= x[i + 1]) {
            cout << "\tSecond derivative = " << ddy[i] << endl;
            break;
        }
    return 0;
}
