#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

using namespace std;

vector <double> first_diff(vector<double> x, vector <double> y, double root_index){
    int n = x.size();
    double left_diff, right_diff, diff;
    if (root_index != 1 && root_index != (n - 2)){
        left_diff = (y[root_index] - y[root_index - 1]) / (x[root_index] - x[root_index - 1]);
        right_diff = (y[root_index + 1] - y[root_index]) / (x[root_index + 1] - x[root_index]);
    
        diff = (y[root_index] - y[root_index - 1]) / (x[root_index] - x[root_index - 1]) \
        + ((y[root_index + 1] - y[root_index])/(x[root_index+1] - x[root_index]) - (y[root_index] - y[root_index - 1])/(x[root_index] - x[root_index-1]) ) \
        / (x[root_index + 1] - x[root_index - 1]) \
        * (2 * x[root_index] - x[root_index] - x[root_index - 1]);
        return vector <double> {left_diff, right_diff, diff};
    }
    cout << "Error!" << endl;
    return vector <double> (0,0);
}

double second_diff(vector <double> x, vector <double> y, double root_index){
    int n = x.size();
    if (root_index != 1 && root_index != (n - 2)){
        double dx2 = 2 * ((y[root_index + 1] - y[root_index])/(x[root_index+1] - x[root_index]) - (y[root_index] - y[root_index - 1])/(x[root_index] - x[root_index-1]) ) \
            / (x[root_index + 1] - x[root_index - 1]) ;
        return dx2;
    }
    cout << "Error!" << endl;
    return 0.0;

}


int main(){
    ofstream fout("answer4.txt");
    // fout.precision(2);
    // fout << fixed;
    int n = 5;
    vector <double> x = {-1, 0, 1, 2, 3};
    vector <double> y = {-1.7854, 0.0, 1.7854, 3.1071, 4.249};
    double X = 2; // индекс 1-цы в векторе x

    vector <double> first_diffs = first_diff(x,y,X);
    double dx2 = second_diff(x,y,X);
    fout << "First diff" << endl;
    fout << "left diff:\t" << first_diffs[0] << endl;
    fout << "right diff:\t" << first_diffs[1] << endl;
    fout << "diff with second order precision:\t" << first_diffs[2] << endl << endl;
    fout << "Second diff" << endl;
    fout << dx2 << endl;
}