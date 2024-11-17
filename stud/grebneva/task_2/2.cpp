#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

int main() {
    vector<vector<double>> matrix(5, vector <double>(3, 0));
    vector<double> vecd(5, 0);

    ifstream fina("matrix.txt"), finb("col.txt");
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if ((i == 0 && j == 0) || (i == 4 && j == 2)) 
                matrix[i][j] = 0;            
            else
                fina >> matrix[i][j];
        }
    }
    for (int i = 0; i < vecd.size(); i++)
        finb >> vecd[i];

    vector<double> p(5, 0), q(5, 0), x(5, 0);
    p[0] = -matrix[0][2]/matrix[0][1];
    q[0] = vecd[0]/matrix[0][1];

    // Прямой ход
    for (int i = 1; i < matrix.size(); i++)
    {
        p[i] = -matrix[i][2]/(matrix[i][1] + matrix[i][0]*p[i-1]);
        q[i] = (vecd[i] - matrix[i][0]*q[i-1])/(matrix[i][1] + matrix[i][0]*p[i-1]);
    }
    x[4] = q[4];

    // Обратный ход
    for (int i = matrix.size() - 2; i > -1; i--)
    {
        x[i] = p[i]*x[i+1] + q[i];
    }
    
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
   
    fout << "Решение системы:" << endl;
    for (size_t i = 0; i < x.size(); ++i)
        fout << "x[" << i << "] = " << x[i] << endl;

    return 0;
}
