#include <iostream>
#include <vector>
#include <ccomplex>
#include <fstream>
#include "matrix.cpp"

using namespace std;


pair<matrix, matrix> lu(matrix u, matrix& roots, bool root_flag)
{
    int n = u.n;
    matrix l(n, n);
    for (int k = 0; k < n; k++)
    {
        matrix prev = u;

        int swap_index = k;
        for (int i = k + 1; i < n; i++)
        {
            if (abs(prev[swap_index][k]) < abs(prev[i][k]))
                swap_index = i;
        }

        swap(u[k], u[swap_index]);
        swap(l[k], l[swap_index]);
        swap(prev[k], prev[swap_index]);
        if (root_flag)
            swap(roots[k], roots[swap_index]);

        for (int i = k + 1; i < n; i++)
        {
            double h = prev[i][k] / prev[k][k];
            l[i][k] = h;
            for (int j = k; j < n; j++)
                u[i][j] = prev[i][j] - h*prev[k][j];
        }
    }
    for (int i = 0; i < n; i++)
        l[i][i] = 1;
    return make_pair(l, u);
}

matrix direct_passage(matrix x_matr, matrix y_vect, bool flag)
{
    int n = x_matr.n;
    matrix res(n, 1);
    int d, first;
    if (flag) 
    {
        first = n-1;
        d = -1;
    } 
    else 
    { 
        first = 0;
        d = 1;
    }
    for (int i = first; i < n && i >= 0; i += d)
    {
        res[i][0] = y_vect[i][0];
        for (int j = 0; j < n; j++)
        {
            if (i != j)
                res[i][0] -= x_matr[i][j] * res[j][0];
        }
        res[i][0] = res[i][0] / x_matr[i][i];
    }
    return res;
}

matrix solve(pair <matrix, matrix> lu, matrix y_vect)
{
    matrix z = direct_passage(lu.first, y_vect, false);
    matrix x = direct_passage(lu.second, z, true);
    return x;
}

matrix inverse(matrix u, matrix roots)
{
    int n = u.n;
    matrix y_vect(n, 1);
    pair <matrix, matrix> lu_pair = lu(u, roots, true);
    matrix res(n, n);
    for (int i = 0; i < n; i++)
    {
        y_vect[max(i - 1, 0)][0] = 0;
        y_vect[i][0] = 1;
        matrix col = solve(lu_pair, y_vect);
        for (int j = 0; j < n; j++)
            res[j][i] = col[j][0];
    }
    return res;
}

double det(matrix u, matrix roots) 
{
    int n = u.n;
    pair <matrix, matrix> lu_pair = lu(u, roots, false);
    double res = 1;
    for (int i = 0; i < n; i++)
        res *= lu_pair.second[i][i];
    return res;
}

matrix solve_tridiagonal(matrix& x_matr, matrix& y_vect)
{
    int n = x_matr.n;
    vector <double> p(n), q(n);
    p[0] = -x_matr[0][1] / x_matr[0][0];
    q[0] = y_vect[0][0] / x_matr[0][0];
    for (int i = 1; i < n; i++)
    {
        if (i != n - 1)
            p[i] = -x_matr[i][i + 1] / (x_matr[i][i] + x_matr[i][i - 1] * p[i - 1]);
        else
            p[i] = 0;
        q[i] = (y_vect[i][0] - x_matr[i][i - 1] * q[i - 1]) / (x_matr[i][i] + x_matr[i][i - 1] * p[i - 1]);
    }
    matrix res(n, 1);
    res[n - 1][0] = q[n - 1];
    for (int i = n - 2; i >= 0; i--)
        res[i][0] = p[i] * res[i + 1][0] + q[i];
    return res;
}

double max_el(matrix x_matr)
{
    double m = 0;
    for (int i = 0; i < x_matr.n; i++)
    {
        double s = 0;
        for (int j = 0; j < x_matr.m; j++)
            s += abs(x_matr[i][j]);
        if (s > m)
            m = s;
    }
    return m;
}

matrix iterations(matrix x_matr, matrix y_vect, double EPS, int& iters_count)
{
    int n = x_matr.n;
    matrix x1(n, n), y1(n, 1);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            x1[i][j] = -x_matr[i][j] / x_matr[i][i];
        x1[i][i] = 0;
    }
    for (int i = 0; i < n; i++)
        y1[i][0] = y_vect[i][0] / x_matr[i][i];
    matrix x = y1;
    double m = max_el(x_matr);
    double cur = m, eps = EPS + 1;
    iters_count = 0;
    while (eps > EPS)
    {
        matrix prev = x;
        x = y1 + x1 * x;
        if (m < 1)
            eps = cur / (1 - m) * max_el(x - prev);
        else
            eps = max_el(x - prev);
        cur = cur * m;
        iters_count++;
    }
    return x;
}

matrix seidel(matrix x_matr, matrix y_vect, double EPS, int& iters_count)
{
    int n = x_matr.n;
    matrix x1(n, n), y1(n, 1);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            x1[i][j] = -x_matr[i][j] / x_matr[i][i];
        x1[i][i] = 0;
    }
    for (int i = 0; i < n; i++)
        y1[i][0] = y_vect[i][0] / x_matr[i][i];
    matrix x = y1;
    double m = abs(x_matr);
    double cur = m, eps = EPS + 1;
    iters_count = 0;
    while (eps > EPS)
    {
        matrix prev = x;
        for (int i = 0; i < n; i++)
        {
            double cur = y1[i][0];
            for (int j = 0; j < n; j++)
                cur += x1[i][j] * x[j][0];
            x[i][0] = cur;
        }
        if (m < 1)
            eps = cur / (1 - m) * abs(x - prev);
        else
            eps = abs(x - prev);
        cur = cur * m;
        iters_count++;
    }
    return x;
}


pair <matrix, matrix> jacobi(matrix a, double EPS)
{
    int n = a.n;
    double eps = EPS + 1;

    matrix eigenvector(n, n);
    for (int i = 0; i < n; i++)
        eigenvector[i][i] = 1;
    
    while (eps > EPS)
    {
        int cur_i = 1, cur_j = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < i; j++)
                if (abs(a[cur_i][cur_j]) < abs(a[i][j]))
                {
                    cur_i = i;
                    cur_j = j;
                }
        }
        matrix u(n, n);
        double phi = 3.14 / 4;
        if (abs(a[cur_i][cur_i] - a[cur_j][cur_j]) > 1e-9)
            phi = 0.5 * atan((2 * a[cur_i][cur_j]) / (a[cur_i][cur_i] - a[cur_j][cur_j]));
        for (int i = 0; i < n; i++)
            u[i][i] = 1;

        u[cur_j][cur_i] = sin(phi);
        u[cur_i][cur_j] = -sin(phi);
        u[cur_i][cur_i] = cos(phi);
        u[cur_j][cur_j] = cos(phi);
        
        eigenvector = eigenvector * u;
        a = transposition(u) * a * u;
        eps = 0;
        
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++)
                eps += a[i][j] * a[i][j];

        eps = sqrt(eps);
    }

    matrix eigenvalue(n, 1);
    for (int i = 0; i < n; i++)
        eigenvalue[i][0] = a[i][i];
    
    return make_pair(eigenvalue, eigenvector);
}

double sign(double x)
{
    return x > 0 ? 1 : -1;
}

pair <matrix, matrix> QR(matrix x_matr)
{
    int n = x_matr.n;
    matrix E(n, n);

    for (int i = 0; i < n; i++)
        E[i][i] = 1;

    matrix q = E;
    for (int i = 0; i < n - 1; i++)
    {
        matrix v(n, 1);
        double s = 0;

        for (int j = i; j < n; j++)
            s += x_matr[j][i] * x_matr[j][i];
        
        v[i][0] = x_matr[i][i] + sign(x_matr[i][i]) * sqrt(s);
        
        for (int j = i + 1; j < n; j++)
            v[j][0] = x_matr[j][i];
        
        matrix h = E - (2.0 / double(transposition(v) * v)) * (v * transposition(v));
        q = q * h;
        x_matr = h * x_matr;
    }
    return make_pair(q, x_matr);
}

vector<complex<double>> get_eigenvalues(matrix x_matr, double eps)
{
    int n = x_matr.n;
    vector<complex<double>> prev(n);
    while (true)
    {
        pair <matrix, matrix> p = QR(x_matr);
        
        x_matr = p.second * p.first;
        
        vector<complex<double>> current;
        for (int i = 0; i < n; i++)
        {
            if (i < n - 1 && abs(x_matr[i + 1][i]) > 1e-9)
            {
                double b = -(x_matr[i][i] + x_matr[i + 1][i + 1]);
                double c = x_matr[i][i] * x_matr[i + 1][i + 1] - x_matr[i][i + 1] * x_matr[i + 1][i];
                double d = b * b - 4 * c;

                complex<double> isMinus;
                if (d > 0)
                    isMinus = complex<double>(1, 0);
                else
                    isMinus = complex<double>(0, 1);
                
                d = sqrt(abs(d));
                current.push_back(0.5 * (-b - isMinus * d));
                current.push_back( 0.5 * (-b + isMinus * d));
                i++;
            }
            else
                current.push_back(x_matr[i][i]);
        }
        bool flag = true;
        for (int i = 0; i < n; i++)
            flag = flag && abs(current[i] - prev[i]) < eps;
        if (flag)
            break;
        prev = current;
    }
    return prev;
}


int main()
{
    cout << "If you want to change input data, open \"res\" directory and and change the data in the file of the corresponding task" << endl;
    cout << "Enter the code of task you want to check:" << endl;
    cout << "1) LU decomposition" << endl;
    cout << "2) Tridiagonal algorithm" << endl;
    cout << "3) Seidel and simple iterations methods" << endl;
    cout << "4) Method Jacobi" << endl;
    cout << "5) QR decomposition" << endl << endl;
    cout << "enter \"exit\" if you want to stop the programm" << endl << endl;

    int command = 8;
    while (command){
        cin >> command;
        if (command == 0)
        {
            cout << "Thank you for using this program. Come back again!";
            return 0;
        } 
        else if (command == 1)
        {
            ifstream file_input("res/input_1.1.txt");
            int n;
            file_input >> n; 
            matrix coeffs(n, n), roots(n, 1);
            file_input >> coeffs >> roots;
            pair <matrix, matrix> lu_pair = lu(coeffs, roots, true);
            cout << "L =\n" << lu_pair.first << "\nU =\n" << lu_pair.second << endl;
            cout << "\nMatrix's determinant: " << det(coeffs, roots);
            cout << "\nEquations solving:\n" << solve(lu_pair, roots);
            cout << "\nInversed matrix:\n" << inverse(coeffs, roots) * coeffs;
            cout << endl << endl;
        }
        else if (command == 2)
        {
            ifstream file_input("res/input_1.2.txt");
            int n;
            file_input >> n; 
            
            matrix coeffs(n, n), roots(n, 1);
            file_input >> coeffs >> roots;

            cout << "Equations solving:\n" << solve_tridiagonal(coeffs, roots);
            cout << endl << endl;
        }  
        else if (command == 3)
        {
            ifstream file_input("res/input_1.3.txt");
            int n;
            file_input >> n;
            matrix coeffs(n, n), roots(n, 1);
            file_input >> coeffs >> roots;
            
            int iters_count = 0;
            cout << "Simple iterations method:\n" << iterations(coeffs, roots, 0.01, iters_count) << "Number of iterations: " << iters_count << endl;
            cout << "\nSeidel's method:\n" << seidel(coeffs, roots, 0.01, iters_count) << "Number of iterations: " << iters_count << endl << endl << endl;
        }
        else if (command == 4)
        {
            ifstream file_input("res/input_1.4.txt");
            int n;
            file_input >> n;
            matrix coeffs(n, n);
            file_input >> coeffs;

            pair <matrix, matrix> p = jacobi(coeffs, 0.01);
            cout << "Eigenvalues:\n" << p.first << "\nEigenvectors:\n" << p.second << endl << endl;
        }
        else if (command == 5)
        {
            ifstream file_input("res/input_1.5.txt");
            int n;
            file_input >> n;
            matrix coeffs(n, n);
            file_input >> coeffs;

            pair<matrix, matrix> p = QR(coeffs);
            cout << "QR decomposition:\nQ:\n" << p.first << "\nR:\n" << p.second;
            
            vector<complex<double>> v = get_eigenvalues(coeffs, 0.01);
            cout << "\nEigenvalues:\n";
            for (int i = 0; i < n; i++)
                cout << v[i] << ' ';
            cout << endl << endl;
        } else
            cout << "Sorry, but this command is invalid" << endl << endl;
    }
}
