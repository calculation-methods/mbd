#pragma once
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>


using namespace std;

class Matrix {
public:
    Matrix();
    Matrix(int n, int m);

    void make_ones();
    void transpose();

    vector<double>& operator[](const int index);
    const vector<double>& operator[](const int index) const;

    friend const Matrix operator+(const Matrix& left, const Matrix& right);
    friend const Matrix operator-(const Matrix& left, const Matrix& right);

    friend const Matrix operator*(const Matrix& left, const Matrix& right);
    friend const vector<double> operator*(const Matrix& left, const vector<double>& right);
    friend const Matrix operator*(const Matrix& left, double right);
    friend const Matrix operator*(double left, const Matrix& right);

    const Matrix& operator=(const Matrix& right);

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

    double get_norm() const;
    double get_upper_norm() const;

    int get_n() const;
    int get_m() const;

    bool is_three_diagonal() const;
    bool is_simmetric() const;

    bool is_quadratic() const;

private:
    vector<vector<double>> _matrix;
    int n_size;
    int m_size;
};

#endif