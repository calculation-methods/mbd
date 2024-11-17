#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

class Matrix {
private:
    int rows_, cols_;
    vector<vector<double>> matrix_;
    vector<int> swp_;

    void SwapMatrix(Matrix &other) {
        swap(rows_, other.rows_);
        swap(cols_, other.cols_);
        swap(matrix_, other.matrix_);
    }

public:
    Matrix(int rows, int cols) {
        rows_ = rows;
        cols_ = cols;
        matrix_.resize(rows_);
        for (int i = 0; i < rows_; ++i) {
            matrix_[i].resize(cols_);
        }
    }
    Matrix(const Matrix &other) : Matrix(other.rows_, other.cols_) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                matrix_[i][j] = other.matrix_[i][j];
            }
        }
    }
    Matrix() : Matrix(1, 1) {}
    Matrix(Matrix &&other) {
        this->SwapMatrix(other);
        other.rows_ = 0;
        other.cols_ = 0;
    }

    int GetRows() const { return rows_; }
    int GetCols() const { return cols_; }
    const vector<int> &GetSwp() const { return swp_; }
    
    void MulNumber(const double num) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                matrix_[i][j] *= num;
            }
        }
    }

    Matrix MulMatrixReturn(const double num) {
        Matrix res = *this;
        res.MulNumber(num);
        return res;
    }

    void MulMatrix(const Matrix &other) {
        Matrix tmp(rows_, other.cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < other.cols_; ++j) {
                for (int k = 0; k < cols_; ++k)
                    tmp.matrix_[i][j] += matrix_[i][k] * other.matrix_[k][j];
            }
        }
        this->SwapMatrix(tmp);
    }

    Matrix MulMatrixReturn(const Matrix &other) {
        Matrix res = *this;
        res.MulMatrix(other);
        return res;
    }

    Matrix Transpose() const {
        Matrix result(cols_, rows_);
        for (int i = 0; i < result.rows_; ++i) {
            for (int j = 0; j < result.cols_; ++j) {
                result.matrix_[i][j] = matrix_[j][i];
            }
        }
        return result;
    }
    pair<Matrix, Matrix> LU(Matrix & L, Matrix & U) {
        swp_.clear();
        int n = this->GetRows();
        for (int k = 0; k < n; ++k) { 
            int index = k;           
            for (int i = k + 1; i < n; ++i) {
                if (abs(U(i, k)) > abs(U(index, k))) {
                    index = i;
                }
            }
            swap(U(k), U(index));
            swap(L(k), L(index));
            swp_.push_back(index);
            for (int i = k + 1; i < n; ++i) {
                double m = U(i, k) / U(k, k);
                L(i, k) = m;
                for (int j = k; j < n; ++j) {
                    U(i, j) -= m * U(k, j);
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            L(i, i) = 1;
        }
        return {L, U};
    }
    Matrix Solve(Matrix &C) {
        Matrix U(*this);
        Matrix L(this->GetCols(), this->GetCols());
        LU(L,U);
        Matrix B(C);
        vector<int> swp = this->GetSwp();
        for (int i = 0; i < swp.size(); ++i) {
            swap(B(i), B(swp[i]));
        }
        int n = this->GetRows();
        
        Matrix Z(n, 1);
        for (int i = 0; i < n; ++i) {
            Z(i, 0) = B(i, 0);
            for (int j = 0; j < i; ++j) {
                Z(i, 0) -= L(i, j) * Z(j, 0);
            }
        }
        
        Matrix X(n, 1);
        for (int i = n - 1; i >= 0; --i) {
            X(i, 0) = Z(i, 0);
            for (int j = i + 1; j < n; ++j) {
                X(i, 0) -= U(i, j) * X(j, 0);
            }
            X(i, 0) = X(i, 0) / U(i, i);
        }
        return X;
    }
    double &operator()(int i, int j) {
        return matrix_[i][j];
    }
    vector<double> &operator()(int row) { return matrix_[row]; }
};

istream& operator>>(istream& stream, Matrix& m)
{
    for (int i = 0; i < m.GetRows(); i++)
    {
        for (int j = 0; j < m.GetCols(); j++)
            stream >> m(i, j);
    }
    return stream;
}

ostream& operator<<(ostream& stream, Matrix m)
{
    for (int i = 0; i < m.GetRows(); i++)
    {
        for (int j = 0; j < m.GetCols(); j++)
            stream << m(i, j) << ' ';
        stream << '\n';
    }
    return stream;
}

ostream& operator<<(ostream& stream, vector<double> v)
{
    for (int i = 0; i < v.size(); ++i) {
        stream << v[i] << " ";
    }
    return stream;
}

vector<double> least_squares_method(vector<double> &x, vector<double> &y, int m) {
    int n = x.size();
    m = m+1; 
    Matrix Y(n, 1), Phi(n, m);
    for (int i = 0; i < n; ++i) {
        Y(i, 0) = y[i];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            Phi(i, j) = pow(x[i], j);
        }
    }
    Matrix Phi_T = Phi.Transpose();
    Matrix B = Phi_T.MulMatrixReturn(Y);
    Matrix res = (Phi_T.MulMatrixReturn(Phi)).Solve(B);
    vector<double> polynom;
    for (int i = 0; i < res.GetRows(); ++i) {
        for (int j = 0; j < res.GetCols(); ++j) {
            polynom.push_back(res(i, j));
        }
    }
    return polynom;
}

double get_error(vector<double> &x, vector<double> &y, vector<double> &p) {
    double f_x = 0, error = 0;
    for (int i = 0; i < x.size(); ++i) {
        f_x = 0;
        for (int j = 0; j < p.size(); ++j) {
            f_x += p[j] * pow(x[i], j);
        }
        error += (f_x - y[i]) * (f_x - y[i]);
    }
    return error;
}


int main() {
    vector<double> x = {1.0, 1.9, 2.8, 3.7, 3.6, 5.5};
    vector<double> y = {3.4142, 2.9818, 3.3095, 3.8184, 4.3599, 4.8318};
    ofstream fout("answer.txt");
    ofstream fout_py_args("py_args.txt");
    fout_py_args << x << "\n";
    fout_py_args << y << "\n";

    vector<double> polynom = least_squares_method(x, y, 1);
    fout << "Least squares polynom [1st power]:\n";
    for (int i = 0; i < polynom.size(); ++i) {
        if (i != polynom.size() - 1)
            fout << polynom[i] << "x^" << i << " + ";
        else
            fout << polynom[i] << "x^" << i << "\n";
    }
    double error = get_error(x, y, polynom);
    fout << "error = " << error << "\n\n";
    fout_py_args << polynom << "\n";


    polynom.clear();
    polynom = least_squares_method(x, y, 2);
    fout << "Least squares polynom [2nd power]:\n";
    for (int i = 0; i < polynom.size(); ++i) {
        if (i != polynom.size() - 1)
            fout << polynom[i] << "x^" << i << " + ";
        else
            fout << polynom[i] << "x^" << i << "\n";
    } 
    error = get_error(x, y, polynom);
    fout << "error = " << error << "\n\n";
    fout_py_args << polynom << "\n";
    fout_py_args.close();
    system("python show_plot.py");
    return 0;
}