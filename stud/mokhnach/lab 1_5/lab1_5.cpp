#include <ccomplex>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define EPS 1e-5

int const n = 3;

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
    Matrix() : Matrix(1, 1) {}
    Matrix(const Matrix &other) : Matrix(other.rows_, other.cols_) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                matrix_[i][j] = other.matrix_[i][j];
            }
        }
    }
    Matrix(Matrix &&other) {
        this->SwapMatrix(other);
        other.rows_ = 0;
        other.cols_ = 0;
    }

    int GetRows() const { return rows_; }
    int GetCols() const { return cols_; }
    const vector<int> &GetSwp() const { return swp_; }
    void SumMatrix(const Matrix &other) {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw runtime_error("Нельзя сложить матрицы разной размерности\n");
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                matrix_[i][j] += other.matrix_[i][j];
            }
        }
    }
    void SubMatrix(const Matrix &other) {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw runtime_error("Нельзя вычитать матрицы разной размерности\n");
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                matrix_[i][j] -= other.matrix_[i][j];
            }
        }
    }
    void MulNumber(const double num) {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                matrix_[i][j] *= num;
            }
        }
    }

    Matrix MulMatrixReturn(const double num) {
        Matrix tmp = *this;
        tmp.MulNumber(num);
        return tmp;
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
        Matrix tmp = *this;
        tmp.MulMatrix(other);
        return tmp;
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
    int sign(double x) {
        if (x > 0)
            return 1;
        if (x < 0)
            return -1;
        return 0;
    }

    pair<Matrix, Matrix> qr_decomposition() {
        int n = this->rows_;
        Matrix E(n, n);
        for (int i = 0; i < n; ++i) {
            E(i, i) = 1;
        }
        Matrix Q = E;
        Matrix A = *this;
        for (int i = 0; i < n - 1; ++i) {
            Matrix H(n, n);
            Matrix V(n, 1);
            double norm = 0;
            for (int j = i; j < n; ++j) {
                norm += A(j, i) * A(j, i);
            }
            norm = sqrt(norm);
            for (int j = i; j < V.GetRows(); ++j) {
                if (j == i) {
                    V(j, 0) = A(i, i) + sign(A(i, i)) * norm;
                } else {
                    V(j, 0) = A(j, i);
                }
            }
            Matrix V_T = V.Transpose();
            H = E - V.MulMatrixReturn(V_T).MulMatrixReturn(2 / (V_T.MulMatrixReturn(V))(0, 0));
            A = H.MulMatrixReturn(A);
            Q = Q.MulMatrixReturn(H);
        }
        return {Q, A};
    }

    vector<complex<double>> qr_method(double eps) {
        int n = this->rows_;
        Matrix A = *this;
        vector<complex<double>> lambda;
        vector<complex<double>> lambda_prev;
        int counter = 0;
        int iter = 50;
        while (true) {
            pair<Matrix, Matrix> QR = A.qr_decomposition();
            Matrix Q = QR.first;
            Matrix R = QR.second;
            A = R.MulMatrixReturn(Q);
            // cout << "A\n";
            // A.ShowMatrix();
            if (counter != iter) {
                ++counter;
                continue;
            }
            for (int i = 0; i < n; i += 1) {
                double sum = 0;
                for (int j = i + 1; j < n; ++j) {
                    sum += abs(A(j, i));
                }
                if (sum < 0.001) {
                    lambda.push_back(A(i, i));
                } else {
                    // (a_jj - Lambda)(a_j+1,j+1 - Lambda) = aj,j+1 * aj+1, j
                    double a = 1;
                    double b = (-1) * (A(i, i) + A(i + 1, i + 1));
                    double c = A(i, i) * A(i + 1, i + 1) - A(i, i + 1) * A(i + 1, i);
                    double d = b * b - 4 * c;
                    complex<double> x1, x2;
                    if (d < 0) {
                        x1 = (-b + sqrt((abs(d))) * complex<double>(0, 1)) / (2 * a);
                        x2 = (-b - sqrt((abs(d))) * complex<double>(0, 1)) / (2 * a);
                    } else {
                        x1 = (-b + sqrt(d)) / (2 * a);
                        x2 = (-b - sqrt(d)) / (2 * a);
                    }
                    lambda.push_back(x1);
                    lambda.push_back(x2);
                    ++i;
                }
            }
            bool exit = true;
            // исключаем первую итерацию
            if (lambda_prev.size() != 0) {
                for (int i = 0; i < lambda.size(); i++) {
                    if (abs(lambda[i] - lambda_prev[i]) > eps) {
                        exit = false;
                        break;
                    }
                }
                if (exit == true)
                    break;
            }
            lambda_prev = lambda;
            lambda.clear();
            counter = 0;
        }
        return lambda;
    }

    Matrix operator+(const Matrix &other) {
        Matrix result(*this);
        result.SumMatrix(other);
        return result;
    }
    Matrix operator-(const Matrix &other) {
        Matrix result(*this);
        result.SubMatrix(other);
        return result;
    }
    Matrix operator=(const Matrix &other) {
        if (this != &other) { // b = b
            Matrix tmp(other);
            this->SwapMatrix(tmp);
        }
        return *this;
    }
    double &operator()(int i, int j) {
        if (i < 0 || i >= rows_ || j < 0 || j >= cols_)
            throw runtime_error("Индекс за пределами матрицы\n");
        return matrix_[i][j];
    }
    vector<double> &operator()(int row) { return matrix_[row]; }
};

ostream &operator<<(ostream &stream, Matrix A) {
    for (int i = 0; i < A.GetRows(); i++) {
        for (int j = 0; j < A.GetCols(); j++)
            stream << A(i, j) << ' ';
        stream << '\n';
    }
    return stream;
}

istream &operator>>(istream &stream, Matrix &A) {
    for (int i = 0; i < A.GetRows(); i++) {
        for (int j = 0; j < A.GetCols(); j++)
            stream >> A(i, j);
    }
    return stream;
}
int main() 
{
    Matrix A(n, n), self_vec(n,n), self_value(n,1);
    double eps;
    ofstream fout("answer.txt");
    fout.precision(2);
    fout << fixed;
    ifstream fin("input.txt");
    fin >> eps;
    fin >> A;
    fout << "Эпсилон = " << to_string(eps) << endl;
    pair<Matrix, Matrix> QR = A.qr_decomposition();
    fout << "Q:\n";
    fout << QR.first;
    fout << "\nR:\n";
    fout << QR.second;
    
    vector<complex<double>> labmda = A.qr_method(eps);
    fout << "\nСобственные значения:\n";
    for (int i = 0; i < labmda.size(); ++i) {
        fout << labmda[i] << " ";
    }
}