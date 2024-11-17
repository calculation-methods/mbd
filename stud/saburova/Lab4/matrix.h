#if !defined(MATRIX)
#define MATRIX

#include <ccomplex>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

#define EPS 1e-5

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
    Matrix Minor(int i, int j) const {
        Matrix result(rows_ - 1, cols_ - 1);
        int ki = 0;
        for (int new_i = 0; new_i < result.rows_; ++new_i) {
            int kj = 0;
            if (new_i == i)
                ki = 1;
            for (int new_j = 0; new_j < result.cols_; ++new_j) {
                if (new_j == j)
                    kj = 1;
                if (new_i + ki < rows_ && new_j + kj < cols_)
                    result.matrix_[new_i][new_j] = matrix_[new_i + ki][new_j + kj];
            }
        }
        return result;
    }

public:
    Matrix(int rows, int cols) {
        if (rows < 1 || cols < 1)
            throw runtime_error(
                "Нельзя создать матрицу с таким количеством столбцов и строчек\n");
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

    void SetRows(int rows) {
        if (rows < 1)
            throw runtime_error("Число строк должно быть натуральным\n");
        Matrix tmp_matrix(rows, cols_);
        for (int i = 0; i < min(tmp_matrix.rows_, rows_); ++i) {
            for (int j = 0; j < cols_; ++j) {
                tmp_matrix.matrix_[i][j] = matrix_[i][j];
            }
        }
        this->SwapMatrix(tmp_matrix);
    }
    void SetCols(int cols) {
        if (cols < 1)
            throw runtime_error("Число столбцов должно быть натуральным\n");
        Matrix tmp_matrix(rows_, cols);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < min(tmp_matrix.cols_, cols_); ++j) {
                tmp_matrix.matrix_[i][j] = matrix_[i][j];
            }
        }
        this->SwapMatrix(tmp_matrix);
    }

    bool EqMatrix(const Matrix &other) const {
        bool flag = true;
        if (rows_ != other.rows_ || cols_ != other.cols_)
            flag = false;
        else {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    if (fabs(matrix_[i][j] - other.matrix_[i][j]) > EPS)
                        flag = false;
                }
            }
        }
        return flag;
    }
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
        if (cols_ != other.rows_)
            throw runtime_error("Нельзя умножать несогласованные матрицы\n");
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

    pair<Matrix, Matrix> LU() {
        swp_.clear();
        int n = this->GetRows();
        Matrix U(*this);
        Matrix L(n, n);
        for (int k = 0; k < n; ++k) { // k - номер итерации в методе Гаусса, номер столбца, который зануляем
            int index = k;            // index - индекс max по модулю элемента в k столбце
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

    Matrix Solve(Matrix &C, Matrix &L, Matrix &U) {
        Matrix B(C);
        vector<int> swp = this->GetSwp();
        for (int i = 0; i < swp.size(); ++i) {
            swap(B(i), B(swp[i]));
        }
        int n = this->GetRows();
        // LUx = b
        // Lz = b
        Matrix Z(n, 1);
        for (int i = 0; i < n; ++i) {
            Z(i, 0) = B(i, 0);
            for (int j = 0; j < i; ++j) {
                Z(i, 0) -= L(i, j) * Z(j, 0);
            }
        }
        // Ux = z
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

    Matrix Solve(Matrix &C) {
        auto [L, U] = this->LU();
        return this->Solve(C, L, U);
    }

    double Determinant() {
        // detA = det(LU) = detL * detU = detU
        double result = 1;
        auto [L, U] = this->LU();
        for (int i = 0; i < rows_; ++i) {
            result *= U(i, i);
        }
        // так как при swap строк меняется знак определителя
        int sign = 0;
        vector<int> swp = this->GetSwp();
        for (int i = 0; i < swp.size(); ++i) {
            if (swp[i] != i)
                ++sign;
        }
        if (sign % 2 != 0)
            result = -result;
        return result;
    }

    Matrix InverseMatrix() {
        int n = this->GetRows();
        Matrix B(n, 1);
        Matrix result(n, n);
        for (int i = 0; i < n; ++i) {
            if (i > 0)
                B(i - 1, 0) = 0;
            B(i, 0) = 1;
            auto [L, U] = this->LU();
            Matrix res_i = this->Solve(B, L, U);
            for (int k = 0; k < n; ++k)
                result(k, i) = res_i(k, 0);
        }
        return result;
    }

    Matrix CalcComplements() const {
        Matrix result(*this);
        if (rows_ != cols_)
            throw runtime_error(
                "Матрицу алгебраических дополнений можно найти только для квадратной "
                "матрицы\n");
        if (rows_ == 1) {
            result.matrix_[0][0] = 1;
        } else {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    Matrix new_matrix = this->Minor(i, j);
                    double minor_det = new_matrix.Determinant();
                    result.matrix_[i][j] = pow(-1, i + j) * minor_det;
                }
            }
        }
        return result;
    }

    Matrix run_through_method(Matrix &B) {
        int n = this->rows_;
        vector<double> P, Q;                               // x_n = P_n * x_n+1 + Q_n
        P.push_back((-1) * (*this)(0, 1) / (*this)(0, 0)); // P[0] = -c1/b1
        Q.push_back(B(0, 0) / (*this)(0, 0));              // Q[0] = d1/b1
        for (int i = 1; i < n; ++i) {
            if (i == n - 1) {
                P.push_back(0); // c_n = 0
            } else {
                P.push_back((-1) * (*this)(i, i + 1) / ((*this)(i, i) + (*this)(i, i - 1) * P[i - 1])); // P_i = -c_i / (b_i + a_i * P_i-1)
            }
            Q.push_back((B(i, 0) - (*this)(i, i - 1) * Q[i - 1]) / ((*this)(i, i) + (*this)(i, i - 1) * P[i - 1])); // Q_i = (d_i - a_i * Q_i-1) / (b_i + a_i * P_i-1)
        }
        Matrix X(n, 1);
        X(n - 1, 0) = Q[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            X(i, 0) = P[i] * X(i + 1, 0) + Q[i];
        }
        return X;
    }

    double norm() {
        double res = 0;
        for (int i = 0; i < this->rows_; ++i) {
            double tmp_res = 0;
            for (int j = 0; j < this->cols_; ++j) {
                tmp_res += abs((*this)(i, j));
            }
            if (tmp_res > res)
                res = tmp_res;
        }
        return res;
    }

    pair<Matrix, int> simple_iterations(Matrix &B, double eps) {
        int n = this->rows_;
        Matrix Alpha(n, n);
        Matrix Beta(n, 1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Alpha(i, j) = (-1) * (*this)(i, j) / (*this)(i, i);
            }
            Alpha(i, i) = 0;
            Beta(i, 0) = B(i, 0) / (*this)(i, i);
        }
        Matrix X(n, 1), Prev_X(n, 1);
        int k = 1;
        Prev_X = Beta;
        X = Beta + Alpha * Prev_X;
        // eps_k = ||Alpha|| / (1 - ||Alpha||) * ||x_k - x_k-1||
        double eps_k = 0;
        double norm = Alpha.norm();
        if (norm >= 1) {
            eps_k = (X - Prev_X).norm();
        } else {
            eps_k = norm / (1 - norm) * (X - Prev_X).norm();
        }
        while (eps_k > eps) { // eps_k <= eps
            Prev_X = X;
            X = Beta + Alpha * X;
            if (norm >= 1) {
                eps_k = (X - Prev_X).norm();
            } else {
                eps_k = norm / (1 - norm) * (X - Prev_X).norm();
            }
            ++k;
        }
        return {X, k};
    }

    pair<Matrix, int> seidel(Matrix &R, double eps) {
        int n = this->rows_;
        Matrix Alpha(n, n), Beta(n, 1), E(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Alpha(i, j) = (-1) * (*this)(i, j) / (*this)(i, i);
            }
            Alpha(i, i) = 0;
            E(i, i) = 1;
            Beta(i, 0) = R(i, 0) / (*this)(i, i);
        }
        // Alpha = B + C
        Matrix C(n, n), B(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j < i)
                    B(i, j) = Alpha(i, j);
                else
                    C(i, j) = Alpha(i, j);
            }
        }
        // x_k+1 = (E - B)^-1 * C * x_k + (E - B)^-1 * Beta
        Matrix X(n, 1), Prev_X(n, 1);
        int k = 1;
        Prev_X = Beta;
        Matrix Tmp_Beta = (E - B).InverseMatrix() * Beta;
        Matrix Tmp_Alpha = (E - B).InverseMatrix() * C;
        X = Tmp_Alpha * Prev_X + Tmp_Beta;
        double eps_k = 0;
        double norm = Alpha.norm();
        if (norm >= 1) {
            eps_k = (X - Prev_X).norm();
        } else {
            eps_k = C.norm() / (1 - norm) * (X - Prev_X).norm();
        }
        // eps_k = ||C|| / (1 - ||Alpha||) * ||x_k - x_k-1||
        while (eps_k > eps) {
            Prev_X = X;
            X = Tmp_Alpha * Prev_X + Tmp_Beta;
            if (norm >= 1) {
                eps_k = (X - Prev_X).norm();
            } else {
                eps_k = C.norm() / (1 - norm) * (X - Prev_X).norm();
            }
            ++k;
        }
        return {X, k};
    }

    // helper
    double sum_square() {
        int n = this->rows_;
        double res = 0;
        // сумма квадратов недиагональных элементов
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j)
                    res += (*this)(i, j) * (*this)(i, j);
            }
        }
        return sqrt(res);
    }

    pair<pair<Matrix, Matrix>, int> jacobi_method(double eps) {
        int n = this->rows_;
        int k = 0;
        pair<int, int> max_index;
        Matrix A = *this;
        Matrix Self_Vectors(n, n);
        for (int i = 0; i < n; ++i) {
            Self_Vectors(i, i) = 1;
        }
        while (A.sum_square() > eps) {
            Matrix U(n, n);
            max_index = {1, 0}; // index abs(max_elem)
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j && abs(A(i, j)) > abs(A(max_index.first, max_index.second)))
                        max_index = {i, j};
                }
            }
            for (int i = 0; i < n; ++i) {
                U(i, i) = 1;
            }
            // phi = 1/2 * arctg (2 * a(i, j) / (a(i, i) - a(j, j)))
            // phi = PI/4, a(i, i) = a(j, j)
            double phi;
            if (A(max_index.first, max_index.first) == A(max_index.second, max_index.second))
                phi = M_PI / 4;
            else
                phi = 0.5 * atan(2 * A(max_index.first, max_index.second) / (A(max_index.first, max_index.first) - A(max_index.second, max_index.second)));
            U(max_index.first, max_index.first) = cos(phi);
            U(max_index.first, max_index.second) = (-1) * sin(phi);
            U(max_index.second, max_index.first) = sin(phi);
            U(max_index.second, max_index.second) = cos(phi);
            Matrix U_T = U.Transpose();
            // A^k+1 = U_T^k * A^k * U^k
            A = U_T.MulMatrixReturn(A).MulMatrixReturn(U);
            Self_Vectors.MulMatrix(U);
            ++k;
        }
        return {{A, Self_Vectors}, k};
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
            // v_1 = a_11 + sign(a11) * ||столбца 1||
            // v_i = a_i1
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
            // H = E - 2 * v * v_t / (v_t * v)
            H = E - V.MulMatrixReturn(V_T).MulMatrixReturn(2 / (V_T.MulMatrixReturn(V))(0, 0));
            A = H.MulMatrixReturn(A);
            Q = Q.MulMatrixReturn(H);
        }
        // Q^-1 = Q_T
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
            auto [Q, R] = A.qr_decomposition();
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
    Matrix operator*(const Matrix &other) {
        Matrix result(*this);
        result.MulMatrix(other);
        return result;
    }
    Matrix operator*(const double num) {
        Matrix result(*this);
        result.MulNumber(num);
        return result;
    }
    bool operator==(const Matrix &other) { return this->EqMatrix(other); }
    Matrix operator=(const Matrix &other) {
        if (this != &other) { // b = b
            Matrix tmp(other);
            this->SwapMatrix(tmp);
        }
        return *this;
    }
    void operator+=(const Matrix &other) { this->SumMatrix(other); }
    void operator-=(const Matrix &other) { this->SubMatrix(other); }
    void operator*=(const Matrix &other) { this->MulMatrix(other); }
    void operator*=(const double num) { this->MulNumber(num); }
    double &operator()(int i, int j) {
        if (i < 0 || i >= rows_ || j < 0 || j >= cols_)
            throw runtime_error("Индекс за пределами матрицы\n");
        return matrix_[i][j];
    }
    vector<double> &operator()(int row) { return matrix_[row]; }

    void ShowMatrix() const {
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                cout << matrix_[i][j] << " ";
            }
            cout << "\n";
        }
    }
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

#endif // MATRIX
