#include "matrix.h"

const Matrix& Matrix::operator=(const Matrix& right) {
    _matrix = right._matrix;
    n_size = right.n_size;
    m_size = right.m_size;
    return *this;
}


vector<double>& Matrix::operator[](const int index) {
    return _matrix[index];
}

const vector<double>& Matrix::operator[](const int index) const {
    return _matrix[index];
}


std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.n_size; ++i) {
        os << endl;
        os.width(8);
        os << matrix[i][0];
        for (int j = 1; j < matrix.m_size; ++j) {
            os << '\t';
            os.width(8);
            os << matrix[i][j];
        }
    }
    os << endl;
    return os;
}


const Matrix operator+(const Matrix& left, const Matrix& right) {
    if (left.n_size != right.n_size || left.m_size != right.m_size) {
        throw "Размеры матрицы не равны!";
    }
    Matrix ans(left.n_size, left.m_size);
    for (int i = 0; i < ans.n_size; ++i) {
        for (int j = 0; j < ans.m_size; ++j) {
            ans[i][j] = left._matrix[i][j] + right._matrix[i][j];
        }
    }
    return ans;
}

const Matrix operator-(const Matrix& left, const Matrix& right) {
    if (left.n_size != right.n_size || left.m_size != right.m_size) {
        throw "Размеры матрицы не равны!";
    }
    Matrix ans(left.n_size, left.m_size);
    for (int i = 0; i < ans.n_size; ++i) {
        for (int j = 0; j < ans.m_size; ++j) {
            ans[i][j] = left._matrix[i][j] - right._matrix[i][j];
        }
    }
    return ans;
}

const Matrix operator*(double left, const Matrix& right) {
    Matrix ans = right;
    for (int i = 0; i < ans.n_size; ++i) {
        for (int j = 0; j < ans.m_size; ++j) {
            ans[i][j] *= left;
        }
    }
    return ans;
}

const Matrix operator*(const Matrix& left, double right) {
    return right * left;
}




const Matrix operator*(const Matrix& left, const Matrix& right) {
    if (left.m_size != right.n_size) {
        throw "Размеры матрицы не равны!";
    }
    Matrix ans(left.n_size, right.m_size);
    for (int i = 0; i < ans.n_size; ++i) {
        for (int j = 0; j < ans.m_size; ++j) {
            for (int k = 0; k < left.m_size; ++k) {
                ans[i][j] += left._matrix[i][k] * right._matrix[k][j];
            }
        }
    }
    return ans;
}

const vector<double> operator*(const Matrix& left, const vector<double>& right) {
    if (left.m_size != (int)right.size()) {
        throw "Размеры матрицы не равны!";
    }
    vector<double> ans(left.n_size, 0.0);
    for (int i = 0; i < left.n_size; ++i) {
        for (int j = 0; j < left.m_size; ++j) {
            ans[i] += left._matrix[i][j] * right[j];
        }
    }
    return ans;
}



int Matrix::get_m() const {
    return m_size;
}

int Matrix::get_n() const {
    return n_size;
}

void Matrix::make_ones() {
    if (!is_quadratic()) {
        throw "Не квадратная матрица";
    }
    _matrix.assign(n_size, vector<double>(m_size, 0.0));
    for (int i = 0; i < n_size; ++i) {
        _matrix[i][i] = 1.0;
    }
}

void Matrix::transpose() {
    vector<vector<double>> temp(m_size, vector<double>(n_size));
    for (int i = 0; i < n_size; ++i) {
        for (int j = 0; j < m_size; ++j) {
            temp[j][i] = _matrix[i][j];
        }
    }
    swap(n_size, m_size);
    _matrix.swap(temp);
}

Matrix::Matrix() {
    _matrix.assign(1, vector<double>(1, 0));
    n_size = m_size = 1;
}

Matrix::Matrix(int n, int m) {
    _matrix.assign(n, vector<double>(m, 0));
    n_size = n;
    m_size = m;
}


bool Matrix::is_quadratic() const {
    return n_size == m_size;
}


bool Matrix::is_three_diagonal() const {
    if (!is_quadratic()) {
        return false;
    }
    for (int i = 0; i < n_size; ++i) {
        for (int j = 0; j < m_size; ++j) {
            if ((abs(i - j) > 1) && _matrix[i][j]) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::is_simmetric() const {
    if (!is_quadratic()) {
        return false;
    }
    for (int i = 0; i < n_size; ++i) {
        for (int j = i + 1; j < m_size; ++j) {
            if (_matrix[i][j] != _matrix[j][i]) {
                return false;
            }
        }
    }
    return true;
}

double Matrix::get_norm() const {
    double max = 0.0;
    for (int i = 0; i < n_size; ++i) {
        double ans = 0.0;
        for (int j = 0; j < m_size; ++j) {
            ans += abs(_matrix[i][j]);
        }
        max = max > ans ? max : ans;
    }
    return max;
}

double Matrix::get_upper_norm() const {
    double max = 0.0;
    for (int i = 0; i < n_size; ++i) {
        double ans = 0.0;
        for (int j = 0; j <= i; ++j) {
            ans += abs(_matrix[i][j]);
        }
        max = max > ans ? max : ans;
    }
    return max;
}