#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using Matrix = std::vector<std::vector<double>>;

Matrix operator * (const Matrix &a, const Matrix &b){
    Matrix res;
    int n = a.size();
    int p = b.size();
    int m = (*b.begin()).size();
    if(a.empty() || b.empty()){
        return res;
    }
    res.resize(n, std::vector<double>(m));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            for(int k = 0; k < p; k++){
                res[i][j] += a[i][k]*b[k][j];
            }
        }
    }
    return res;
}


class SymmetricalMatrix{
    public:
        void print_vector(const std::vector<double> &b);
        void print_matrix(const Matrix &matrix);
        SymmetricalMatrix(std::string filename);
        std::vector<double> eigen_val;
        Matrix eigen_vec;
    private:
        int n;
        double epsilon;
        Matrix data;
        bool checkSymmetric(Matrix &matrix);
        bool checkPresicion(Matrix &matrix);
        std::pair<int, int> maxElem(Matrix &matrix);
        Matrix E();
        void transpose(Matrix &matrix);
};



void SymmetricalMatrix::print_matrix(const Matrix &matrix){
    if(matrix.empty()){
        std::cout << "Error: couldn't print empty matrix" << std::endl;
        return;
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << std::round(matrix[i][j]*100)/100.0  << '\t';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void SymmetricalMatrix::print_vector(const std::vector<double> &b){
    if(b.empty()){
        std::cout << "Error: couldn't print empty vector" << std::endl;
        return;
    }
    for(int i = 0; i < n; i++){
            std::cout << std::round(b[i]*100)/100.0  << '\t';
    }
    std::cout << std::endl << std::endl;
}



bool SymmetricalMatrix::checkSymmetric(Matrix &matrix){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(matrix[i][j] != matrix[j][i]){
                return false;
            }
        }
    }
    return true;
}


SymmetricalMatrix::SymmetricalMatrix(std::string filename){
    std::ifstream idescr(filename);
        // Вычисляем размер матрицы
        int cnt = 0;
        while(idescr >> epsilon){
            cnt++;
        }
        n = sqrt(cnt - 1);
    idescr.close();
    idescr.open(filename);
        // Вводим систему
        data.resize(n, std::vector<double>(n));
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                idescr >> data[i][j];
            }
        }
    idescr.close();

    // Проверяем симметричность матрицы
    if(!checkSymmetric(data)){
        std::cout << "Error: the matrix is not symmetric" << std::endl;
    } else {
        // Создаем матрицу собственных векторов V = E
        eigen_vec = E();
        int k = 0;
        while(checkPresicion(data)){
            // Получаем максимальный недиагональный элемент
            std::pair<int, int> indexes = maxElem(data);
            int l = indexes.first; 
            int m = indexes.second;
            // Создаем матрицу поворота (U)
            Matrix U = E();
            // Находим угол поворота (phi)
            double phi;
            if(data[m][m] == data [l][l]){
                phi = atan(1);
            } else {
                phi = atan(2*data[l][m]/(data[l][l]-data[m][m]))/2;
            }
            // Инициализируем матрицу поворота
            U[l][l] = cos(phi);
            U[l][m] = -sin(phi);
            U[m][l] = sin(phi);
            U[m][m] = cos(phi);
            // Получаем транспонированную матрицу поворота (UT)
            Matrix UT = U;
            transpose(UT);
            // Находим диагональную матрицу (lambda = UT x data x U)
            data = UT*data*U;
            // Находим собственные векторы
            eigen_vec = eigen_vec*U;
            k++;
        }
        // Сохраняем собственные значения
        eigen_val.resize(n);
        for(int i = 0; i < n; i++){
            eigen_val[i] = data[i][i];
        }
        std::cout << "Выполнено " << k << " итераций\n" << std::endl;
    }
}


bool SymmetricalMatrix::checkPresicion(Matrix &matrix){
    double f = 0;
    for(int i = 0; i < n; i++){
        for(int j = i+1; j < n; j++){
            f += matrix[i][j]*matrix[i][j];
        }
    }
    return sqrt(f) > epsilon;
}


std::pair<int, int> SymmetricalMatrix::maxElem(Matrix &matrix){
    double maxElem = 0;
    int l = 0;
    int m = 1;
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            if(fabs(matrix[i][j]) > maxElem){
                maxElem = fabs(matrix[i][j]);
                l = i;
                m = j;
            }
        }
    }
    return std::make_pair(l, m); 
}


Matrix SymmetricalMatrix::E(){
    Matrix matrix(n, std::vector<double>(n));
    for(int i = 0; i < n; i++){
        matrix[i][i] = 1;
    }
    return matrix;
}


void SymmetricalMatrix::transpose(Matrix &matrix){
    Matrix res(n, std::vector<double>(n, 0));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            res[i][j] = matrix[j][i];
        }
    }
    matrix = res;
}


int main(int argc, char** argv){
    SymmetricalMatrix matrix = SymmetricalMatrix(argv[1]);
    std::cout << "Собственные значения: " << std::endl;
    matrix.print_vector(matrix.eigen_val);
    std::cout << "Собственные векторы: " << std::endl;
    matrix.print_matrix(matrix.eigen_vec);

    return 0;
}