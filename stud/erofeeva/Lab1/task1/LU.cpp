#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>

using Matrix = std::vector<std::vector<double>>;


class LUMatrix{
    public:
        std::vector<double> x;
        Matrix L;
        Matrix U;
        Matrix inv;
        LUMatrix(std::string filename);
        void print_matrix(Matrix &matrix);
        void print_vector(std::vector<double> &b);
        double get_determinant(Matrix &matrix);
    private:
        int n;
        Matrix data;
        std::vector<double> b;
        void decompose();
        std::vector<double> solve_system(std::vector<double> &b);
        void transpose(Matrix &matrix);
        void get_inv();
};


LUMatrix::LUMatrix(std::string filename){
    std::ifstream idescr(filename);
        //вычисляем размер матрицы
        double d; int cnt = 0;
        while(idescr >> d){
            cnt++;
        }
        n = (-1 + sqrt(1 + 4*cnt)) / 2;
    idescr.close();
    idescr.open(filename);
        // вводим систему
        data.resize(n, std::vector<double>(n));
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                idescr >> data[i][j];
            }
        }
        b.resize(n);
        for(int i = 0; i < n; i++){
            idescr >> b[i];
        }
    idescr.close();
    // LU разложение
    L.resize(n, std::vector<double>(n));
    U.resize(n, std::vector<double>(n));
    U = data;
    decompose();
    // нахождение решения системы
    x = solve_system(b);
    // нахождение обратной матрицы
    get_inv();
}


void LUMatrix::print_matrix(Matrix &matrix){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            std::cout << std::round(matrix[i][j]*100)/100.0  << '\t';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void LUMatrix::print_vector(std::vector<double> &b){
    for(int i = 0; i < n; i++){
            std::cout << std::round(b[i]*100)/100.0  << '\t';
    }
    std::cout << std::endl << std::endl;
}


void LUMatrix::decompose(){
    for(int k = 0; k < n; k++){
        L[k][k] = 1;
        for(int i = k + 1; i < n; i++){
            double mu = U[i][k]/U[k][k];
            L[i][k] = mu;
            for(int j = k; j < n; j++){
                U[i][j] -= mu*U[k][j];
            }
        }
    }
}


std::vector<double> LUMatrix::solve_system(std::vector<double> &rs){
    // Находим z из Lz = b;
    std::vector<double> z(n, 0);
    for(int i = 0; i < n; i++){
        z[i] = rs[i];
        for(int j = 0; j < i; j++){
            z[i] -= L[i][j]*z[j];
        }
    }
    // Находим x из Ux = z;
    std::vector<double> x(n,0);
    for(int i = n - 1; i >= 0; i--){
        x[i] = z[i];
        for(int j = i + 1; j < n; j++){
            x[i] -= U[i][j]*x[j];
        }
        x[i] /= U[i][i];
    }
    return x;
}


void LUMatrix::transpose(Matrix &matrix){
    Matrix res(n, std::vector<double>(n, 0));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            res[i][j] = matrix[j][i];
        }
    }
    matrix = res;
}


void LUMatrix::get_inv(){
    Matrix E(n, std::vector<double>(n));
    for(int i = 0; i < n; i++){
        E[i][i] = 1;
    }
    std::vector<double> res;
    for(std::vector<double> e: E){
        inv.push_back(solve_system(e));
    }
    transpose(inv);
}


double LUMatrix::get_determinant(Matrix &matrix){
    double provide = matrix[0][0];
    for(int i = 1; i < n; i++){
        provide *= matrix[i][i];
    }
    return provide;
}


int main(int argc, char** argv){
    LUMatrix matrix = LUMatrix(argv[1]);

    // Вывод L и U
    std::cout << std::endl;
    std::cout << "\tМатрица L:" << std::endl;
    matrix.print_matrix(matrix.L);
    std::cout << "\tМатрица U:" << std::endl;
    matrix.print_matrix(matrix.U);

    // Вывод решения системы
    std::cout << "\tРешение системы:" << std::endl;
    matrix.print_vector(matrix.x);

    // Нахождение и вывод определителя
    std::cout << "Детерминант: ";
    std::cout << matrix.get_determinant(matrix.U) << std::endl;
    std::cout << std::endl;

    // Вывод обратной матрицы
    std::cout << "\tОбратная матрица:" << std::endl;
    matrix.print_matrix(matrix.inv);
    return 0;
}