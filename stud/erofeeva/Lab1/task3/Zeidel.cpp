#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using Matrix = std::vector<std::vector<double>>;


class IterationMatrix{
    public:
        IterationMatrix(std::string filename, bool useZeidel);
        std::vector<double> solve;
        void print_vector(const std::vector<double> &x);
    private:
        int n;
        double epsilon;
        Matrix data;
        std::vector<double> b;
        void toYakobi(Matrix &a, std::vector<double> &d);
        double getNorm(const Matrix &a);
        double getDiffNorm(const std::vector<double> &xk, const std::vector<double> &xk_inc);
        std::vector<double> simpleIteration(double norm);
        std::vector<double> zeidel(double norm);
};


IterationMatrix::IterationMatrix(std::string filename, bool useZeidel){
    std::ifstream idescr(filename);
        // Вычисляем размер матрицы
        int cnt = 0;
        while(idescr >> epsilon){
            cnt++;
        }
        n = (-1 + sqrt(1 + 4*(cnt-1))) / 2;
    idescr.close();
    idescr.open(filename);
        // Вводим систему
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

    // Приведение к эквивалентому виду
    toYakobi(data, b);

    // Вычисляем норму матрицы А
    double g = getNorm(data);
    if(g >= 1){
        std::cout << "\tУсловие сходимости не выполнено!" << std::endl;
    } else {
        // Находим решение системы с точностью epsilon
        solve = useZeidel? zeidel(g) : simpleIteration(g);
    }
}


void IterationMatrix::toYakobi(Matrix &a, std::vector<double> &d){
    for(int i = 0; i < n; i++){
        if(!a[i][i]){
            break;
        }
        for(int j = 0; j < n; j++){
            a[i][j] = (i == j)? a[i][i]: -a[i][j]/a[i][i];
        }
        d[i] /= a[i][i];
        a[i][i] = 0;
    }
}


double IterationMatrix::getNorm(const Matrix &a){
    double g = 0;
    for(int i = 0; i < n; i++){
        double sum = 0;
        for(int j = 0; j < n; j++){
            sum += fabs(a[i][j]);
        }
        g = (sum > g)? sum: g;
    }
    return g;
}


double IterationMatrix::getDiffNorm(const std::vector<double> &xk, const std::vector<double> &xk_inc){
    double f = 0;
    for(int i = 0; i < n; i++){
        if(fabs(xk[i] - xk_inc[i]) > f){
            f = fabs(xk[i] - xk_inc[i]);
        }
    }
    return f;
}


std::vector<double> IterationMatrix::simpleIteration(double norm){
    std::vector<double> xk(n);
    std::vector<double> xk_inc(b);
    double precision = getDiffNorm(xk, xk_inc) * norm / (1 - norm);
    int k = 0; 
    while(precision > epsilon) {
        xk = xk_inc;
        for(int i = 0; i < n; i++){
            xk_inc[i] = b[i];
            for(int j = 0; j < n; j++){
                xk_inc[i] += data[i][j]*xk[j];
            }
        }
        precision = getDiffNorm(xk, xk_inc) * norm / (1 - norm);
        k++;
    }; 
    std::cout << "Выполнено " << k << " итераций" << std::endl;
    return xk_inc;
}


std::vector<double> IterationMatrix::zeidel(double norm){
    std::vector<double> xk(b);
    std::vector<double> xk_inc(n);
    double precision = getDiffNorm(xk, xk_inc) * norm / (1 - norm);
    int k = 0;
    while(precision > epsilon) {
        xk = xk_inc;
        for(int i = 0; i < n; i++){
            xk_inc[i] = b[i];
            for(int j = 0; j < n; j++){
                xk_inc[i] += data[i][j]*xk_inc[j];
            }
        }
        precision = getDiffNorm(xk, xk_inc) * norm / (1 - norm);
        k++;
    };
    std::cout << "Выполнено " << k << " итераций методом Зейделя" << std::endl;
    return xk_inc;
}


void IterationMatrix::print_vector(const std::vector<double> &x){
    for(int i = 0; i < n; i++){
            std::cout << std::round(x[i]*100)/100.0  << '\t';
    }
    std::cout << std::endl << std::endl;
}




int main(int argc, char **argv){
    std::string method = argv[1];
    IterationMatrix matrix = IterationMatrix(argv[2], method == "-z");
    if(matrix.solve.empty()){
        std::cout << "Решение не найдено." << std::endl;
    } else {
        std::cout << "Решение системы:" << std::endl;
        matrix.print_vector(matrix.solve);
    }
    return 0;
}