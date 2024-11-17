#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>


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

Matrix operator * (double alpha, const Matrix &A){
    Matrix res;
    int n = A.size();
    int m = (*A.begin()).size();
    if(A.empty()){
        return res;
    }
    res.resize(n, std::vector<double>(m));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            res[i][j] = alpha * A[i][j];
        }
    }
    return res;
}


Matrix operator - (const Matrix &a, const Matrix &b){
    Matrix res;
    int n = a.size();
    int m = (*a.begin()).size();
    if(a.empty() || b.empty()){
        return res;
    }
    res.resize(n, std::vector<double>(m));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            res[i][j] = a[i][j] - b[i][j];
        }
    }
    return res;
}


std::pair<double, double> discriminant(double a, double b, double c){
    double D = b*b - 4 * a * c;
    if(D < 0){
        std::cout << "Error: discriminant less than zero" << std::endl;
        return std::make_pair(0, 0);
    }
    double x1 = (-b + sqrt(D)) / (2 * a); 
    double x2 = (-b - sqrt(D)) / (2 * a);
    return std::make_pair(x1, x2);
}



class QRMatrix{
    public:
        void print_vector(const std::vector<double> &b);
        void print_matrix(const Matrix &matrix);
        QRMatrix(std::string filename);
        std::vector<double> eigenVal;
    private:
        int n;
        double epsilon;
        Matrix data;
        Matrix E();
        void transpose(Matrix &matrix);

        Matrix getH(const Matrix &A, int col);
        double getNorm(const std::vector<double> &b);
        Matrix Q;
        Matrix R;
        void decompose(const Matrix &A);
        Matrix prevIter;
        std::pair<double, double> getRoots(const Matrix &A, int i);
        void getEigenVal(const Matrix &A);
};


QRMatrix::QRMatrix(std::string filename){
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
    eigenVal.resize(n);
    getEigenVal(data);
}
 

Matrix QRMatrix::E(){
    Matrix matrix(n, std::vector<double>(n));
    for(int i = 0; i < n; i++){
        matrix[i][i] = 1;
    }
    return matrix;
}


void QRMatrix::transpose(Matrix &matrix){
    Matrix res((*matrix.begin()).size(), std::vector<double>(matrix.size()));
    for(int i = 0; i < (*matrix.begin()).size(); i++){
        for(int j = 0; j < matrix.size(); j++){
            res[i][j] = matrix[j][i];
        }
    }
    matrix = res;
}


void QRMatrix::print_matrix(const Matrix &matrix){
    if(matrix.empty()){
        std::cout << "Error: couldn't print empty matrix" << std::endl;
        return;
    }
    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j < (*matrix.begin()).size(); j++){
            std::cout << std::round(matrix[i][j]*10000)/10000.0  << '\t';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


void QRMatrix::print_vector(const std::vector<double> &b){
    if(b.empty()){
        std::cout << "Error: couldn't print empty vector" << std::endl;
        return;
    }
    for(int i = 0; i < n; i++){
            std::cout << std::round(b[i]*100)/100.0  << '\t';
    }
    std::cout << std::endl << std::endl;
}


Matrix QRMatrix::getH(const Matrix &A, int col){
    std::vector<double> a(n);
    for(int i = 0; i < n; i++){ 
        a[i] = A[i][col];
    }
    Matrix v(1, std::vector<double>(n));
    v[0][col] = (a[col] < 0)? a[col] - getNorm(a): a[col] + getNorm(a);
    for(int i = col + 1; i < n; i++){
        v[0][i] = a[i];
    }
    Matrix vT(v);
    transpose(v);
    Matrix vTv = vT*v;
    Matrix vvT = v*vT;
    Matrix temp = (2 / vTv[0][0])*vvT;
    Matrix indentity = E();
    return indentity - temp;
}


double QRMatrix::getNorm(const std::vector<double> &b){
    double sum = 0;
    for(double elem: b){
        sum += elem*elem;
    }
    return sqrt(sum);
}


void QRMatrix::decompose(const Matrix &A){
    Q = E();
    R = A;
    for(int i = 0; i < n - 1; i++){
        Matrix H = getH(R, i);
        Q = Q * H;
        R = H * R;
    }
}


std::pair<double, double> QRMatrix::getRoots(const Matrix &A, int i){
    double a, b, c;
    a = 1;
    b = -A[i][i] - A[i+1][i+1];
    c = A[i][i]*A[i+1][i+1] - A[i][i+1]*A[i+1][i];
    std::pair<double, double> lambdas = discriminant(a, b, c);
    return lambdas;
}


void QRMatrix::getEigenVal(const Matrix &A){
    Matrix Ak = A;
    int i = 0;
    int complexCol = -1;
    int cnt = 0;
    while(i < n){
        decompose(Ak);
        Ak = R * Q;
        std::vector<double> a(n - i - 1);
        for(int j = i + 1; j < n; j++){     
            a[j - i - 1] = Ak[j][i];
        }
        if(getNorm(a) < epsilon){
            eigenVal[i] = Ak[i][i];
            i++;
        } else {
            std::vector<double> b(n - i - 1);
            for(int j = i + 2; j < n; j++){
                b[j - i - 2] = Ak[j][i];
            }
            bool cond0 = getNorm(b) < epsilon;
            if(complexCol == i){
                std::pair<double, double> lambdas = getRoots(A, i);
                std::pair<double, double> prevLambdas = getRoots(prevIter, i);
                bool cond1 = fabs(lambdas.first - prevLambdas.first) < epsilon;
                bool cond2 = fabs(lambdas.second - prevLambdas.second) < epsilon;
                if(cond0 && cond1 && cond2){
                    eigenVal[i] = lambdas.first;
                    eigenVal[i+1] = lambdas.second;
                    i += 2;
                }
            }
            if(cond0){
                complexCol = i;
            }
        }
        prevIter = Ak;cnt++;
    }
}


int main(int argc, char **argv){
    QRMatrix matrix = QRMatrix(argv[1]);
    matrix.print_vector(matrix.eigenVal);
    return 0;
}