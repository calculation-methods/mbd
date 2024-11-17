#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<vector<double>> transposing(vector<vector<double>> matrix) {
    int n = matrix.size();
    vector<vector<double>> result(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i][j] = matrix[j][i];
        }
    }
    return result;
}

double normalise(vector <double> vector){
    int n = vector.size();
    double mag = 0;
    for (int i=0; i<n; i++){
        mag += vector[i] * vector[i];
    }

    return sqrt(mag);
}

double dot_product(vector<double> first, vector<double> second){
    double sum = 0;
    int n = first.size();
    for (int i=0; i<n; i++){
        sum += first[i] * second[i];
    }
    return sum;
}

void standardize_matrix(vector<vector <double>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i<n; i++){
        double mag = normalise(matrix[i]);
        for (int j=0;j<n;j++){
            matrix[i][j] /= mag;
        }
    }
}

vector <double> subtract_vector(vector<double> first, vector<double> second){
    int n = first.size();
    vector <double> res(n,0);
    for (int i = 0; i<n; i++){
        res[i] = first[i] - second[i];
    }
    return res;
}

vector<vector <double>> orthogonalize(vector<vector<double>> matrix) {
    int n = matrix.size();
    vector <vector <double>> transposed_matrix = transposing(matrix);
    vector <vector <double>> B(n, vector<double>(n,0));
    B[0] = transposed_matrix[0];
    for (int i=1; i<n; i++){
        vector <double> projection(n,0);
        for (int count=0; count<i; count++){
            double k = dot_product(transposed_matrix[i], B[count]) / dot_product(B[count], B[count]);
            for (int j = 0; j<n; j++){
                projection[j] += k * B[count][j];
            }
        }
        B[i] = subtract_vector(transposed_matrix[i], projection);
    }
    return B;
}

vector <vector <double>> matrix_product(vector <vector <double>> A, vector <vector <double>> B){
    int n = A.size();
    vector <vector <double>> R(n, vector<double>(n,0));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                R[i][j] += A[i][k] * B[k][j];
    return R;
}

void decompose_QR(vector<vector <double>> &matrix, vector<vector <double>> &Q, vector<vector <double>> &R){
    Q = orthogonalize(matrix);
    standardize_matrix(Q);
    Q = transposing(Q);
    R = matrix_product(transposing(Q), matrix);
}

vector <double> calculate_eigenvalues(vector<vector<double>> matrix, double epsilon)
{
    int n = matrix.size();
    vector <double> prev_eigenvalues(n);
    vector<vector<double>> Q, R;
    vector <double> cur_eigenvalues(n,0);
    while (true) {
        decompose_QR(matrix,Q,R);
        matrix = matrix_product(R,Q);
        for (int i = 0; i < n; i++) {
            if (i < n - 1 && abs(matrix[i + 1][i]) > 1e-7) {
                double b = -(matrix[i][i] + matrix[i + 1][i + 1]);
                double c = matrix[i][i] * matrix[i + 1][i + 1] - matrix[i][i + 1] * matrix[i + 1][i];
                double discriminant = b * b - 4 * c;

                if (discriminant > 0) {
                    cur_eigenvalues[i] = 0.5 * (-b - sqrt(discriminant));
                    cur_eigenvalues[i + 1] = 0.5 * (-b + sqrt(discriminant));
                    i++;
                } else {
                    cur_eigenvalues[i] = (-b / 2);
                    cur_eigenvalues[i + 1] = (-b / 2);
                    i++;
                }
            } else {
                cur_eigenvalues[i] = matrix[i][i];
            }
        }
        bool ok = true;
        for (int i = 0; i < n; i++) {
            ok = ok && abs(cur_eigenvalues[i] - prev_eigenvalues[i]) < epsilon;
        }
        if (ok)
            break;
        prev_eigenvalues = cur_eigenvalues;
    }
    return prev_eigenvalues;
}

int main() {
    vector<vector<double>> matrix = {
            {2, -4, 5},
            {-5, -2, -3},
            {1, -8, -3}
    };

    vector<vector<double>> Q, R;
    decompose_QR(matrix, Q, R);

    double epsilon = 0.1;
    vector<double> eigenvalues = calculate_eigenvalues(matrix, epsilon);
    cout << "Eigenvalues:" << endl;
    for(int i = 0; i < 3; i++){
        cout << eigenvalues[i] << endl;
    }
    return 0;
}
