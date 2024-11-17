#include "lu.hpp"

using namespace std;

using matrix = matrix_t<double>;
using vec = vector<double>;
using lu = lu_t<double>;

int main() {
    cout.precision(6);
    cout << fixed;
    int n;
    cin >> n;
    matrix a(n);
    cin >> a;
    lu lu_a(a);
    vec b(n);
    for (int i = 0; i < n; ++i) {
        cin >> b[i];
    }
    vec x = lu_a.solve(b);
    cout << "System solution:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "x" << i + 1 << " = " << x[i] << endl;
    }
    cout << "\nMatrix determinant: " << lu_a.get_det() << endl;
    cout << "\nInverse matrix:" << endl;
    cout << lu_a.inv_matrix();
}
