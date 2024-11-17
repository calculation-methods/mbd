#include <iostream>
#include <cmath>
#include <utility>
#include <vector>
using namespace std;

double F1(double x1, double x2)
{
	return (pow(x1, 2) + 16)*x2 - 64;
}

double F2(double x1, double x2)
{
	return pow((x1 - 2), 2) + pow((x2 - 2), 2) - 16;
}

double dF1_x1(double x1, double x2)
{
	return x2*x1*2;
}

double dF1_x2(double x1)
{
	return pow(x1, 2) + 16;
}

double dF2_x1(double x1)
{
	return 2*x1-4;
}

double dF2_x2(double x2)
{
	return 2*x2-4;
}

double max_diff(double a1, double b1, double a2, double b2)
{
	double max = abs(b1-a1);

	if (max > abs(b2 - a2))
		return max;
	else
		return abs(b2 - a2);
}

double determ2x(vector<vector<double>>& A)
{
	return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

pair<double, double> Newton(double a1, double a2, const double precision)
{
	cout << "\n__________________________________________________Newton Method________________________________________________";

	vector<vector<double>> A1 = { {F1(a1,a2), dF1_x2(a1)},     {F2(a1,a2),dF2_x2(a2)}  };
	vector<vector<double>> A2 = { {dF1_x1(a1,a2), F1(a1,a2)},  {dF2_x1(a1), F2(a1,a2)} };
	vector<vector<double>> J  = { {dF1_x1(a1,a2), dF1_x2(a1)}, {dF2_x1(a1),dF2_x2(a2)} };

	double b1 = a1 - determ2x(A1) / determ2x(J);
	double b2 = a2 - determ2x(A2) / determ2x(J);

	double diff = max_diff(a1, b1, a2, b2);

	cout << "\na1: " << a1 << "\t\ta2: " << a2 << "\t\tb1: " << b1 << "\t\tb2: " << b2 << "\t\tdiff: " << diff;
	a1 = b1; a2 = b2;

	while (precision < diff)
	{
		A1 = { {F1(a1,a2), dF1_x2(a1)},     {F2(a1,a2),dF2_x2(a2)} };
		A2 = { {dF1_x1(a1,a2), F1(a1,a2)},  {dF2_x1(a1), F2(a1,a2)} };
		J = { {dF1_x1(a1,a2), dF1_x2(a1)}, {dF2_x1(a1),dF2_x2(a2)} };

		b1 = a1 - determ2x(A1) / determ2x(J);
		b2 = a2 - determ2x(A2) / determ2x(J);


		diff = max_diff(a1, b1, a2, b2);

		cout << "\na1: " << a1 << "\t\ta2: " << a2 << "\t\tb1: " << b1 << "\t\tb2: " << b2 << "\t\tdiff: " << diff;
		a1 = b1; a2 = b2;
	}

	return make_pair(b1, b2);
}

double phi1(double a2)
{
	return sqrt(16 - pow((a2 - 2), 2)) + 2;
}

double phi2(double a1)
{
	return 64 / (pow(a1, 2) + 16);
}

pair<double, double> simpleIteration(double a1, double a2, const double precision)
{
	cout << "\n____________________________________________Simple Iteration Method____________________________________________";
	double b1 = phi1(a2);
	double b2 = phi2(a1);

	double diff = max_diff(a1, b1, a2, b2);

	cout << "\na1: " << a1 << "\t\ta2: " << a2 << "\t\tb1: " << b1 << "\t\tb2: " << b2 << "\t\tdiff: " << diff;
	a1 = b1; a2 = b2;

	while (precision < diff)
	{
		b1 = phi1(a2);
		b2 = phi2(a1); 

		diff = max_diff(a1, b1, a2, b2);

		cout << "\na1: " << a1 << "\t\ta2: " << a2 << "\t\tb1: " << b1 << "\t\tb2: " << b2 << "\t\tdiff: " << diff;
		a1 = b1; a2 = b2;
	}

	return make_pair(b1, b2);
}

int main()
{
	double a1;
	cout << "Input a1: ";
	cin >> a1;

	double a2;
	cout << "Input a2: ";
	cin >> a2;

	double precision;
	cout << "Input precision: ";
	cin >> precision;

	simpleIteration(a1, a2, precision);

	cout << '\n';

	Newton(a1, a2, precision);
}
