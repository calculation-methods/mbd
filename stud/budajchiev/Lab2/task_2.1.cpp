#include <iostream>
#include <cmath>
using namespace std;

double iterationalF(double x)
{
	return log(sqrt(1 - x*x) + 0.1);
}

double simpleIteration(double a, const double precision)
{
	cout << "\n______________________Simple Iteration Method______________________";
	double b = iterationalF(a);
	double diff = abs(b - a);
	cout << "\na: " << a << "\t\tb: " << b << "\t\tdiff: " << diff;
	a = b;

	while (precision < diff)
	{
		b = iterationalF(a);
		diff = abs(b - a);
		cout << "\na: " << a << "\t\tb: " << b << "\t\tdiff: " << diff;
		a = b;
	}

	return b;
}

double F(double x)
{
	return (sqrt(1 - x * x) - pow(M_E, x) + 0.1);
}

double dF(double x)
{
	return ((-x / sqrt(1 - x * x)) - pow(M_E, x));
}

double Newton(double a, const double precision)
{
	cout << "\n____________________________Newton Method__________________________";
	double b = a - F(a)/dF(a);
	double diff = abs(b - a);
	cout << "\na: " << a << "\t\tb: " << b << "\t\tdiff: " << diff;
	a = b;

	while (precision < diff)
	{
		b = a - F(a) / dF(a);
		diff = abs(b - a);
		cout << "\na: " << a << "\t\tb: " << b << "\t\tdiff: " << diff;
		a = b;
	}

	return b;
}

int main()
{
	double a;
	cout << "Input a: ";
	cin >> a;

	double precision;
	cout << "Input precision: ";
	cin >> precision;

	simpleIteration(a, precision);

	cout << '\n';

	Newton(a, precision);
}
