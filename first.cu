#include <iostream>
#include <cuda.h>
using namespace std;

_global_ void AddIntsCUDA(int *a, int *b)
{
a[0] += b[0]
}

int main()
{
int a=5, b=9;
int *d_a, *d_b;
cudaMalloc(&d_a, sizeof(int));
cudaMalloc(

return 0;
}
