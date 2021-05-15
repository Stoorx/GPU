__kernel void mul(__global int* m1, __global int* m2, __global int* m3, int m, int k, int n) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int sum = 0;
    
    for (int i = 0; i < k; i++)
        sum += m1[row * k + i] * m2[i * n + col];
    
    m3[col * m + row] = sum * 4;
}