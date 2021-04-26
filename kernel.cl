__kernel void mul(__global float* m1, __global float* m2, __global float* m3, int m, int k, int n) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0;
    
    for (int i = 0; i < k; i++)
        sum += m1[row * k + i] * m2[i * n + col];
    
    m3[col * n + row] = sum;
}