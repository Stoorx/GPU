__kernel void transposeSimple(__global int* m1, __global int* m2, int m, int n) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    m2[col * n + row] = m1[row * m + col];
}

__kernel void transposeLocal(__global int* m1, __global int* m2, int m, int n) {
    int locRow       = get_local_id(0);
    int locCol       = get_local_id(1);
    int workgroupRow = get_group_id(0);
    int workgroupCol = get_group_id(1);
    int globalRow1   = workgroupRow * 16 + locRow;
    int globalCol1   = workgroupCol * 16 + locCol;
    int globalRow2   = workgroupCol * 16 + locRow;
    int globalCol2   = workgroupRow * 16 + locCol;
    
    __local int buffer[16][16];
    if(globalCol1 < m && globalRow1 < n) {
        buffer[locCol][locRow] = m1[globalRow1 * m + globalCol1];
        barrier(CLK_LOCAL_MEM_FENCE);
        m2[globalRow2 * n + globalCol2] = buffer[locRow][locCol];
    }
}

__kernel void transposeLocalBanksafe(__global int* m1, __global int* m2, int m, int n) {
    int locRow       = get_local_id(0);
    int locCol       = get_local_id(1);
    int workgroupRow = get_group_id(0);
    int workgroupCol = get_group_id(1);
    int globalRow1   = workgroupRow * 16 + locRow;
    int globalCol1   = workgroupCol * 16 + locCol;
    int globalRow2   = workgroupCol * 16 + locRow;
    int globalCol2   = workgroupRow * 16 + locCol;
    
    __local int buffer[16][16 + 1];
    if(globalCol1 < m && globalRow1 < n) {
        buffer[locCol][locRow] = m1[globalRow1 * m + globalCol1];
        barrier(CLK_LOCAL_MEM_FENCE);
        m2[globalRow2 * n + globalCol2] = buffer[locRow][locCol];
    }
}