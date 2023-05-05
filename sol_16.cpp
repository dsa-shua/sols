#include "matmul.h"
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <vector>

const int num_threads = 8;
int row_per_thread;
int col_per_thread;
int segment_length;

std::vector<std::thread> threads;

int* generate_0(int r, int c){
        // Generates a r x c matrices with 0's
        int* matrix = new int[r*c];
        for(int i = 0; i < r; i++)
                for(int j = 0; j < c; j++)
                        matrix[i*c + j] = 0;
        return matrix;
}

void _transpose(const int* const target, int* destination, int r, int c, int tid){
  int start_row = tid*row_per_thread;
  int end_row = start_row + row_per_thread;
  
  for (int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      destination[j*r + i] = target[i*c + j];
}

void transpose(const int* const target, int* destination, int r, int c){
  threads.clear();
  for(int i = 0; i < num_threads; i++)
    threads.push_back(std::thread(_transpose, target, destination, r, c, i));
  
  for(auto& thread: threads)
    thread.join();
  threads.clear();
}



void tA0_B0(const int* const mat1, int* mat2, int* const mat3,int r, int c, int tid){
  int start_row = tid*(segment_length);
  int end_row = segment_length + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[i*r*2 + j] += mat1[i*r*2 + k] * mat2[j*c*2 + k]; // no shifting
}

void tA0_B1(const int* const mat1, int* mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = segment_length+ tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[i*r*2 + j+c] += mat1[i*r*2 + k] * mat2[(j+r)*c*2 + k]; // keep A, row B
}

void tA1_B2(const int* const mat1, int* mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[i*r*2 + j] += mat1[i*r*2 + k+c] * mat2[j*c*2 + k+c]; // col A, col B
}

void tA1_B3(const int* const mat1, int* mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length)+ tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[i*r*2 + j+c] += mat1[i*r*2 + k+c] * mat2[(j+r)*c*2 + k+c]; // col A, rowB col B
}

void tA2_B0(const int* const mat1, int* mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[(i+r)*r*2 + j] += mat1[(i+r)*r*2 + k] * mat2[j*c*2 + k]; // row A
}

void tA2_B1(const int* const mat1, int* mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[(i+r)*r*2 + j+c] += mat1[(i+r)*r*2 + k] * mat2[(j+r)*c*2 + k]; // row A, row B
}

void tA3_B2(const int* const mat1, int* mat2, int* const mat3, int  r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[(i+r)*r*2 + j] += mat1[(i+r)*r*2 + k+c] * mat2[j*c*2 + k+c]; // col row A, col B
}

void tA3_B3(const int* const mat1, int* mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[(i+r)*r*2 + j+c] += mat1[(i+r)*r*2 + k+c] * mat2[(j+r)*c*2 + k+c]; // col row A, col row B

}

void proc0(const int* const mat1, int* mat2, int* const mat3, int r, int  c){
  std::thread t0(tA0_B0, mat1,mat2,mat3,r,c,0);
  std::thread t1(tA0_B1, mat1,mat2,mat3,r,c,0);
  std::thread t2(tA2_B0, mat1,mat2,mat3,r,c,0);
  std::thread t3(tA2_B1, mat1,mat2,mat3,r,c,0);
  
  std::thread t4(tA0_B0, mat1,mat2,mat3,r,c,1);
  std::thread t5(tA0_B1, mat1,mat2,mat3,r,c,1);
  std::thread t6(tA2_B0, mat1,mat2,mat3,r,c,1);
  std::thread t7(tA2_B1, mat1,mat2,mat3,r,c,1);
  
  std::thread t8(tA0_B0, mat1,mat2,mat3,r,c,2);
  std::thread t9(tA0_B1, mat1,mat2,mat3,r,c,2);
  std::thread t10(tA2_B0, mat1,mat2,mat3,r,c,2);
  std::thread t11(tA2_B1, mat1,mat2,mat3,r,c,2);
  
  std::thread t12(tA0_B0, mat1,mat2,mat3,r,c,3);
  std::thread t13(tA0_B1, mat1,mat2,mat3,r,c,3);
  std::thread t14(tA2_B0, mat1,mat2,mat3,r,c,3);
  std::thread t15(tA2_B1, mat1,mat2,mat3,r,c,3);
  
  
  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  t6.join();
  t7.join();
  t8.join();
  t9.join();
  t10.join();
  t11.join();
  t12.join();
  t13.join();
  t14.join();
  t15.join();
}

void proc1(const int* const mat1, int* mat2, int* const mat3, int r, int c){

  std::thread t0(tA1_B2, mat1,mat2,mat3,r,c,0);
  std::thread t1(tA1_B3, mat1,mat2,mat3,r,c,0);
  std::thread t2(tA3_B2, mat1,mat2,mat3,r,c,0);
  std::thread t3(tA3_B3, mat1,mat2,mat3,r,c,0);
  
  std::thread t4(tA1_B2, mat1,mat2,mat3,r,c,1);
  std::thread t5(tA1_B3, mat1,mat2,mat3,r,c,1);
  std::thread t6(tA3_B2, mat1,mat2,mat3,r,c,1);
  std::thread t7(tA3_B3, mat1,mat2,mat3,r,c,1);
  
  std::thread t8(tA1_B2, mat1,mat2,mat3,r,c,2);
  std::thread t9(tA1_B3, mat1,mat2,mat3,r,c,2);
  std::thread t10(tA3_B2, mat1,mat2,mat3,r,c,2);
  std::thread t11(tA3_B3, mat1,mat2,mat3,r,c,2);
  
  std::thread t12(tA1_B2, mat1,mat2,mat3,r,c,3);
  std::thread t13(tA1_B3, mat1,mat2,mat3,r,c,3);
  std::thread t14(tA3_B2, mat1,mat2,mat3,r,c,3);
  std::thread t15(tA3_B3, mat1,mat2,mat3,r,c,3);
  
  
  t0.join();
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
  t6.join();
  t7.join();
  t8.join();
  t9.join();
  t10.join();
  t11.join();
  t12.join();
  t13.join();
  t14.join();
  t15.join();
}



void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n, const int m) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < m; k++)
        matrixC[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
}


void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n, const int m) {
        
  row_per_thread = n / num_threads;
  int r2 = m/2;
  int c2 = n/2;
  segment_length r2/4; // 16 threads
  int* transposed = generate_0(n,m);
  transpose(matrixB, transposed,n,m);

  proc0(matrixA,transposed, matrixC, r2,c2);
  proc1(matrixA,transposed, matrixC, r2,c2);

}
