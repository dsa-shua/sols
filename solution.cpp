#include "matmul.h"
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <vector>

const int num_threads = 16;
int row_per_thread;
int col_per_thread;
int segment_length;
int items_per_thread;
std::vector<std::thread> threads;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*       HELPER FUNCTIONS       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int* generate_0(int r, int c){
        // Generates a r x c matrices with 0's
        int* matrix = new int[r*c];
        for(int i = 0; i < r; i++)
                for(int j = 0; j < c; j++)
                        matrix[i*c + j] = 0;
        return matrix;
}

void _transpose(const int* const target, int* destination, int r, int c, int tid){
  // Procedure for each thread during transposing
  int start_row = tid*row_per_thread;
  int end_row = start_row + row_per_thread;
  
  for (int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      destination[j*r + i] = target[i*c + j];
}

void transpose(const int* const target, int* destination, int r, int c){
// Transpose a matrix parallel by splitting the target matrix (the one
// to be parallelized) into <num_thread> rows.
//
// The resulting transposed matrix is saved on the destination
  threads.clear();
  for(int i = 0; i < num_threads; i++)
    threads.push_back(std::thread(_transpose, target, destination, r, c, i));
  
  for(auto& thread: threads)
    thread.join();
  threads.clear();
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*       COL WISE IMPLEMENTATION       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void _col_wise(int tid, const int* const mat1, const int* const mat2, int* const mat3, const int r, const int c){
// Procedure for each col wise thread mat_mul
  int start_col = tid*col_per_thread;
  int end_col = start_col + col_per_thread;

  for(int i = 0; i < r; i++)
    for(int j = start_col; j < end_col; j++) // parallelizing on cols
      for(int k = 0; k < r; k++)
        mat3[i*r+ j] += mat1[i*c +k] *mat2[k*r +j];
}

void col_wise(const int* const mat1, const int* const mat2, int* const mat3, const int r, const int c){
// Do matrix multiplication by splitting the matrix by its columns,
// give a subsection of that column to a thread.

  col_per_thread = c / num_threads;
  for(int tid = 0; tid < num_threads; tid++)
    threads.push_back(std::thread(_col_wise,tid,mat1,mat2,mat3,r,c));
  
  for(auto& thread: threads)
    thread.join();
  threads.clear();
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*       ROW WISE IMPLEMENTATION       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


void _row_wise(int tid, const int* const m1, const int* const m2, int* const m3, const int r, const int c){
  // procedure for each row_wise matmul thread
  int start_row = tid*row_per_thread;
  int end_row = start_row + row_per_thread;

  for(int i = start_row; i < end_row; i++) // parallelizing on rows
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        m3[i*r +j] += m1[i*c + k] * m2[k*r + j];
}

void row_wise(const int* const m1, const int* const m2,  int* const m3, const int r, const int c){
  // Do matmul by splitting rows into <num_threads> subsections
  // and assigning one subsection to a thread to work with
  row_per_thread = r / num_threads;
  for(int tid = 0; tid < num_threads; tid++)
    threads.push_back(std::thread(_row_wise,tid,m1,m2,m3,r,c));
  
  for(auto& thread: threads)
    thread.join();
  threads.clear();
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*     PARALLEL TRANSPOSE IMPLEMENTATION     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


void _parallel_transpose(const int* const mat1, int* mat2, int* const mat3, const int r, const int c, int tid){
  // procedure for parallel transpose threads
  int start_row = tid*row_per_thread;
  int end_row = start_row + row_per_thread;
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[i*r + j] += mat1[i*r + k] * mat2[j*c +k]; // notice the difference against non transpose

}

void parallel_transpose(const int* const mat1, const int* const mat2, int* const mat3, const int r, const int c){
  // Parallel Matrix Multiplication where the second matrix is transposed
  // to allow that current data array to fit into the cache line
  // We then split the work by rows to <num_threads>
  row_per_thread = r / num_threads;
  int* t_mat2 = generate_0(r,c);
  transpose(mat2, t_mat2,r,c);

  for(int tid = 0; tid < num_threads; tid++)
    threads.push_back(std::thread(_parallel_transpose, mat1,t_mat2,mat3,r,c,tid));
  
  for(auto& thread: threads)
    thread.join();
  threads.clear();
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*        BLOCKED MATRIX IMPLEMENTATION     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Functions of Blocked Matrix Submatrix Multiplication
void A0_B0(const int* const mat1, const int* const mat2, int* const mat3,int r, int c, int tid){
  int start_row = tid*(segment_length);
  int end_row = segment_length + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
  mat3[i*r*2 + j] += mat1[i*c*2 + k] * mat2[k*r*2 + j]; // no shifting
}

void A0_B1(const int* const mat1, const int* const mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = segment_length+ tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[i*r*2 + j+c] += mat1[i*c*2 +k] * mat2[k*r*2 + j+c]; // shift col B
}

void A1_B2(const int* const mat1, const int* const mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[i*r*2 + j] += mat1[i*c*2 + k+r] * mat2[(k+r)*r*2 + j]; // shift col A, row B
}

void A1_B3(const int* const mat1, const int* const mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length)+ tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[i*r*2 + j+c] += mat1[i*c*2 + k+c] * mat2[(k+r)*r*2 + j+c]; // col C, col A, row+col B
}

void A2_B0(const int* const mat1, const int* const  mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[(i+r)*r*2 + j] += mat1[(i+r)*c*2 + k] * mat2[k*r*2 + j]; // row C, row A
}

void A2_B1(const int* const mat1, const int* const mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[(i+r)*r*2 + j+c] += mat1[(i+r)*c*2 + k] * mat2[k*r*2 + j+c]; // row C, row A, col B
}

void A3_B2(const int* const mat1, const int* const mat2, int* const mat3, int  r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[(i+r)*r*2 + j] += mat1[(i+r)*c*2 + k+c] * mat2[(k+r)*r*2 + j]; // row C, row A, col B;
}

void A3_B3(const int* const mat1, const int* const mat2, int* const mat3, int r, int c, int tid){
  int start_row = 0 + tid*(segment_length);
  int end_row = (segment_length) + tid*(segment_length);
  
  for(int i = start_row; i < end_row; i++)
    for(int j = 0; j < c; j++)
      for(int k = 0; k < r; k++)
        mat3[(i+r)*r*2 + j+c] += mat1[(i+r)*c*2 + k+c] * mat2[(k+r)*r*2 +j+c]; // shift all
}

void blocked0(const int* const mat1, const int* const mat2, int* const mat3, int r, int c){
  // Phase 1 Matrix Mul
  std::thread t0(A0_B0, mat1,mat2,mat3,r,c,0);
    std::thread t1(A0_B1, mat1,mat2,mat3,r,c,0);
    std::thread t2(A2_B0, mat1,mat2,mat3,r,c,0);
    std::thread t3(A2_B1, mat1,mat2,mat3,r,c,0);
  
   std::thread t4(A0_B0, mat1,mat2,mat3,r,c,1);
    std::thread t5(A0_B1, mat1,mat2,mat3,r,c,1);
   std::thread t6(A2_B0, mat1,mat2,mat3,r,c,1);
    std::thread t7(A2_B1, mat1,mat2,mat3,r,c,1);
  
    std::thread t8(A0_B0, mat1,mat2,mat3,r,c,2);
    std::thread t9(A0_B1, mat1,mat2,mat3,r,c,2);
    std::thread t10(A2_B0, mat1,mat2,mat3,r,c,2);
    std::thread t11(A2_B1, mat1,mat2,mat3,r,c,2);
  
    std::thread t12(A0_B0, mat1,mat2,mat3,r,c,3);
    std::thread t13(A0_B1, mat1,mat2,mat3,r,c,3);
   std::thread t14(A2_B0, mat1,mat2,mat3,r,c,3);
    std::thread t15(A2_B1, mat1,mat2,mat3,r,c,3);
  
    t0.join();t1.join();t2.join();t3.join();
    t4.join();t5.join();t6.join();t7.join();
  t8.join();t9.join();t10.join();t11.join();
    t12.join();t13.join();t14.join();t15.join();
}

void blocked1(const int* const mat1, const int* const mat2, int* const mat3, const int r, const int c){
  // Phase 2 Matmul
  std::thread t0(A1_B2, mat1,mat2,mat3,r,c,0);
  std::thread t1(A1_B3, mat1,mat2,mat3,r,c,0);
   std::thread t2(A3_B2, mat1,mat2,mat3,r,c,0);
  std::thread t3(A3_B3, mat1,mat2,mat3,r,c,0);
  
  std::thread t4(A1_B2, mat1,mat2,mat3,r,c,1);
    std::thread t5(A1_B3, mat1,mat2,mat3,r,c,1);
    std::thread t6(A3_B2, mat1,mat2,mat3,r,c,1);
    std::thread t7(A3_B3, mat1,mat2,mat3,r,c,1);
  
    std::thread t8(A1_B2, mat1,mat2,mat3,r,c,2);
    std::thread t9(A1_B3, mat1,mat2,mat3,r,c,2);
    std::thread t10(A3_B2, mat1,mat2,mat3,r,c,2);
    std::thread t11(A3_B3, mat1,mat2,mat3,r,c,2);
  
    std::thread t12(A1_B2, mat1,mat2,mat3,r,c,3);
    std::thread t13(A1_B3, mat1,mat2,mat3,r,c,3);
    std::thread t14(A3_B2, mat1,mat2,mat3,r,c,3);
    std::thread t15(A3_B3, mat1,mat2,mat3,r,c,3);
  
    t0.join();t1.join();t2.join();t3.join();
    t4.join();t5.join();t6.join();t7.join();
  t8.join();t9.join();t10.join();t11.join();
    t12.join();t13.join();t14.join();t15.join();

}

void blocked_matrix(const int* const mat1, const int* const mat2, int* const mat3, const int r, const int c){
  /* Solution
   * We can divide any even dimensioned matrix into 4 sub matrices C0 C1 C2 C3
   * If we wish to do C = A x B where A,B,C are even dimensioned matrices,
   * then we can perform matrix multiplication where we divide the matrices
   * into 4 sub blocks where the dimensions are half of the original matrix.
   * Supposing we have C with n*m dimensions then each submatrix will have
   * dimensions of n/2 * m/2.
   *
   * We partition a block as follow:
   *   ______________         _______________
   *  |            |          |      |      |
   *  |            |          |  A0  |  A1  |
   *  |     A      |  -=>     |______|______|
   *  |            |          |      |      |
   *  |            |          |  A2  |  A3  |
   *  |____________|          |______|______|
   *
   * To perform the blocked division we treat each sub matrix as if it was just
   * an ordinary element of the matrix and perform matrix mulitplication.
   *
   * We have to following procedure:
   * C0 = A0xB0 + A1xB2
   * C1 = A0xB1 + A1xB3
   * C2 = A2xB0 + A3xB3
   * C3 = A2xB1 + A3xB3
   *
   * To parallelize this, we can split the evaluation of each
   * sumatrix of C into two phases. For example, in C0 we let
   * A0xB0 as phase 1 and A1xB2 as phase 2. This will allow us
   * to avoid a race condtion which can optimize the procedure.
   *
   * */
  
  printf("CAlled\n");

  int r2 = r/2; // midpoint for rows
  int c2 = c/2; // midpoint for cols
  
  segment_length = r2/4; // 16 threads
  
  blocked0(mat1,mat2,mat3,r2,c2); // do first part
  blocked1(mat1,mat2,mat3,r2,c2); // do second part
  
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*     BLOCKED TRANSPOSE IMPLEMENTATION     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


// Multiplication of Submatrices AX and BX where BX is already transposed
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

/* PHASE 1 OF BLOCKED TRANSPOSE MULTIPICATION
 *  Spawing 16 threads manually because each block
 *  multiplication requires a different function.
 *
 *  ex) A0 x B0 and A0 x B1 cannot be generalized into
 *  a single function for a thread to know what to run
 *
 *  This may seem  inefficient but 16 threads would do the trick
 *  hence not that much of an overhead.
 * */
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
  
  // Thread joining (manua)
  t0.join();t1.join();t2.join();t3.join();
  t4.join();t5.join();t6.join();t7.join();
  t8.join();t9.join();t10.join();t11.join();
  t12.join();t13.join();t14.join();t15.join();
}

void proc1(const int* const mat1, int* mat2, int* const mat3, int r, int c){
/* PHASE 2 OF BLOCKED TRANSPOSE MULTIPLICATION
 *  Similar to PHASE 1, just doing the second part
 * */
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
  
  // Thread joining
  t0.join();t1.join();t2.join();t3.join();
  t4.join();t5.join();t6.join();t7.join();
  t8.join();t9.join();t10.join();t11.join();
  t12.join();t13.join();t14.join();t15.join();
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
  // BLOCKED TRANSPOSE Implementation
  row_per_thread = n / num_threads;
  int r2 = m/2;
  int c2 = n/2;
  segment_length = r2/4; // 16 threads
  int* transposed = generate_0(n,m);
  transpose(matrixB, transposed,n,m);

  proc0(matrixA,transposed, matrixC, r2,c2);
  proc1(matrixA,transposed, matrixC, r2,c2);


  /*  OTHER IMPLEMENTATIONS  */

  /*
  //Col Wise Implementation
  std::cout << "Col Wise Implementation" << std::endl;
  col_wise(matrixA, matrixB, matrixC, n,m);

   //Row Wise Implementation
  std::cout << "Row Wise Implementation" << std::endl;
  row_wise(matrixA, matrixB, matrixC, n,m);

   //Parallel Transpose Implementation
  std::cout << "Parallel Transpose Implementation" << std::endl;
  parallel_transpose(matrixA, matrixB, matrixC, n,m);

   //Blocked Implementation (Not Transposed)
  std::cout << "Blocked Implementation" << std::endl;
  blocked_matrix(matrixA, matrixB, matrixC, n,m);
   */
}
