#include "matrix.h"
#include <x86intrin.h>
#include <omp.h>

int allocate_matrix(matrix **mat, int rows, int cols) {
    *mat = malloc(sizeof(matrix));
    (*mat)->dim.rows = rows;
    (*mat)->dim.cols = cols;
    (*mat)->dim.rxc = rows * cols;
    (*mat)->data = calloc((*mat)->dim.rxc, sizeof(float)); 
    return 0;
}

int allocate_matrix_s(matrix **mat, shape s) {
    return allocate_matrix(mat, s.rows, s.cols);
}

int eye(matrix **mat, shape s) {
    assert(allocate_matrix_s(mat, s) == 0);
    int min, i; 
    min = (s.rows < s.cols) ? s.rows : s.cols;
    int chunk = s.cols + 1; 
    int cond = min*chunk;
    #pragma omp parallel for 
        for (i = 0; i <= cond; i+= chunk) { 
            (*mat)->data[i] = 1;
        }
    return 0;
}

void free_matrix(matrix *mat) {
    free(mat->data);
    free(mat);
}

void dot_product(matrix *vec1, matrix *vec2, float *result) {
    assert(same_size(vec1, vec2) && vec1->dim.cols == 1);
    *result = 0;
    __m128 matrix1, matrix2, first; 
    float store[4];
    float * vector1 = vec1->data;
    float * vector2 = vec2->data;
    float temp = 0;
    int cond2 = vec1->dim.rows;
    int cond1 = cond2/4 * 4; 
    int i; 
        #pragma omp for
            for (i = 0; i < cond1; i+=4) {
                matrix1 = _mm_loadu_ps(vector1 + i);
                matrix2 = _mm_loadu_ps(vector2 + i);
                first = _mm_dp_ps(matrix1, matrix2, 0xf1);
                _mm_store_ps(store, first);
                temp += store[0];
            }
            for (i = cond1; i < cond2; i++) {
                temp += *(vector1 + i) * *(vector2 + i);
            }
        #pragma omp critical 
            *result += temp;
}

void outer_product(matrix *vec1, matrix *vec2, matrix *dst) {
    assert(vec1->dim.cols == 1 && vec2->dim.cols == 1 && vec1->dim.rows == dst->dim.rows && vec2->dim.rows == dst->dim.cols);
    int cond1 = vec1->dim.rows;
    int cond2 = vec2->dim.rows;
    float * vector1 = vec1->data; 
    float * vector2 = vec2->data; 
    int i, j;
    for (i = 0; i < cond1; i++) {
        for (j = 0; j < cond2; j++) {
            dst->data[(i*cond2 + j)] = *(vector1 + i) * *(vector2 + j);
        }
    }
}

void matrix_power(matrix *mat, int pow, matrix *dst) { //binary lab implementation .... possible to do w/o recursion? 
    assert(mat != dst && same_size(mat, dst) && mat->dim.rows == mat->dim.cols);
    if (pow == 1) {
        copy(mat, dst);
        return;
    }
    if (pow == 2) {
        matrix_multiply(mat, mat, dst);
        return;
    }

    matrix* intermediate;
    eye(&intermediate, dst->dim);
    copy(intermediate, dst);
    if (pow == 0) {
        free(intermediate);
        return;
    }
    int i;
    for (i = 0; i < pow; i++) {
        matrix_multiply(intermediate, mat, dst); //not in place mult 
        copy(dst, intermediate);
    }
    free_matrix(intermediate);
}

void matrix_multiply(matrix *mat1, matrix *mat2, matrix *dst) {
    assert (mat1->dim.cols == mat2->dim.rows && dst->dim.rows == mat1->dim.rows && dst->dim.cols == mat2->dim.cols);
    int row1 = mat1->dim.rows;
    int col2 = mat2->dim.cols;
    int col1 = mat1->dim.cols;
    int row2 = mat2->dim.rows;
    matrix* temp;
    allocate_matrix(&temp, col2, row2);
    matrix_transpose(mat2,temp);
    unsigned int i, j, k;
    int cond = col1/16 * 16;
    #pragma omp parallel for collapse(2) private(k)
    for (i = 0; i < row1; i++) {
        for (j = 0; j < col2; j++) {
            __m128 matrix1, matrix2, first, second, third, fourth, total;
            float store[4] = {0, 0, 0, 0};
            float * mat1_data = mat1->data; 
            float * temp_data = temp->data; 
            float * temp1, * temp2;
            temp1 = mat1_data + i*col1;
            int dst_index = i*col2 + j;
            temp2 = temp_data + j*row2;
            dst->data[dst_index] = 0; //Ensures that the destination matrix is zeros initially. 
            float result = 0.0;
                for (k = 0; k < cond; k+=16) {
                
                    matrix1 = _mm_loadu_ps(temp1 + k);
                    matrix2 = _mm_loadu_ps(temp2 + k);
                    first = _mm_dp_ps(matrix1, matrix2, 0xf1); //dont use func or load backwards, use dot in this file  

                    matrix1 = _mm_loadu_ps(temp1 + k + 4);
                    matrix2 = _mm_loadu_ps(temp2 + k + 4);
                    second = _mm_dp_ps(matrix1, matrix2, 0xf2);

                    matrix1 = _mm_loadu_ps(temp1 + k + 8);
                    matrix2 = _mm_loadu_ps(temp2 + k + 8);
                    third = _mm_dp_ps(matrix1, matrix2, 0xf4);

                    matrix1 = _mm_loadu_ps(temp1 + k + 12);
                    matrix2 = _mm_loadu_ps(temp2 + k + 12);
                    fourth = _mm_dp_ps(matrix1, matrix2, 0xf8);

                    total =  _mm_add_ps(_mm_add_ps(_mm_add_ps(first, second), third), fourth);
                    _mm_storeu_ps(store, total);
                
                        result += store[0];
                        result += store[1];
                        result += store[2];
                        result += store[3];
                
                }
                // tail case 
                for (k = cond; k < col1; k++) {
                    result += *(temp1 + k) * *(temp2 + k);
                }
                    dst->data[dst_index] += result; 
            
        }
    }
    free_matrix(temp);
}

void matrix_scale(matrix *mat, float scalar, matrix *dst) {
    assert(same_size(mat, dst));
    float * temp = mat->data;
    int i;
    #pragma omp parallel for 
        for (i = 0; i < mat->dim.rxc; i++) {
            dst->data[i] = *(temp + i) * scalar;    
        }
}

void apply_func(matrix* mat, matrix* dst, float (*f)(float)) {
    assert(same_size(mat, dst));
    float * temp = mat->data;
    int i;
    for (i = 0; i < mat->dim.rxc; i++) {
        dst->data[i] = f(*(temp + i));
    }
}

void matrix_multiply_elementwise(matrix *mat1, matrix *mat2, matrix *dst) {
    assert(same_size(mat1, mat2) && same_size(mat1, dst));
    float * temp1 = mat1->data;
    float * temp2 = mat2->data;
    int i;
    #pragma omp parallel for
        for(i = 0; i < mat1->dim.rxc; i++)
            dst->data[i] = *(temp1 + i) * *(temp2 + i);
}

void matrix_add(matrix *mat1, matrix *mat2, matrix *dst) {
    assert(same_size(mat1, mat2) && same_size(mat1, dst));
    float * temp1 = mat1->data;
    float * temp2 = mat2->data;
    int i;
    for (i = 0; i < dst->dim.rxc; i++) {
        dst->data[i] = *(temp1 + i) + *(temp2 + i);
    }
}

void matrix_transpose(matrix *m, matrix *dst) {
    assert(m->dim.rows == dst->dim.cols && m->dim.cols == dst->dim.rows);
    int dst_col = dst->dim.cols;
    int dst_row = dst->dim.rows; 
    int m_col = m->dim.cols; 
    int dst_index;
    float * m_data = m->data; 
    float * m_index; 
    int i, j; 
    for (i = 0; i < dst_row; i++) {
        dst_index = i*dst_col;
        m_index = m_data + i; 
        for ( j = 0; j < dst_col; j++) {
            dst->data[dst_index + j] = *(m_index + (j * m_col));
        }  
    }
}

void copy(matrix *src, matrix *dst) {
    assert(same_size(src, dst));
    float * temp = src->data; 
    int i;
    for (i = 0; i < src->dim.rxc; i++) {
        dst->data[i] = *(temp + i);
    }
}


int get_rows(matrix *mat) {
    return mat->dim.rows;
}

int get_cols(matrix *mat) {
    return mat->dim.cols;
}

void get_matrix_as_array(float *arr, matrix *mat) {
    float * temp = mat->data; 
    int j;
    for (j = 0; j < mat->dim.rxc; j++) {
        arr[j] = *(temp + j);
    }
}

matrix* arr_to_matrix(float *arr, int rows, int cols) {
    matrix *m;
    int i,j, temp;
    allocate_matrix(&m, rows, cols);

    for (i = 0; i < rows; i++) {
        temp = i*cols;
        for (j = 0; j < cols; j++) {
            set_loc(m, i, j, arr[(temp + j)]);
        }
    }
    return m;
}

void set_loc(matrix *mat, int row, int col, float val) {
    assert (row < mat->dim.rows && col < mat->dim.cols && row >= 0 && col >= 0);
    mat->data[(row*(mat->dim.cols) + col)] = val;
}

int same_size(matrix *mat1, matrix *mat2) {
    return mat1 && mat2 && mat1->dim.rows == mat2->dim.rows && mat1->dim.cols == mat2->dim.cols;
}

float get_loc(matrix *mat, int row, int col) {
    return mat->data[(row*(mat->dim.cols) + col)];
}