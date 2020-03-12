#include "matrix.h"
#include <cstring>
#include <cassert>
#include <iostream>
#include <cstdlib>
#include <ctime>

#define MAX(a, b) (a > b ? a : b)
#define INDEX(a, b) (a * m + b)

Matrix::Matrix() {}

Matrix::Matrix(const Matrix &other) {
    n = other.n;
    m = other.m;
    data = new double[n * m];
    std::memcpy(data, other.data, sizeof(double) * n * m);
}

Matrix::Matrix(int n, int m) : n(n), m(m) {
    data = new double[n * m];
}

Matrix::Matrix(int n, int m, double *data) : n(n), m(m) {
    this->data = new double[n * m];
    std::memcpy(this->data, data, sizeof(double) * n * m);
}

Matrix::~Matrix() {
    delete[] data;
}

void Matrix::init(int n, int m) {
    this->n = n;
    this->m = m;
}

void Matrix::init(int n, int m, double *data) {
    this->n = n;
    this->m = m;
    this->data = new double[n * m];
    std::memcpy(this->data, data, sizeof(double) * n * m);
}

Matrix Matrix::operator*(Matrix &other) {
    Matrix retmat(n, other.m);
    assert(m == other.n);
    double dot;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < other.m; j++) {
            dot = 0;
            for (int k = 0; k < m; k++) {
                dot += data[INDEX(i, k)] * other.data[INDEX(k, j)];
            }
            retmat[i][j] = dot;
        }
    }
    return retmat;
}

double* Matrix::operator*(double *vec) {
    double *retvec = new double[n];
    double dot;
    for (int i = 0; i < n; i++) {
        dot = 0;
        for (int j = 0; j < m; j++) {
            dot += vec[j] * data[INDEX(i,j)];
        }
        retvec[i] = dot;
    }
    return retvec;
}

double* Matrix::operator[](int index) {
    return data + (index * m);
}

void Matrix::pretty_print() {
    for (int i = 0; i < n * m; i++) {
        std::cout << data[i] << ' ';
        if (!((i + 1) % m)) std::cout << std::endl;
    }
}

Matrix random_matrix(int n, int m) {
    std::srand(std::time(NULL));
    double *data = new double[n * m];
    for (int i = 0; i < n * m; i++) {
        data[i] = (std::rand() / (double)RAND_MAX) * 2 - 1;
    }
    Matrix random_matrix(n, m, data);
    return random_matrix;
}
