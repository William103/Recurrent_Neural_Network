#ifndef MATRIX_H
#define MATRIX_H

class Matrix {
    public:
        Matrix();
        Matrix(const Matrix &other);
        Matrix(int n, int m);
        Matrix(int n, int m, double *data);

        void init(int n, int m);
        void init(int n, int m, double *data);

        ~Matrix();

        Matrix  operator*(Matrix &other);

        // NOTE: assumes length of vector = m UB otherwise
        // Also, user is expected to free vec
        double* operator*(double *vec);

        double* operator[](int index);

        void pretty_print();

    private:
        int n, m;
        double *data;
};

Matrix random_matrix(int n, int m);

#endif
