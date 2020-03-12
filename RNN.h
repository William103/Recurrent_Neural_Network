#ifndef RNN_H
#define RNN_H

#include "matrix.h"

class RNN {
    public:
        RNN(double (*f_activation_h)(double), double (*f_activation_y)(double),
                int output_size, int input_size, int h_size);
        ~RNN();
        double* prop(double *x);

    private:
        Matrix XtH;
        Matrix HtH;
        double *h;
        double (*f_activation_h)(double);
        double (*f_activation_y)(double);
        double *bh;
        double *by;
        int output_size;
        int input_size;
        int h_size;
};

#endif
