#ifndef RNN_H
#define RNN_H

#include "matrix.h"

class RNN {
    public:
        RNN(double (*f_activation_h)(double), double (*f_activation_y)(double),
                int output_size, int input_size, int h_size);
        ~RNN();
        double* prop(double *x);
        double back_prop();

    private:
        Matrix XtH;
        Matrix htH;
        Matrix HtY;
        double *h;
        double (*f_activation_h)(double);
        double (*f_activation_y)(double);
        double (*d_f_activation_h)(double);
        double (*d_f_activation_y)(double);
        double *bh;
        double *by;
        double training_rate;
        int output_size;
        int input_size;
        int h_size;
        int batch_size;
        int batch;
};

#endif
