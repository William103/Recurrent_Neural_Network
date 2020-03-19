#include "RNN.h"
#include <cstring>
#include <cstdlib>
#include <ctime>

RNN::RNN(double (*f_activation_h)(double), double (*f_activation_y)(double),
    int output_size, int input_size, int h_size) :
        f_activation_h(f_activation_h), f_activation_y(f_activation_y),
        output_size(output_size), h_size(h_size), input_size(input_size)
{
    std::srand(std::time(NULL));
    h = new double[h_size];
    std::memset(h, 0, sizeof(double) * h_size);
    XtH = random_matrix(h_size, input_size);
    htH = random_matrix(h_size, h_size);
    HtY = random_matrix(h_size, output_size);
    bh = new double[h_size];
    by = new double[output_size];
    for (int i = 0; i < h_size; i++) {
        bh[i] = (std::rand() / (double) RAND_MAX) * 2 - 1;
    }
    for (int i = 0; i < output_size; i++) {
        by[i] = (std::rand() / (double) RAND_MAX) * 2 - 1;
    }
    batch = 0;
}

RNN::~RNN() {
    delete[] h;
    delete[] bh;
    delete[] by;
}

double* RNN::prop(double *x) {
    double *wx = XtH * x;
    double *wh = htH * h;
    double *y = new double[output_size];
    double *hy;
    for (int i = 0; i < h_size; i++) {
        h[i] = f_activation_h(wx[i] + wh[i] + bh[i]);
    }
    hy = HtY * h;
    for (int i = 0; i < output_size; i++) {
        y[i] = f_activation_y(hy[i] + by[i]);
    }
    delete[] wx;
    delete[] wh;
    return y;
}

double RNN::back_prop() {
    dE/dw = dC * dsy * h * dsh * (x + h_{t-1} * dh_{t-1}/dw)
}
