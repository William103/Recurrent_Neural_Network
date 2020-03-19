#include "RNN.h"
#include "utils.h"
#include <iostream>

int main() {
    RNN net(sigmoid, sigmoid, 26, 1, 100);
    double input[] = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    std::cout << *net.prop(input) << std::endl;
    std::cout << *net.prop(input) << std::endl;
    std::cout << *net.prop(input) << std::endl;
    std::cout << *net.prop(input) << std::endl;
    std::cout << *net.prop(input) << std::endl;
    std::cout << *net.prop(input) << std::endl;
    std::cout << *net.prop(input) << std::endl;
    return 0;
}
