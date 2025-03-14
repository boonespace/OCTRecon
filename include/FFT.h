#ifndef FFT_H
#define FFT_H

#include <armadillo>
#include "fftw3/include/fftw3.h"

class FFT
{
private:
    int m_size_x;
    fftw_plan m_plan_forward;
    fftw_plan m_plan_backward;

public:
    explicit FFT() : m_size_x(0), m_plan_forward(nullptr), m_plan_backward(nullptr) {};
    ~FFT();

    void set(int size_x);

    // Perform FFT, ensuring thread safety
    arma::cx_vec computeFFT(const arma::vec &inputMatrix); // Fourier Transform
    arma::vec computeIFFT(const arma::cx_vec &inputMatrix); // Inverse Fourier transform

private:
    // Data conversion between armadillo and fftw
    void arma2fftw(const arma::vec &input, fftw_complex *in);
    void arma2fftw(const arma::cx_vec &input, fftw_complex *in);
    void fftw2arma(arma::cx_vec &output, fftw_complex *out);
    void fftw2arma(arma::vec &output, fftw_complex *out);
};

#endif // FFT_H