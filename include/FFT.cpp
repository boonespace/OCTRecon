#include "FFT.h"

void FFT::set(int size_x)
{
    m_size_x = size_x;
    // Create an independent FFT plan for this instance
    // armadillo is column-first, fftw is row-first, so the x and y dimensions need to be swapped
    // in and out are both nullptr, support dft method
    m_plan_forward = fftw_plan_dft_1d(size_x, nullptr, nullptr, FFTW_FORWARD, FFTW_ESTIMATE);
    m_plan_backward = fftw_plan_dft_1d(size_x, nullptr, nullptr, FFTW_BACKWARD, FFTW_ESTIMATE);
}

FFT::~FFT()
{
    fftw_destroy_plan(m_plan_forward);
    fftw_destroy_plan(m_plan_backward);
}

void FFT::arma2fftw(const arma::vec &input, fftw_complex *in)
{
    if (m_size_x != input.n_rows)
        throw std::runtime_error("Error: Input matrix size does not match FFT plan size!");

    // Fill in input data
    for (int i = 0; i < m_size_x; i++)
    {
        int fftw_index = i;
        in[fftw_index][0] = input(i); // real
        in[fftw_index][1] = 0.0f;     // imag
    }
}

void FFT::arma2fftw(const arma::cx_vec &input, fftw_complex *in)
{
    if (m_size_x != input.n_rows)
        throw std::runtime_error("Error: Input matrix size does not match FFT plan size!");

    // Fill in input data
    for (int i = 0; i < m_size_x; i++)
    {
        int fftw_index = i;
        in[fftw_index][0] = input(i).real(); // real
        in[fftw_index][1] = input(i).imag(); // imag
    }
}

void FFT::fftw2arma(arma::cx_vec &output, fftw_complex *out)
{
    // Store the FFT result in the Armadillo complex matrix
    for (int i = 0; i < m_size_x; ++i)
    {
        int fftw_index = i;
        output(i) = std::complex<float>(out[fftw_index][0], out[fftw_index][1]);
    }
}

void FFT::fftw2arma(arma::vec &output, fftw_complex *out)
{
    // Store the FFT result in the Armadillo matrix
    float norm_factor = m_size_x;
    for (int i = 0; i < m_size_x; ++i)
    {
        int fftw_index = i;
        output(i) = out[fftw_index][0] / norm_factor;
    }
}

// Perform FFT, ensuring thread safety
arma::cx_vec FFT::computeFFT(const arma::vec &inputMatrix)
{
    arma::cx_vec output(m_size_x);
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * m_size_x);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * m_size_x);

    arma2fftw(inputMatrix, in);
    fftw_execute_dft(m_plan_forward, in, out);
    fftw2arma(output, out);

    fftw_free(in);
    fftw_free(out);
    return output;
}

// Perform IFFT, ensuring thread safety
arma::vec FFT::computeIFFT(const arma::cx_vec &inputMatrix)
{
    arma::vec output(m_size_x);
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * m_size_x);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * m_size_x);

    arma2fftw(inputMatrix, in);
    fftw_execute_dft(m_plan_backward, in, out);
    fftw2arma(output, out);

    fftw_free(in);
    fftw_free(out);
    return output;
}

