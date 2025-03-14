#ifndef OCTRECON_H
#define OCTRECON_H

#include <iostream>
#include <vector>
#include <armadillo>
#include "DataStorage.h"
#include "FFT.h"
#include <opencv2/opencv.hpp>
#include <numbers>
#include "Logger.h"

class Recon
{
private:
    arma::Cube<int16_t> m_data_bin;
    arma::cube m_data_amplitude;
    arma::cube m_data_phase;
    std::string imagename;            // Path to the input image file.
    std::string outputname_amplitude; // Path to the output image file.
    std::string outputname_phase;     // Path to the output image file.
    size_t m_num_header = 8; // fixed value
    size_t m_num_ascan = 2048; // fixed value
    size_t m_num_bscan = 0;
    size_t m_num_cscan = 0;
    int m_maxIteration;
    float m_rho;
    float m_lambda;
    FFT m_fft;
    arma::vec m_window;
    bool m_visualize_frames = false;

    void saveArmaCubeToMultipageTIFF(const arma::cube &cube, const std::string &filename);
    void imshow(const arma::mat &matrix, const std::string &winname = "Image", int waittime = 0);
    void imagesc(const arma::mat &matrix, const std::string &winname = "Colormap Image", int waittime = 0);

public:
    Logger logger;
    Recon(int maxIteration = 0, float rho = 0.01, float lambda = 100, bool visualize_frames = false);
    bool readData(std::string filename);
    bool reconstruction();
};

#endif // OCTRECON_H