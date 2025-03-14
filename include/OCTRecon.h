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
#include "Config.h"

enum class OutputMode {
    recon_binary,
    recon_tiff16,
    recon_tiff08,
};

OutputMode string_to_output_mode(const std::string& str);
std::string output_mode_to_string(OutputMode mode);


class Recon
{
private:
    arma::Cube<int16_t> m_data_bin;
    arma::cx_cube m_data_fft;
    arma::cube m_data_amplitude;
    arma::cube m_data_phase;
    std::string imagename;           
    std::string outputname_amplitude;
    std::string outputname_phase;    
    std::string outputname_fft;
    size_t m_num_header = 8;
    size_t m_num_ascan = 2048;
    size_t m_num_bscan = 0;
    size_t m_num_cscan = 0;
    int m_maxIteration;
    float m_rho;
    float m_lambda;
    FFT m_fft;
    arma::vec m_window;
    bool m_visualize_frames = false;
    OutputMode m_output_mode = OutputMode::recon_binary;
    std::vector<std::string> m_output_data;

    void save(const arma::cube& cube, const std::string& filename);
    void save(const arma::cx_cube& cube, const std::string& filename);
    void saveArmaCubeToMultipageTIFF(const arma::cube &cube, const std::string &filename, int bit_num);
    void imshow(const arma::mat &matrix, const std::string &winname = "Image", int waittime = 0);
    void imagesc(const arma::mat &matrix, const std::string &winname = "Colormap Image", int waittime = 0);

public:
    Logger logger;
    Recon(Config config);
    bool readData(std::string filename);
    bool reconstruction();
};

#endif // OCTRECON_H