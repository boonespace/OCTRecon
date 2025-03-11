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

enum DataType
{
    INT8,
    UINT8,
    INT16,
    UINT16,
    FLOAT,
    FLOAT32
};

std::unordered_map<DataType, size_t> dataTypeSizes = {
    {INT8, sizeof(int8_t)},
    {UINT8, sizeof(uint8_t)},
    {INT16, sizeof(int16_t)},
    {UINT16, sizeof(uint16_t)},
    {FLOAT, sizeof(float)},
    {FLOAT32, sizeof(float)}};

size_t getDataTypeSize(DataType type)
{
    return dataTypeSizes[type];
}

class Recon
{
private:
    std::shared_ptr<arma::Cube<int16_t>> m_data_bin = std::make_shared<arma::Cube<int16_t>>();
    std::shared_ptr<arma::cube> m_data_amplitude = std::make_shared<arma::cube>();
    std::shared_ptr<arma::cube> m_data_phase = std::make_shared<arma::cube>();
    std::string imagename;            // Path to the input image file.
    std::string outputname_amplitude; // Path to the output image file.
    std::string outputname_phase;     // Path to the output image file.
    size_t m_size_x;
    size_t m_size_y;
    size_t m_size_z;
    size_t m_total_elements;
    size_t m_expectedSize;
    DataType m_dtype;
    int m_maxIteration;
    float m_rho;
    float m_lambda;
    size_t m_dtypeSize;
    int m_headerSize;
    FFT m_fft;
    arma::vec m_window;

    void saveArmaCubeToMultipageTIFF(const arma::cube &cube, const std::string &filename);
    void imshow(const arma::mat &matrix, const std::string &winname = "Image", int waittime = 0);
    void imagesc(const arma::mat &matrix, const std::string &winname = "Colormap Image", int waittime = 0);

public:
    Logger logger;
    Recon(size_t size_x, size_t size_y, size_t size_z, DataType dtype, int maxIteration = 0, float rho = 0.01, float lambda = 100, int headerSize = 0);
    bool readData(std::string filename);
    bool reconstruction();
};

#endif // OCTRECON_H