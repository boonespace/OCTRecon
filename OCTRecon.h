#ifndef OCTRECON_H
#define OCTRECON_H

#include <iostream>
#include <vector>
#include <armadillo>
#include "DataStorage.h"
#include "FFT.h"
#include <opencv2/opencv.hpp>
#include <numbers>

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

void imshow(const arma::mat &matrix, const std::string &winname = "Image", int waittime = 0)
{
    cv::Mat cvMatrix(matrix.n_rows, matrix.n_cols, CV_64F);

    // Armadillo to OpenCV
    for (size_t i = 0; i < matrix.n_rows; ++i)
    {
        for (size_t j = 0; j < matrix.n_cols; ++j)
        {
            cvMatrix.at<double>(i, j) = matrix(i, j);
        }
    }

    cv::normalize(cvMatrix, cvMatrix, 0, 255, cv::NORM_MINMAX);
    cvMatrix.convertTo(cvMatrix, CV_8U);

    // Display Image
    cv::namedWindow("Armadillo Matrix", cv::WINDOW_NORMAL);
    cv::imshow("Armadillo Matrix", cvMatrix);
    cv::waitKey(waittime);
}

void imagesc(const arma::mat &matrix, const std::string &winname = "Colormap Image", int colormap = cv::COLORMAP_JET, int waittime = 0)
{
    cv::Mat cvMatrix(matrix.n_rows, matrix.n_cols, CV_64F);

    // Armadillo to OpenCV
    for (size_t i = 0; i < matrix.n_rows; ++i)
    {
        for (size_t j = 0; j < matrix.n_cols; ++j)
        {
            cvMatrix.at<double>(i, j) = matrix(i, j);
        }
    }

    cv::normalize(cvMatrix, cvMatrix, 0, 255, cv::NORM_MINMAX);
    cvMatrix.convertTo(cvMatrix, CV_8U);

    // Apply colormap
    cv::Mat coloredMatrix;
    cv::applyColorMap(cvMatrix, coloredMatrix, colormap);

    // Display Image
    cv::namedWindow("Armadillo Matrix", cv::WINDOW_NORMAL);
    cv::imshow("Armadillo Matrix", coloredMatrix);
    cv::waitKey(waittime);
}

class Recon
{
private:
    std::shared_ptr<arma::cube> m_data_bin = std::make_shared<arma::cube>();
    std::shared_ptr<arma::cube> m_data_gray = std::make_shared<arma::cube>();
    size_t m_size_x;
    size_t m_size_y;
    size_t m_size_z;
    size_t m_total_elements;
    size_t m_expectedSize;
    DataType m_dtype;
    size_t m_dtypeSize;
    int m_headerSize;
    FFT m_fft;
    arma::vec m_window;

public:
    Recon(size_t size_x, size_t size_y, size_t size_z, DataType dtype, int headerSize = 0)
        : m_size_x(size_x), m_size_y(size_y), m_size_z(size_z), m_dtype(dtype), m_headerSize(headerSize)
    {
        m_total_elements = size_x * size_y * size_z;
        m_dtypeSize = getDataTypeSize(dtype);
        m_expectedSize = m_total_elements * m_dtypeSize + headerSize;
        m_data_bin->resize(size_x, size_y, size_z);
        int N = size_x - 8;
        m_data_gray->resize(N / 2, size_y, size_z);
        m_fft.set(N);

        m_window.resize(N);
        for (int n = 0; n < N; ++n)
        {
            m_window[n] = 0.5 * (1 - std::cos(2 * std::numbers::pi * n / (N - 1)));
        }
    }

    void readData(std::string filename)
    {
        std::filesystem::path filepath = std::filesystem::path(filename);
        // 读取数据
        std::ifstream file(filepath, std::ios::binary);

        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();

        if (fileSize != m_expectedSize)
        {
            file.close();
            std::cerr << "File size (" << fileSize << " bytes) does not match expected (" << m_expectedSize << " bytes)!" << std::endl;
        }

        file.seekg(m_headerSize, std::ios::beg);
        if (!file)
        {
            std::cerr << "Error: Failed to seek to the data position!" << std::endl;
        }

        size_t count = 0;
        char buffer[8];
        double *dataPtr = m_data_bin->memptr();

        while (file.read(buffer, m_dtypeSize))
        {
            int16_t value; // TODO: Dynamic settings by dtype
            std::memcpy(&value, buffer, m_dtypeSize);
            dataPtr[count] = value;
            count++;
        }

        // Check if the data was read completely

        if (count < m_total_elements)
        {
            std::cerr << "Warning: Insufficient data read, the file may be corrupted or incomplete!" << std::endl;
        }
        else
        {
            std::cout << "Data reading completed successfully. Total " << count << " data points read." << std::endl;
        }

        file.close();
    }

    void reconstruction()
    {
        for (int k = 0; k < m_data_bin->n_slices; k++)
        {
            for (int j = 0; j < m_data_bin->n_cols; j++)
            {
                arma::vec ascan = m_data_bin->slice(k).col(j);
                ascan.shed_rows(0, 7);
                ascan %= m_window;
                arma::cx_vec ascan_fft = m_fft.computeFFT(ascan);
                arma::vec ascan_abs = arma::abs(ascan_fft);
                ascan_abs = 70.0 * arma::log10(ascan_abs + 1);
                ascan_abs = arma::pow(ascan_abs, 3);
                ascan_abs.shed_rows(ascan_abs.n_rows - 1024, ascan_abs.n_rows - 1);
                m_data_gray->slice(k).col(j) = ascan_abs;
            }
            arma::mat matrix = m_data_gray->slice(k);
            imagesc(matrix, "Armadillo Matrix", 2, 1);
        }
    }
};

#endif // OCTRECON_H