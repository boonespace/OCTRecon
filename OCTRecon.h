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

    cv::normalize(cvMatrix, cvMatrix, 0, 65535, cv::NORM_MINMAX);
    cvMatrix.convertTo(cvMatrix, CV_16U);

    // Display Image
    cv::namedWindow("Armadillo Matrix", cv::WINDOW_NORMAL);
    cv::imshow("Armadillo Matrix", cvMatrix);
    cv::waitKey(waittime);
}

void imagesc(const arma::mat &matrix, const std::string &winname = "Colormap Image", int waittime = 0)
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
    int colormap = cv::COLORMAP_JET;
    cv::applyColorMap(cvMatrix, coloredMatrix, colormap);

    // Display Image
    cv::namedWindow("Armadillo Matrix", cv::WINDOW_NORMAL);
    cv::imshow("Armadillo Matrix", coloredMatrix);
    cv::waitKey(waittime);
}

void saveArmaCubeToMultipageTIFF(const arma::cube &cube, const std::string &filename)
{
    std::vector<cv::Mat> images_cv;
    
    // Overall normalization
    float min_value = cube.min();
    float max_value = cube.max();
    arma::cube normalized_cube = (cube - min_value) / (max_value - min_value) * 65535.0;

    // Convert arma::cube to std::vector<cv::Mat>
    for (size_t k = 0; k < cube.n_slices; k++)
    {
        arma::mat matrix_arma = normalized_cube.slice(k);
        cv::Mat matrix_cv(cube.n_rows, cube.n_cols, CV_16U);
        // Armadillo to OpenCV
        for (size_t i = 0; i < matrix_arma.n_rows; i++)
        {
            for (size_t j = 0; j < matrix_arma.n_cols; j++)
            {
                matrix_cv.at<uint16_t>(i, j) = matrix_arma(i, j);
            }
        }
        // matrix_cv.convertTo(matrix_cv, CV_16U);
        images_cv.push_back(matrix_cv);
    }

    // Save to multipage TIFF file
    cv::imwrite(filename, images_cv);
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
        arma::Cube<int16_t> data(m_size_x, m_size_y, m_size_z);
        file.read(reinterpret_cast<char*>(data.memptr()), m_expectedSize);
        *m_data_bin = arma::conv_to<arma::cube>::from(data);
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
                ascan_fft.shed_rows(ascan_fft.n_rows - 1024, ascan_fft.n_rows - 1);
                arma::vec ascan_abs = arma::abs(ascan_fft);
                ascan_abs = 70.0 * arma::log10(ascan_abs + 1);
                ascan_abs = arma::pow(ascan_abs, 3);
                m_data_gray->slice(k).col(j) = ascan_abs;
            }
            arma::mat matrix = m_data_gray->slice(k);
            imagesc(matrix, "Armadillo Matrix", 1);
        }
        saveArmaCubeToMultipageTIFF(*m_data_gray, "a.tif");
    }
};

#endif // OCTRECON_H