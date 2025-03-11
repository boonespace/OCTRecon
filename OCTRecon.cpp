#include "OCTRecon.h"
#include "Encoding.h"
#include "toml++/toml.hpp"
#include <omp.h>

void Recon::imshow(const arma::mat &matrix, const std::string &winname, int waittime)
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
    cv::namedWindow(winname, cv::WINDOW_NORMAL);
    cv::imshow(winname, cvMatrix);
    cv::waitKey(waittime);
}

void Recon::imagesc(const arma::mat &matrix, const std::string &winname, int waittime)
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
    cv::namedWindow(winname, cv::WINDOW_NORMAL);
    cv::imshow(winname, coloredMatrix);
    cv::waitKey(waittime);
}

void Recon::saveArmaCubeToMultipageTIFF(const arma::cube &cube, const std::string &filename)
{
    int num_threads = omp_get_num_procs()*0.8; // Get the number of CPU cores
    omp_set_num_threads(num_threads);      // Set to the number of cores

    std::vector<cv::Mat> images_cv;

    // Overall normalization
    float min_value = cube.min();
    float max_value = cube.max();
    arma::cube normalized_cube = (cube - min_value) / (max_value - min_value) * 65535.0;

    // Convert arma::cube to std::vector<cv::Mat>
    for (int k = 0; k < cube.n_slices; k++)
    {
        arma::mat matrix_arma = normalized_cube.slice(k);
        cv::Mat matrix_cv(cube.n_rows, cube.n_cols, CV_16U);
        // Armadillo to OpenCV
// #pragma omp parallel for
        for (int i = 0; i < matrix_arma.n_rows; i++)
        {
            for (int j = 0; j < matrix_arma.n_cols; j++)
            {
                matrix_cv.at<uint16_t>(i, j) = matrix_arma(i, j);
            }
        }
        // matrix_cv.convertTo(matrix_cv, CV_16U);
        images_cv.push_back(matrix_cv);
    }

    // Save to multipage TIFF file
    std::string name = Utf8ToGbk(filename);
    cv::imwrite(name, images_cv);
}

Recon::Recon(size_t size_x, size_t size_y, size_t size_z, DataType dtype, int maxIteration, float rho, float lambda, int headerSize)
    : m_size_x(size_x), m_size_y(size_y), m_size_z(size_z), m_dtype(dtype), m_rho(rho), m_lambda(lambda), m_maxIteration(maxIteration), m_headerSize(headerSize)
{
    logger.log(Logger::LogLevel::INFO, "Recon", "Init", "Initializing reconstruction module");
    m_total_elements = size_x * size_y * size_z;
    m_dtypeSize = getDataTypeSize(dtype);
    m_expectedSize = m_total_elements * m_dtypeSize + headerSize;
    m_data_bin->resize(size_x, size_y, size_z);
    int N = size_x - 8;
    m_data_amplitude->resize(N / 2, size_y, size_z);
    m_data_phase->resize(N / 2, size_y, size_z);
    m_fft.set(N);

    m_window.resize(N);
    for (int n = 0; n < N; ++n)
    {
        m_window[n] = 0.5 * (1 - std::cos(2 * std::numbers::pi * n / (N - 1)));
    }
    logger.log(Logger::LogLevel::INFO, "Recon", "Init", "Successfully Initializion");
}

bool Recon::readData(std::string filename)
{
    imagename = filename;
    logger.log(Logger::LogLevel::INFO, "Recon", "Read", "Processing file: " + std::filesystem::path(imagename).filename().string());
    

    std::filesystem::path image_path(imagename);
    std::string image_path_noext = image_path.replace_filename(image_path.stem().string()).string();
    outputname_amplitude = image_path_noext + "_amplitude.tif";
    outputname_phase = image_path_noext + "_phase.tif";

    // read bin file
    std::ifstream file(imagename, std::ios::binary);

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();

    if (fileSize != m_expectedSize)
    {
        file.close();
        std::ostringstream errorMsg;
        errorMsg << "File size (" << fileSize << " bytes) does not match expected (" << m_expectedSize << " bytes)!" << std::endl;
        logger.log(Logger::LogLevel::ERROR, "Recon", "Read", errorMsg.str());
        return false;
    }

    file.seekg(m_headerSize, std::ios::beg);
    if (!file)
    {
        file.close();
        logger.log(Logger::LogLevel::ERROR, "Recon", "Read", "Failed to seek to the data position!");
        return false;
    }

    size_t count = 0;
    char buffer[8];
    // arma::Cube<int16_t> data(m_size_x, m_size_y, m_size_z);
    file.read(reinterpret_cast<char *>(m_data_bin->memptr()), m_expectedSize);
    // *m_data_bin = arma::conv_to<arma::cube>::from(data);
    file.close();
    logger.log(Logger::LogLevel::INFO, "Recon", "Read", "Successfully Read");
    return true;
}

bool Recon::reconstruction()
{
    logger.log(Logger::LogLevel::INFO, "Recon", "Process", "Reconstructing: " + std::filesystem::path(imagename).filename().string());
    int num_threads = omp_get_num_procs()*0.8; // Get the number of CPU cores
    omp_set_num_threads(num_threads);      // Set to the number of cores
    try
    {
        for (int k = 0; k < m_data_bin->n_slices; k++)
        {
#pragma omp parallel for
            for (int j = 0; j < m_data_bin->n_cols; j++)
            {
                // Extract A-Scan data and remove the first 8 bits of identifier
                arma::vec ascan = arma::conv_to<arma::vec>::from(m_data_bin->slice(k).col(j));
                ascan.shed_rows(0, 7);
                // Add window to reduce spectrum leakage
                ascan %= m_window;
                // Fourier Transform
                arma::cx_vec ascan_fft = m_fft.computeFFT(ascan);
                // Extract amplitude and phase
                arma::vec ascan_abs = arma::abs(ascan_fft);
                arma::vec ascan_arg = arma::arg(ascan_fft);
                // ADMM Iteration
                if (m_maxIteration>0) {
                    arma::cx_vec r = ascan_fft; // init value
                    arma::cx_vec u(r.n_elem, arma::fill::zeros);
                    float rho = m_rho, lambda = m_lambda;
                    for (int iter = 0; iter < m_maxIteration; iter++) {
                        arma::cx_vec r_new = ascan_fft + rho * (r - u);
                        float lambda_new = lambda / rho;
                        arma::vec zeros(r.n_elem, arma::fill::zeros);
                        arma::vec s = arma::abs(r_new) - lambda_new;
                        r = arma::sign(r_new) % arma::max(arma::abs(r_new) - lambda_new, zeros);
                        u = u + r;
                        float error = arma::norm(r);
                        if (error < 1e-4)
                            break;
                    }
                    ascan_abs = arma::abs(r);
                }
                // Retain half and remove the inverted image
                ascan_abs.shed_rows(ascan_abs.n_rows - 1024, ascan_abs.n_rows - 1);
                ascan_arg.shed_rows(ascan_arg.n_rows - 1024, ascan_arg.n_rows - 1);
                // Signal Enhancement
                ascan_abs = 70.0 * arma::log10(ascan_abs + 1);
                ascan_abs = arma::pow(ascan_abs, 3);
                // Save
                m_data_amplitude->slice(k).col(j) = ascan_abs;
                m_data_phase->slice(k).col(j) = ascan_arg;
            }
            arma::mat matrix = m_data_amplitude->slice(k);
            imagesc(matrix, "magnitude", 1);
            matrix = m_data_phase->slice(k);
            imagesc(matrix, "phase", 1);
        }
        logger.log(Logger::LogLevel::INFO, "Recon", "Process", "Successfully Reconstruct");
        logger.log(Logger::LogLevel::INFO, "Recon", "Process", "Writing to Tiff File");
        saveArmaCubeToMultipageTIFF(*m_data_amplitude, outputname_amplitude);
        saveArmaCubeToMultipageTIFF(*m_data_phase, outputname_phase);
        logger.log(Logger::LogLevel::INFO, "Recon", "Process", "Successfully write");
    }
    catch (const std::exception& e)
    {
        logger.log(Logger::LogLevel::ERROR, "Reconstruction", "Process", "Reconstruction failed for: " + std::filesystem::path(imagename).filename().string());
        return false;
    }
    return true;
}

int main()
{
    std::setlocale(LC_ALL, ".UTF-8");

    // Logger
    Logger logger;

    // Configure
    std::string filename = "config.toml";
    auto config = toml::parse_file(filename);
    std::string path = *config["path"].value<std::string>();
    std::string extension = *config["extension"].value<std::string>();
    int size_x = *config["size_x"].value<int>();
    int size_y = *config["size_y"].value<int>();
    int size_z = *config["size_z"].value<int>();
    int maxIteration = *config["maxIteration"].value<bool>();
    float rho = *config["rho"].value<float>();
    float lambda = *config["lambda"].value<float>();

    // Adjust according to the actual project directory structure
    logger.log(Logger::LogLevel::INFO, "Data", "Init", "Directory scan started: " + path);
    DataStorage ds;
    ds.getFromFolder(path, extension);
    logger.log(Logger::LogLevel::INFO, "Data", "Load", "Directory scan complete");

    // Perform image reconstruction
    Recon recon(size_x, size_y, size_z, DataType::INT16, maxIteration, rho, lambda);

    for (int i = 0; i < ds.length; i++)
    {
        std::string filepath = ds.readname(i);
        std::string filename = std::filesystem::path(filepath).filename().string();
        

        if (!recon.readData(filepath))
        {
            continue;
        }

        if (!recon.reconstruction())
        {
            continue;
        }

        logger.log(Logger::LogLevel::INFO, "Reconstruction", "Success", "Successfully reconstructed: " + filename);
    }

    logger.log(Logger::LogLevel::INFO, "System", "Shutdown", "Program completed successfully");
    return 0;
}