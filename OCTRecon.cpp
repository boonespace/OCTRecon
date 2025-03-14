#include "OCTRecon.h"
#include "Encoding.h"
#include "toml++/toml.hpp"
#include <omp.h>

void Recon::imshow(const arma::mat &matrix, const std::string &winname, int waittime)
{
    cv::Mat cvMatrix(matrix.n_rows, matrix.n_cols, CV_64F);
    arma::mat matrix_t = matrix.t();
    std::memcpy(cvMatrix.data, matrix_t.memptr(), matrix_t.n_elem * sizeof(double));

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
    arma::mat matrix_t = matrix.t();
    std::memcpy(cvMatrix.data, matrix_t.memptr(), matrix_t.n_elem * sizeof(double));

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
    std::vector<cv::Mat> images_cv;

    // Overall normalization
    float min_value = cube.min();
    float max_value = cube.max();
    arma::cube normalized_cube = (cube - min_value) / (max_value - min_value) * 65535.0;

    // Convert arma::cube to std::vector<cv::Mat>
    for (int k = 0; k < cube.n_slices; k++)
    {
        arma::mat matrix_arma = normalized_cube.slice(k);
        cv::Mat matrix_cv(cube.n_rows, cube.n_cols, CV_64F);
        arma::mat matrix_t = matrix_arma.t();
        std::memcpy(matrix_cv.data, matrix_t.memptr(), matrix_t.n_elem * sizeof(double));
        matrix_cv.convertTo(matrix_cv, CV_16U);
        images_cv.push_back(matrix_cv);
    }

    // Save to multipage TIFF file
    std::string name = Utf8ToGbk(filename);
    cv::imwrite(name, images_cv);
}

Recon::Recon(int maxIteration, float rho, float lambda, bool visualize_frames)
    : m_rho(rho), m_lambda(lambda), m_maxIteration(maxIteration), m_visualize_frames(visualize_frames)
{
}

bool Recon::readData(std::string filename)
{
    imagename = filename;
    logger.log(Logger::LogLevel::INFO, "Recon", "Reading", "Processing file: " + std::filesystem::path(imagename).filename().string());

    std::filesystem::path image_path(imagename);
    std::string image_path_noext = image_path.replace_filename(image_path.stem().string()).string();
    outputname_amplitude = image_path_noext + "_amplitude.tif";
    outputname_phase = image_path_noext + "_phase.tif";

    // read bin file
    FILE* file = fopen(filename.c_str(), "rb"); // Open the file in binary mode
    if (!file) {
        logger.log(Logger::LogLevel::ERROR, "Recon", "Reading", "Unable to open file");
        return 1;
    }

    // Obtain file size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    // Obtain the number of int16_t
    size_t numElements = fileSize / sizeof(int16_t);

    // Allocate memory
    int16_t* data = (int16_t*)malloc(fileSize);
    if (!data) {
        logger.log(Logger::LogLevel::ERROR, "Recon", "Reading", "Memory allocation failed");
        fclose(file);
        return 1;
    }

    // Read data
    size_t elementsRead = fread(data, sizeof(int16_t), numElements, file);
    fclose(file);

    // Check if reading is successful
    if (elementsRead != numElements) {
        logger.log(Logger::LogLevel::ERROR, "Recon", "Reading", "Error reading data");
        free(data);
        return 1;
    }

    // Judge the data shape
    m_num_bscan = (size_t) data[6];
    m_num_cscan = elementsRead / m_num_bscan / (m_num_header + m_num_ascan);

    // Copy data to m_data_bin so that it can automatically manage memory
    m_data_bin.resize(m_num_header + m_num_ascan, m_num_bscan, m_num_cscan);
    std::memcpy(m_data_bin.memptr(), data, fileSize);
    free(data);

    // Allocate size
    size_t m_num_half_ascan = m_num_ascan / 2;
    m_data_amplitude.resize(m_num_half_ascan, m_num_bscan, m_num_cscan);
    m_data_phase.resize(m_num_half_ascan, m_num_bscan, m_num_cscan);
    m_fft.set(m_num_ascan);

    // Pre-generate window function (Hann window)
    m_window.resize(m_num_ascan);
    for (int n = 0; n < m_num_ascan; ++n)
    {
        m_window[n] = 0.5 * (1 - std::cos(2 * std::numbers::pi * n / (m_num_ascan - 1)));
    }

    logger.log(Logger::LogLevel::TRACE, "Recon", "Reading", "Successfully Read");
    return true;
}

bool Recon::reconstruction()
{
    logger.log(Logger::LogLevel::INFO, "Recon", "Process", "Reconstructing\r");
    float next_progress = 0.0f;
    int num_threads = omp_get_num_procs()*0.8; // Get the number of CPU cores
    omp_set_num_threads(num_threads);      // Set to the number of cores
    try
    {
        for (int k = 0; k < m_data_bin.n_slices; k++)
        {
#pragma omp parallel for
            for (int j = 0; j < m_data_bin.n_cols; j++)
            {
                // Extract A-Scan data and remove the first 8 bits of identifier
                arma::vec ascan = arma::conv_to<arma::vec>::from(m_data_bin.slice(k).col(j));
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
                m_data_amplitude.slice(k).col(j) = ascan_abs;
                m_data_phase.slice(k).col(j) = ascan_arg;
            }
            if (m_visualize_frames) {
                arma::mat matrix = m_data_amplitude.slice(k);
                imagesc(matrix, "magnitude", 1);
                matrix = m_data_phase.slice(k);
                imagesc(matrix, "phase", 1);
            }
            logger.log(Logger::LogLevel::INFO, "Recon", "Process", std::format("Reconstructing {:.2f}%\r", 100.f*k/m_data_bin.n_slices));
        }
        logger.log(Logger::LogLevel::INFO, "Recon", "Process", std::format("Reconstructing {:.2f}%", 100.f));
        logger.log(Logger::LogLevel::INFO, "Recon", "Writing", "Writing to Tiff File");
        saveArmaCubeToMultipageTIFF(m_data_amplitude, outputname_amplitude);
        saveArmaCubeToMultipageTIFF(m_data_phase, outputname_phase);
        logger.log(Logger::LogLevel::TRACE, "Recon", "Success", "Successfully");
    }
    catch (const std::exception& e)
    {
        logger.log(Logger::LogLevel::ERROR, "Recon", "Process", "Failed for: " + std::filesystem::path(imagename).filename().string());
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
    int maxIteration = *config["maxIteration"].value<bool>();
    float rho = *config["rho"].value<float>();
    float lambda = *config["lambda"].value<float>();
    bool visualize_frames = *config["visualize_frames"].value<bool>();

    // Adjust according to the actual project directory structure
    logger.log(Logger::LogLevel::INFO, "Data", "Init", "Directory scan started: " + path);
    DataStorage ds;
    ds.getFromFolder(path, extension);
    logger.log(Logger::LogLevel::INFO, "Data", "Load", "Directory scan complete");

    // Perform image reconstruction
    Recon recon(maxIteration, rho, lambda, visualize_frames);

    for (int i = 0; i < ds.length; i++)
    {
        std::string filepath = ds.readname(i);
        std::string filename = std::filesystem::path(filepath).filename().string();
        
        bool isSuccess = recon.readData(filepath) && recon.reconstruction();
        if (!(isSuccess)) continue;
    }

    logger.log(Logger::LogLevel::INFO, "System", "Shutdown", "Program completed successfully");
    return 0;
}