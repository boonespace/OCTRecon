#include "OCTRecon.h"
#include "Encoding.h"
#include "toml++/toml.hpp"

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
    std::string name = Utf8ToGbk(filename);
    cv::imwrite(name, images_cv);
}

Recon::Recon(size_t size_x, size_t size_y, size_t size_z, DataType dtype, int headerSize)
    : m_size_x(size_x), m_size_y(size_y), m_size_z(size_z), m_dtype(dtype), m_headerSize(headerSize)
{
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
}

bool Recon::readData(std::string filename)
{
    imagename = filename;

    std::filesystem::path image_path(imagename);
    outputname_amplitude = image_path.replace_filename(image_path.stem().string() + "_amplitude.tif").string();
    outputname_phase = image_path.replace_filename(image_path.stem().string() + "_phase.tif").string();

    // read bin file
    std::ifstream file(imagename, std::ios::binary);

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();

    if (fileSize != m_expectedSize)
    {
        file.close();
        std::cerr << "File size (" << fileSize << " bytes) does not match expected (" << m_expectedSize << " bytes)!" << std::endl;
        return false;
    }

    file.seekg(m_headerSize, std::ios::beg);
    if (!file)
    {
        std::cerr << "Error: Failed to seek to the data position!" << std::endl;
        return false;
    }

    size_t count = 0;
    char buffer[8];
    arma::Cube<int16_t> data(m_size_x, m_size_y, m_size_z);
    file.read(reinterpret_cast<char *>(data.memptr()), m_expectedSize);
    *m_data_bin = arma::conv_to<arma::cube>::from(data);
    file.close();
    return true;
}

bool Recon::reconstruction()
{
    try
    {
        for (int k = 0; k < m_data_bin->n_slices; k++)
        {
            for (int j = 0; j < m_data_bin->n_cols; j++)
            {
                // Extract A-Scan data and remove the first 8 bits of identifier
                arma::vec ascan = m_data_bin->slice(k).col(j);
                ascan.shed_rows(0, 7);
                // Add window to reduce spectrum leakage
                ascan %= m_window;
                // Fourier Transform
                arma::cx_vec ascan_fft = m_fft.computeFFT(ascan);
                // Extract amplitude and phase
                arma::vec ascan_abs = arma::abs(ascan_fft);
                arma::vec ascan_arg = arma::arg(ascan_fft);
                // // ADMM Iteration
                // arma::cx_vec r = ascan_fft; // init value
                // arma::cx_vec u(r.n_elem, arma::fill::zeros);
                // float rho = 0.01, lambda = 200;
                // for (int iter = 0; iter < 10; iter++){
                //     arma::cx_vec r_new = ascan_fft + rho*(r-u);
                //     float lambda_new = lambda / rho;
                //     arma::vec zeros(r.n_elem, arma::fill::zeros);
                //     arma::vec s = arma::abs(r_new)-lambda_new;
                //     r = arma::sign(r_new) % arma::max(arma::abs(r_new)-lambda_new, zeros);
                //     u = u + r;
                //     float error = arma::norm(r);
                //     if (error<1e-4)
                //         break;
                // }
                // ascan_abs = arma::abs(r);
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
            // arma::mat matrix = m_data_amplitude->slice(k);
            // imagesc(matrix, "magnitude", 1);
            // matrix = m_data_phase->slice(k);
            // imagesc(matrix, "phase", 1);
        }
        saveArmaCubeToMultipageTIFF(*m_data_amplitude, outputname_amplitude);
        saveArmaCubeToMultipageTIFF(*m_data_phase, outputname_phase);
    }
    catch (const std::exception& e)
    {
        return false;
    }
    return true;
}

int main()
{
    std::setlocale(LC_ALL, ".UTF-8");

    // Logger
    Logger logger;
    logger.log(Logger::LogLevel::INFO, "System", "Startup", "Program started");

    // Configure
    std::string filename = "config.toml";
    auto config = toml::parse_file(filename);
    std::string path = *config["path"].value<std::string>();
    std::string extension = *config["extension"].value<std::string>();
    int size_x = *config["size_x"].value<int>();
    int size_y = *config["size_y"].value<int>();
    int size_z = *config["size_z"].value<int>();
    logger.log(Logger::LogLevel::INFO, "Config", "Load", "Configuration loaded successfully");

    // Adjust according to the actual project directory structure
    logger.log(Logger::LogLevel::INFO, "Data", "Init", "Directory scan started: " + path);
    DataStorage ds;
    ds.getFromFolder(path, extension);
    logger.log(Logger::LogLevel::INFO, "Data", "Load", "Directory scan complete");

    // Perform image reconstruction
    logger.log(Logger::LogLevel::INFO, "Reconstruction", "Init", "Initializing reconstruction module");
    Recon recon(size_x, size_y, size_z, DataType::INT16);

    for (int i = 0; i < ds.length; i++)
    {
        std::string filepath = ds.readname(i);
        std::string filename = std::filesystem::path(filepath).filename().string();
        logger.log(Logger::LogLevel::INFO, "Reconstruction", "File", "Processing file: " + filename);

        if (!recon.readData(filepath))
        {
            logger.log(Logger::LogLevel::ERROR, "Reconstruction", "Read", "Failed to read file: " + filename);
            continue;
        }

        logger.log(Logger::LogLevel::INFO, "Reconstruction", "Process", "Reconstructing: " + filename);
        if (!recon.reconstruction())
        {
            logger.log(Logger::LogLevel::ERROR, "Reconstruction", "Process", "Reconstruction failed for: " + filename);
            continue;
        }

        logger.log(Logger::LogLevel::INFO, "Reconstruction", "Success", "Successfully reconstructed: " + filename);
    }

    logger.log(Logger::LogLevel::INFO, "System", "Shutdown", "Program completed successfully");
    return 0;
}