#include "OCTRecon.h"
#include "Config.h"
#include <omp.h>

int main()
{
    std::setlocale(LC_ALL, ".UTF-8");

    // Configure
    Config config("config.toml");

    // Adjust according to the actual project directory structure
    DataStorage ds(config.path, config.extension);

    // Perform image reconstruction
    Recon recon(config);

    for (int i = 0; i < ds.length; i++)
    {
        std::string filepath = ds.readname(i);
        std::string filename = std::filesystem::path(filepath).filename().string();

        bool isSuccess = recon.readData(filepath) && recon.reconstruction();
        if (!(isSuccess)) continue;
    }

    return 0;
}