#include "OCTRecon.h"
#include "toml++/toml.hpp"

int main()
{
    // Configure
    std::string filename = "config.toml";
    // std::ifstream config_file(filename);
    auto config = toml::parse_file(filename);
    std::string path = *config["path"].value<std::string>();
    std::string extension = *config["extension"].value<std::string>();
    int size_x = *config["size_x"].value<int>();
    int size_y = *config["size_y"].value<int>();
    int size_z = *config["size_z"].value<int>();
    
    // Adjust according to the actual project directory structure
    DataStorage ds;
    ds.getFromFolder(path, extension);

    // Perform image reconstruction
    Recon recon(size_x, size_y, size_z, DataType::INT16);

    for (int i = 0; i < ds.length; i++)
    {
        recon.readData(ds.readname(i));
        recon.reconstruction();
    }

    return 0;
}