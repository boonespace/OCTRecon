#ifndef CONFIG_H
#define CONFIG_H

#include <fstream>
#include "toml++/toml.hpp"

class Config
{
public:
    std::string path;
    std::string extension;

    int maxIteration;
    float rho;
    float lambda;

    bool visualize_frames;
    std::string output_mode;
    std::vector<std::string> output_data;

public:

    Config(const std::string& filename)
    {
        auto config = toml::parse_file(filename);

        path = *config["path"].value<std::string>();
        extension = *config["extension"].value<std::string>();

        maxIteration = *config["maxIteration"].value<bool>();
        rho = *config["rho"].value<float>();
        lambda = *config["lambda"].value<float>();

        visualize_frames = *config["visualize_frames"].value<bool>();

        output_mode = *config["output_mode"].value<std::string>();

        auto output_data_list = config["output_data"].as_array();
        for (const auto& item : *output_data_list) {
            std::string str = *item.value<std::string>();
            output_data.push_back(str);
        }
    }
};

#endif // CONFIG_H