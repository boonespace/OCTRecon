#include "DataStorage.h"

// Takes a folder path and file extension as parameters.
DataStorage::DataStorage(std::string folderPath, std::string extension)
{
    getFromFolder(folderPath, extension);
}

// Method to retrieve all files with given extension from specified folder.
void DataStorage::getFromFolder(std::string folderPath, std::string extension)
{

    // Check if folder exists. Throw exception if not.
    bool isexist = std::filesystem::exists(folderPath);
    if (!isexist)
    {
        throw std::invalid_argument("Folder does not exist or is not a directory");
    }

    // Iterate over each entry in the folder and add file names with given extension to files vector.
    for (const auto &entry : std::filesystem::directory_iterator(folderPath))
    {
        if (entry.path().extension() == extension)
            files.push_back(entry.path().string());
    }

    // Sort file names in the files vector.
    std::sort(files.begin(), files.end());

    // Update length variable with number of files in the folder.
    length = files.size();
}

// Method that returns name of file at given index.
std::string DataStorage::readname(size_t index)
{
    return (index < length) ? files[index] : files[length - 1];
}