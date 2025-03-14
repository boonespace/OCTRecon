#ifndef DATASTORAGE_H
#define DATASTORAGE_H

#include <vector>
#include <string>
#include <filesystem>
#include "Logger.h"

// DataStorage class to store and retrieve data from a folder.
class DataStorage {
public:

    
    std::vector<std::string> files; // Vector of file names in the folder.    
    size_t length; // Number of files in the folder.

    Logger logger;

    DataStorage() = default;

    // Takes a folder path and file extension as parameters.
    DataStorage(std::string folderPath, std::string extension);

    // Method to retrieve all files with given extension from specified folder.
    void getFromFolder(std::string folderPath, std::string extension);

    // Method that returns name of file at given index.
    std::string readname(size_t index);
};

#endif // DATASTORAGE_H