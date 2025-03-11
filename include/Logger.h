#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <chrono>
#include <algorithm>
#include <unordered_map>

// Class representing a logger that outputs messages to the console (stdout) with different log levels and color support.
class Logger
{
public:

    // Enumerated type for logging level, which can be one of DEBUG, INFO, WARNING, or ERROR.
    enum class LogLevel
    {
        TRACE,
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    // Flag to indicate if the logger supports colored output. Default is true.
    bool supportcolor = true;

    // Minimum log level that will be recorded by this logger. Messages with a lower severity than this level will be ignored.
    LogLevel minLogLevel = LogLevel::INFO;

    // Method for logging messages. Takes in the log level, module name, step description, and message text as parameters.
    void log(LogLevel level, const std::string &module, const std::string &step, const std::string &message)
    {
        // Skip logging if current log level is lower than minimum log level
        if (level < minLogLevel)
            return;

        // Get current time and format it as "Year-Month-Day Hour:Minute:Second.Milliseconds"
        auto now = std::chrono::system_clock::now();
        std::time_t time_now = std::chrono::system_clock::to_time_t(now);
        std::tm local_tm = *std::localtime(&time_now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::ostringstream timestamp;
        timestamp << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S")           // Year-Month-Day Hour:Minute:Second
                  << "." << std::setfill('0') << std::setw(3) << ms.count(); // Milliseconds

        // Convert log level to string representation for use in the log message
        std::string levelStr = logLevelToString(level);

        // Format the log entry based on the log level, timestamp, module, step, and message
        // The format differs depending on whether color support is enabled or not
        std::ostringstream logMessage;
        if (!supportcolor)
            logMessage << timestamp.str() << " [" << levelStr << "] [" << module << "] [" << step << "] - " << message;
        else
        {
            logMessage << timecolor << timestamp.str() << nonecolor;
            logMessage << levelcolor(level) << " [" << levelStr << "]" << nonecolor;
            logMessage << modulecolor << " [" << module << "]" << nonecolor;
            logMessage << stepcolor << " [" << step << "]" << nonecolor;
            logMessage << " - " << message;
        }

        // Output the formatted log message to the console (stdout)
        if (message.back()!='\r')
            logMessage << std::endl;
        std::cout << logMessage.str();
        
        // // If the log level is ERROR, terminate the program immediately
        // if (level == LogLevel::ERROR)
        //     std::exit(1); // Use exit code 1 to signify an error
    }

    // Method for converting a LogLevel enumeration value into its string representation.
    std::string logLevelToString(LogLevel level)
    {
        static const std::unordered_map<LogLevel, std::string> logLevelMap = {
            {LogLevel::TRACE, "TRACE"},
            {LogLevel::DEBUG, "DEBUG"},
            {LogLevel::INFO, "INFO"},
            {LogLevel::WARNING, "WARNING"},
            {LogLevel::ERROR, "ERROR"},
        };

        auto it = logLevelMap.find(level);
        return (it != logLevelMap.end()) ? it->second : "INFO";
    }

    // Method for converting a string representation of a log level into its corresponding LogLevel enumeration value.
    LogLevel stringToLogLevel(std::string levelStr)
    {
        std::transform(levelStr.begin(), levelStr.end(), levelStr.begin(), ::toupper);
        static const std::unordered_map<std::string, LogLevel> logLevelMap = {
            {"TRACE", LogLevel::TRACE},
            {"DEBUG", LogLevel::DEBUG},
            {"INFO", LogLevel::INFO},
            {"WARNING", LogLevel::WARNING},
            {"ERROR", LogLevel::ERROR},
        };

        auto it = logLevelMap.find(levelStr);
        return (it != logLevelMap.end()) ? it->second : LogLevel::INFO; // Use the default value for incorrect configuration
    }

private:
    // Color codes used for different parts of the log message when color support is enabled.
    std::string timecolor = "\033[32m";
    std::string modulecolor = "\033[35m";
    std::string stepcolor = "\033[0m";
    std::string nonecolor = "\033[0m";

    // Method for getting the color code corresponding to a given log level.
    std::string levelcolor(LogLevel level)
    {
        switch (level)
        {
        case LogLevel::INFO:
            return "\033[32m";
        case LogLevel::DEBUG:
            return "\033[36m";
        case LogLevel::WARNING:
            return "\033[33m";
        case LogLevel::ERROR:
            return "\033[31m";
        case LogLevel::TRACE:
            return "\033[31m";
        default:
            return "\033[0m";
        }
    }
};

#endif // LOGGER_H