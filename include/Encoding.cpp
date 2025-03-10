#include "Encoding.h"
#include <windows.h>

// GBK To UTF-8
std::string GbkToUtf8(const std::string& gbkStr) {
    // The number of wide characters required to convert GBK to UTF-16
    int wLen = MultiByteToWideChar(CP_ACP, 0, gbkStr.c_str(), -1, nullptr, 0);
    if (wLen <= 0) return "";

    std::wstring wstr(wLen, 0);
    MultiByteToWideChar(CP_ACP, 0, gbkStr.c_str(), -1, &wstr[0], wLen);

    // The number of bytes required to convert UTF-16 to UTF-8
    int utf8Len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (utf8Len <= 0) return "";

    std::string utf8Str(utf8Len - 1, 0);  // -1 removes the trailing '\0'
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &utf8Str[0], utf8Len, nullptr, nullptr);

    return utf8Str;
}

// UTF-8 To GBK
std::string Utf8ToGbk(const std::string& utf8_str) {
    // First convert UTF-8 to UTF-16 (wide characters)
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, NULL, 0);
    std::wstring wide_str(wide_len, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &wide_str[0], wide_len);

    // Convert UTF-16 to GBK
    int gbk_len = WideCharToMultiByte(CP_ACP, 0, wide_str.c_str(), -1, NULL, 0, NULL, NULL);
    std::string gbk_str(gbk_len, 0);
    WideCharToMultiByte(CP_ACP, 0, wide_str.c_str(), -1, &gbk_str[0], gbk_len, NULL, NULL);

    return gbk_str;
}