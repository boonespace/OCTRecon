#ifndef ENCODING_H
#define ENCODING_H

#include <iostream>
#include <string>

std::string GbkToUtf8(const std::string& gbkStr);
std::string Utf8ToGbk(const std::string& utf8_str);

#endif // ENCODING_H