// Reads the shared vectors' flat text form: one case per line, space-separated,
// '#' comments. Host-only.
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

inline std::vector<std::vector<std::string>> read_cases(const std::string &path, const char *op) {
    std::vector<std::vector<std::string>> cases;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream words(line);
        std::vector<std::string> fields;
        for (std::string w; words >> w;) fields.push_back(w);
        if (!fields.empty() && fields[0] == op) cases.push_back(fields);
    }
    return cases;
}
