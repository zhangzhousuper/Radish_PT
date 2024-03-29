#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

class GuiDataContainer {
public:
  GuiDataContainer() : TracedDepth(0) {}
  int TracedDepth;
};

namespace utilityCore {
extern float clamp(float f, float min, float max);
extern glm::vec3 clampRGB(glm::vec3 baseColor);
extern bool replaceString(std::string &str, const std::string &from,
                          const std::string &to);
extern std::vector<std::string> tokenizeString(std::string str);
extern std::string convertIntToString(int number);
extern std::istream &
safeGetline(std::istream &is,
            std::string &t); // Thanks to http://stackoverflow.com/a/6089413
} // namespace utilityCore
