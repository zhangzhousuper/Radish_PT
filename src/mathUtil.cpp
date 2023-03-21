#include "mathUtil.h"

namespace Math {
bool epsilonCheck(float a, float b) {
  if (fabs(fabs(a) - fabs(b)) < EPSILON) {
    return true;
  } else {
    return false;
  }
}

glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation,
                                    glm::vec3 scale) {
  glm::mat4 translationMat = glm::translate(glm::mat4(1.0f), translation);
  glm::mat4 rotationMat = glm::rotate(glm::mat4(1.0f), rotation.x * PI / 180.f,
                                      glm::vec3(1.f, 0.f, 0.f));
  rotationMat =
      rotationMat * glm::rotate(glm::mat4(1.0f), rotation.y * PI / 180.f,
                                glm::vec3(0.f, 1.f, 0.f));
  rotationMat =
      rotationMat * glm::rotate(glm::mat4(1.0f), rotation.z * PI / 180.f,
                                glm::vec3(0.f, 0.f, 1.f));
  glm::mat4 scaleMat = glm::scale(glm::mat4(1.0), scale);
  return translationMat * rotationMat * scaleMat;
}
} // namespace Math