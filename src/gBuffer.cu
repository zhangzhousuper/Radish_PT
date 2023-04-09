#include "gBuffer.h"

__global__ void renderGBuffer(DevScene *scene, Camera cam, GBuffer gBuffer) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam.resolution.x || y >= cam.resolution.y) {
    return;
  }
  int idx = x + y * cam.resolution.x;

  float aspect = cam.resolution.x / (float)cam.resolution.y;
  float tanFovY = glm::tan(glm::radians(cam.fov.y));
  glm::vec2 pixelsize = 1.f / glm::vec2(cam.resolution);
  glm::vec2 scr = glm::vec2(x, y) * pixelsize;
  glm::vec2 ruv = scr + pixelsize * glm::vec2(0.5f);
  ruv = 1.f - ruv * 2.f;

  glm::vec3 pLens(0.f);
  glm::vec3 pFocus =
      glm::vec3(ruv * glm::vec2(aspect, 1.f) * tanFovY, 1.f) * cam.focalDist;
  glm::vec3 dir = pFocus - pLens;

  Ray ray;
  ray.origin = cam.position + cam.right * pLens.x + cam.up * pLens.y;
  ray.direction = glm::normalize(glm::mat3(cam.right, cam.up, cam.view) * dir);

  Intersection intersect;
  scene->intersect(ray, intersect);

  if (intersect.primId != NullPrimitive) {
    bool isLight =
        scene->materials[intersect.matId].type == Material::Type::Light;
    int matId = intersect.matId;
    if (isLight) {
      matId = NullPrimitive - 1;
#if SCENE_LIGHT_SINGLE_SIDED
      if (glm::dot(intersect.norm, ray.direction) < 0.f) {
        intersect.primId = NullPrimitive;
      }
#endif
    }
    Material material = scene->getTexturedMaterialAndSurface(intersect);

    gBuffer.albedo[idx] = material.baseColor;
    gBuffer.getNormal()[idx] = ENCODE_NORM(intersect.norm);
    gBuffer.getPrimId()[idx] = matId;
#if DENOISER_ENCODE_POSITION
    gBuffer.getDepth()[idx] = glm::distance(intersect.pos, ray.origin);
#else
    gBuffer.getPos()[idx] = intersect.pos;
#endif
    glm::ivec2 lastPos = gBuffer.lastCam.getRasterCoord(intersect.pos);
    if (lastPos.x >= 0 && lastPos.x < gBuffer.width && lastPos.y >= 0 &&
        lastPos.y < gBuffer.height) {
      gBuffer.motion[idx] = lastPos.y * cam.resolution.x + lastPos.x;
    } else {
      gBuffer.motion[idx] = -1;
    }
  } else {
    glm::vec3 albedo(0.f);
    if (scene->envMap != nullptr) {
      glm::vec2 uv = Math::toPlane(ray.direction);
      albedo = scene->envMap->linearSample(uv);
    }
    gBuffer.albedo[idx] = albedo;
    gBuffer.getNormal()[idx] = GBuffer::NormT(0.f);
    gBuffer.getPrimId()[idx] = NullPrimitive;
#if DENOISER_ENCODE_POSITION
    gBuffer.getDepth()[idx] = 1.f;
#else
    gBuffer.getPos()[idx] = glm::vec3(0.f);
#endif
    gBuffer.motion[idx] = 0;
  }
}

void GBuffer::update(const Camera &cam) {
  lastCam = cam;
  frameIdx ^= 1;
}

void GBuffer::render(DevScene *scene, const Camera &cam) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  constexpr int BlockSize = 8;
  dim3 blockSize(BlockSize, BlockSize);
  dim3 blockNum(ceilDiv(cam.resolution.x, BlockSize),
                ceilDiv(cam.resolution.y, BlockSize));
  cudaEventRecord(start, 0);
  renderGBuffer<<<blockNum, blockSize>>>(scene, cam, *this);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GBuffer runtime%.3f ms\n", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCUDAError("renderGBuffer");
}