#include "main.h"
#include "preview.h"
#include <cstring>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
GuiDataContainer *guiData;
RenderState *renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

void testAABB() {
  AABB boxes[] = {
      {glm::vec3(-1.f), glm::vec3(1.f)}, {glm::vec3(0.f), glm::vec3(1.f)},
      {glm::vec3(0.f), glm::vec3(1.f)},  {glm::vec3(0.f), glm::vec3(1.f)},
      {glm::vec3(0.f), glm::vec3(1.f)},
  };

  Ray ray[] = {
      {glm::vec3(-0.1f), glm::normalize(glm::vec3(1.f, 0.f, 0.f))},
      {glm::vec3(0.f, 0.1f, 0.5f), glm::normalize(glm::vec3(1.f, 1.f, 0.f))},
      {glm::vec3(-1.f), glm::normalize(glm::vec3(1.f, 0.f, 0.f))},
      {glm::vec3(1.1f), glm::normalize(glm::vec3(1.f, 1.f, 0.f))},
      {glm::vec3(2.f), glm::normalize(glm::vec3(-1.f))},
  };

  for (int i = 0; i < sizeof(boxes) / sizeof(AABB); i++) {
    float dist;
    bool intersec = boxes[i].intersect(ray[i], dist);
    std::cout << intersec << " " << dist << "\n";
  }
}

/**
 * GLM intersection returns false when triangle is back-faced
 */
void testTriangle() {
  glm::vec3 v[] = {glm::vec3(-1.f, -1.f, 0.f), glm::vec3(1.f, -1.f, 0.f),
                   glm::vec3(1.f, 1.f, 0.f)};
  glm::vec3 ori(0.f, 0.f, 1.f);
  glm::vec3 dir(0.f, 0.f, -1.f);
  glm::vec2 bary;
  float dist;
  bool hit = intersectTriangle({ori, dir}, v[0], v[1], v[2], bary, dist);
  std::cout << hit << " "
            << vec3ToString(glm::vec3(1.f - bary.x - bary.y, bary)) << "\n";
  glm::vec3 hitPos =
      v[0] * (1.f - bary.x - bary.y) + v[1] * bary.x + v[2] * bary.y;
  std::cout << vec3ToString(hitPos) << "\n";
  hit = intersectTriangle({-ori, -dir}, v[0], v[1], v[2], bary, dist);
  std::cout << hit << " "
            << vec3ToString(glm::vec3(1.f - bary.x - bary.y, bary)) << "\n";
}

void testDiscreteSampler() {
  std::vector<float> distrib = {.1f, .2f, .3f, .4f, 2.f, 3.f, 4.f};

  DiscreteSampler<float> sampler(distrib);

  int stats[7] = {0, 0, 0, 0, 0, 0, 0};

  thrust::default_random_engine rng(time(nullptr));

  for (int i = 0; i < 100000; i++) {
    float r1 = thrust::uniform_real_distribution<float>(0.f, 1.f)(rng);
    float r2 = thrust::uniform_real_distribution<float>(0.f, 1.f)(rng);
    stats[sampler.sample(r1, r2)]++;
  }

  for (auto i : stats) {
    std::cout << i << " ";
  }
  std::cout << "\n";
}

int main(int argc, char **argv) {
  startTimeString = currentTimeString();

  if (argc < 2) {
    printf("Usage: %s SCENEFILE.txt\n", argv[0]);
    return 1;
  }

  const char *sceneFile = argv[1];

  testDiscreteSampler();
  exit(0);
  // Load scene file
  scene = new Scene(sceneFile);

  // Create Instance for ImGUIData
  guiData = new GuiDataContainer();

  // Set up camera stuff from loaded path tracer settings
  iteration = 0;
  renderState = &scene->state;
  Camera &cam = renderState->camera;
  width = cam.resolution.x;
  height = cam.resolution.y;

  glm::vec3 view = cam.view;
  glm::vec3 up = cam.up;
  glm::vec3 right = glm::cross(view, up);
  up = glm::cross(right, view);

  cameraPosition = cam.position;

  // compute phi (horizontal) and theta (vertical) relative 3D axis
  // so, (0 0 1) is forward, (0 1 0) is up
  glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
  glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
  phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
  theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
  ogLookAt = cam.lookAt;
  zoom = glm::length(cam.position - ogLookAt);

  // Initialize CUDA and GL components
  init();

  // Initialize ImGui Data
  InitImguiData(guiData);
  InitDataContainer(guiData);

  scene->buildDevData();
  // GLFW main loop
  mainLoop();

  scene->clear();
  Resource::clear();

  return 0;
}

void saveImage() {
  float samples = iteration;
  // output image file
  Image img(width, height);

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      int index = x + (y * width);
      glm::vec3 pix = renderState->image[index] / samples;
      pix = Math::gammaCorrection(Math::ACES(pix));
      img.setPixel(width - 1 - x, y, pix);
    }
  }

  std::string filename = renderState->imageName;
  std::ostringstream ss;
  ss << filename << "." << startTimeString << "." << samples << "samp";
  filename = ss.str();

  // CHECKITOUT
  img.savePNG(filename);
  // img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
  if (camchanged) {
    iteration = 0;
    Camera &cam = renderState->camera;
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.y = zoom * cos(theta);
    cameraPosition.z = zoom * cos(phi) * sin(theta);

    cam.view = -glm::normalize(cameraPosition);
    glm::vec3 v = cam.view;
    glm::vec3 u = glm::vec3(0, 1, 0); // glm::normalize(cam.up);
    glm::vec3 r = glm::cross(v, u);
    cam.up = glm::cross(r, v);
    cam.right = r;

    cam.position = cameraPosition;
    cameraPosition += cam.lookAt;
    cam.position = cameraPosition;
    camchanged = false;
  }

  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
  // use this buffer

  if (iteration == 0) {
    pathTraceFree();
    pathTraceInit(scene);
  }

  if (iteration < renderState->iterations) {
    uchar4 *pbo_dptr = NULL;
    iteration++;
    cudaGLMapBufferObject((void **)&pbo_dptr, pbo);

    // execute the kernel
    int frame = 0;
    pathTrace(pbo_dptr, frame, iteration);

    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);
  } else {
    saveImage();
    pathTraceFree();
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
  }
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mods) {
  if (action == GLFW_PRESS) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
      saveImage();
      glfwSetWindowShouldClose(window, GL_TRUE);
      break;
    case GLFW_KEY_S:
      saveImage();
      break;
    case GLFW_KEY_SPACE:
      camchanged = true;
      renderState = &scene->state;
      Camera &cam = renderState->camera;
      cam.lookAt = ogLookAt;
      break;
    }
  }
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
  if (MouseOverImGuiWindow()) {
    return;
  }
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed =
      (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed =
      (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow *window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY)
    return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  } else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  } else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
    cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
