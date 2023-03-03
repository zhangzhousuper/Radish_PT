#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "glslUtility.hpp"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>


#include "image.h"
#include "mathUtil.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"


using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Scene *scene;
extern int iteration;

extern int width;
extern int height;

void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mods);
void mousePositionCallback(GLFWwindow *window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
