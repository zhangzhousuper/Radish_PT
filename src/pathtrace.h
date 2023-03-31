#pragma once

#include <device_launch_parameters.h>

#include "scene.h"
#include <vector>

void InitDataContainer(GuiDataContainer *guiData);
void pathTraceInit(Scene *scene);
void pathTraceFree();
// void pathTrace(glm::vec3 *DirectIllum, glm::vec3 *IndirectIllum);
void pathTrace(glm::vec3 *DirectIllum, glm::vec3 *IndirectIllum, int iter);

void copyImageToPBO(uchar4 *devPBO, glm::vec3 *devImage, int width, int height,
                    int toneMapping, float scale = 1.f);
void copyImageToPBO(uchar4 *devPBO, glm::vec2 *devImage, int width, int height);
void copyImageToPBO(uchar4 *devPBO, float *devImage, int width, int height);
void copyImageToPBO(uchar4 *devPBO, int *devImage, int width, int height);
