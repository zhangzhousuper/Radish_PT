#pragma once

#include <device_launch_parameters.h>

#include "scene.h"
#include <vector>

void InitDataContainer(GuiDataContainer *guiData);
void pathTraceInit(Scene *scene);
void pathTraceFree();
void pathTrace(uchar4 *pbo, int frame, int iteration);
