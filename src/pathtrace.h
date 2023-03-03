#pragma once

#include <device_launch_parameters.h>

#include "scene.h"
#include <vector>

void InitDataContainer(GuiDataContainer *guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
