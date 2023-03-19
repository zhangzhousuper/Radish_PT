#include "common.h"

int Settings::traceDepth = 0;
int Settings::toneMapping = ToneMapping::ACES;
int Settings::tracer = Tracer::Streamed;
bool Settings::sortMaterial = false;

bool State::camChanged = true;