#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
extern GLuint pbo;

std::string currentTimeString();
bool init();
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);

enum SampleMode { BSDF, DirectLi, MIS };