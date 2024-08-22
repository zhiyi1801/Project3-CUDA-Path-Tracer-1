#include "main.h"
#include "preview.h"
#include <cstring>

extern "C"
{
	__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

bool camchanged = true;
SampleMode sampleMode = SampleMode::MIS;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
bool posInit = true;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

void testfunc()
{
	Bounds3 box{ glm::vec3(0), glm::vec3(30, 30, 30) };
	Ray ray{ glm::vec3(10,10,40), glm::vec3(0,0,-1) };
	bool b = box.IntersectP(ray);
}

int main(int argc, char** argv) {

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		std::cout << "No CUDA-capable devices were detected." << std::endl;
	}
	else {
		for (int device = 0; device < deviceCount; ++device) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, device);

			std::cout << "Device " << device << ":\"" << prop.name << "\""
				<< " compute capability: " << prop.major << "." << prop.minor
				<< ", core clock: " << prop.clockRate / 1000 << " MHz"
				<< ", memory clock: " << prop.memoryClockRate / 1000 << " MHz"
				<< ", memory bus width: " << prop.memoryBusWidth << " bits"
				<< ", L2 cache size: " << prop.l2CacheSize << " bytes"
				<< ", max threads per block: " << prop.maxThreadsPerBlock
				<< ", max blocks per SM: " << prop.maxBlocksPerMultiProcessor
				<< ", regs per block: " << prop.regsPerBlock
				<< ", warp size: " << prop.warpSize
				<< std::endl;
		}
	}

	//testfunc();
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];

	// Load scene file
	scene = new Scene(sceneFile);


	// Set the data that will be passed to the gpu
	//scene->setDevData();

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	if (posInit)
	{
		glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
		glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
		phi = glm::degrees(glm::atan(view.z, view.x));
		theta = glm::degrees(glm::sin(view.y));
		theta = glm::clamp(theta, -89.f, 89.f);
	}
	else
	{
		float radianTheta = glm::radians(theta), radianPhi = glm::radians(phi);
		cam.view = glm::vec3(glm::cos(radianTheta) * glm::cos(radianPhi), glm::sin(radianTheta), glm::cos(radianTheta) * glm::sin(radianPhi));
		cam.lookAt = cam.position + cam.view;
	}

	cam.right = glm::cross(cam.view, cam.up);
	cam.up = glm::cross(cam.right, cam.view);
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);
	camchanged = true;

	// Initialize CUDA and GL components
	init();

	// Initialize ImGui Data
	InitImguiData(guiData);
	InitDataContainer(guiData);

	scene->setDevData();
	// GLFW main loop
	mainLoop();

	Resource::clear();
	scene->clear();

	return 0;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index] / samples;
#if TONEMAPPING
			pix = gammaCorrection(ACESFilm(pix));
#endif // TONEMAPPING
			img.setPixel(width - 1 - x, y, glm::vec3(pix));
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
/*		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta)*/;

		//cam.view = -glm::normalize(cameraPosition);
		float radianTheta = glm::radians(theta), radianPhi = glm::radians(phi);
		cam.view = glm::vec3(glm::cos(radianTheta) * glm::cos(radianPhi), glm::sin(radianTheta), glm::cos(radianTheta) * glm::sin(radianPhi));
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::normalize(glm::cross(r, v));
		cam.right = glm::normalize(r);

		//cam.position = cameraPosition;
		//cameraPosition += cam.lookAt;
		//cam.position = cameraPosition;
		camchanged = false;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathtraceFree();
		pathtraceInit(scene);
	}
	//if (iteration == 1)
	//{
	//	saveImage();
	//	pathtraceFree();
	//	cudaDeviceReset();
	//	exit(EXIT_SUCCESS);
	//}
	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		pathtrace(pbo_dptr, frame, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		saveImage();
		pathtraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
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
			Camera& cam = renderState->camera;
			cam.lookAt = ogLookAt;
			break;
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width * 40.f;
		theta += (ypos - lastY) / height * 40.f;
		theta = glm::clamp(theta, -89.f, 89.f);
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.position -= (float)(xpos - lastX) * right * 0.01f;
		cam.position += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}
