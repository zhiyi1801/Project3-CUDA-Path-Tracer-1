#include <iostream>
#include <string>
#include <stb_image.h>
#include <stb_image_write.h>

#include "image.h"
#include "utilities.h"

image::image(int x, int y) :
        width(x),
        height(y),
        pixels(new glm::vec3[x * y]) {
}

// from stb_image.h:
// If you load LDR images through this interface, those images will
// be promoted to floating point values, run through the inverse of
// constants corresponding to the above:
//
//     stbi_ldr_to_hdr_scale(1.0f);
//     stbi_ldr_to_hdr_gamma(2.2f);
image::image(std::string filePath, float gamma)
{
    int channels;
    stbi_ldr_to_hdr_gamma(gamma);
    float *data = stbi_loadf(filePath.c_str(), &width, &height, &channels, 3);

    if (!data)
    {
        std::cout << "\t[Fail to load image: " + filePath + "]" << stbi_failure_reason() <<  std::endl;
        pixels = nullptr;
        return;
    }
    pixels = new glm::vec3[width * height];
    memcpy(pixels, data, width * height * sizeof(glm::vec3));

    stbi_image_free(data);
}

image::~image() {
    if (pixels)
    {
       delete pixels;
    }
}

void image::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < width && y < height);
    pixels[(y * width) + x] = pixel;
}

void image::savePNG(const std::string &baseFilename) {
    savePNG(baseFilename, 3);
}

void image::savePNG(const std::string& baseFilename, int channels) {
    unsigned char* bytes = new unsigned char[3 * width * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char)pix.x;
            bytes[3 * i + 1] = (unsigned char)pix.y;
            bytes[3 * i + 2] = (unsigned char)pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), width, height, 3, bytes, width * channels);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), width, height, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}
