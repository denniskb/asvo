#include "Texture.h"

#include <vector_types.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;

Texture::Texture() : m_pData(nullptr), m_width(0), m_height(0) {}

Texture::Texture(char const* fileName, int width, int height)
    : m_pData(nullptr), m_width(width), m_height(height) {
  // TODO: Handle errors more gracefully
  if (fileName != nullptr) {
    int resolution = width * height;

    FILE* file;
    file = fopen(fileName, "rb");

    assert(file != nullptr);

    std::vector<char> hImgRGB(3 * resolution);

    fread(&hImgRGB[0], 1, 3 * resolution, file);
    fclose(file);

    std::vector<uchar4> hImgRGBA(resolution);
    for (int i = 0; i < resolution; ++i) {
      hImgRGBA[i] = make_uchar4(hImgRGB[3 * i], hImgRGB[3 * i + 1],
                                hImgRGB[3 * i + 2], 255);
    }

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&m_pData, &desc, width, height);
    cudaMemcpyToArray(m_pData, 0, 0, &hImgRGBA[0], resolution * sizeof(uchar4),
                      cudaMemcpyHostToDevice);

    initTexture();
  }
}

Texture::Texture(Texture const& copy) : m_pData(nullptr) { copyFrom(copy); }

Texture& Texture::operator=(Texture const& rhs) {
  copyFrom(rhs);

  return *this;
}

Texture::~Texture() {
  cudaDestroyTextureObject(m_texture);
  cudaFreeArray(m_pData);
}

int Texture::width() const { return m_width; }

int Texture::height() const { return m_height; }

cudaTextureObject_t const& Texture::textureObject() const { return m_texture; }

void Texture::copyFrom(Texture const& other) {
  if (this == &other) {
    return;
  }

  m_width = other.width();
  m_height = other.height();
  int resolution = m_width * m_height;

  if (m_pData != nullptr) {
    cudaDestroyTextureObject(m_texture);
    cudaFreeArray(m_pData);
  }

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
  cudaMallocArray(&m_pData, &desc, m_width, m_height);

  cudaMemcpyArrayToArray(m_pData, 0, 0, other.m_pData, 0, 0,
                         resolution * sizeof(uchar4), cudaMemcpyDeviceToDevice);

  initTexture();
}

void Texture::initTexture() {
  cudaResourceDesc resDesc;
  std::memset(&resDesc, 0, sizeof(resDesc));

  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = m_pData;

  cudaTextureDesc texDesc;
  std::memset(&texDesc, 0, sizeof(texDesc));

  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.normalizedCoords = true;
  texDesc.readMode = cudaReadModeNormalizedFloat;

  cudaCreateTextureObject(&m_texture, &resDesc, &texDesc, nullptr);
}