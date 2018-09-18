#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <memory>
#include <utility>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "VideoLoader.h"


using PictureSequence = NVVL::PictureSequence;
using LayerDesc = NVVL::LayerDesc;

template<typename T>
T* new_data(size_t *pitch, size_t width, size_t height) {
    T* data;
    if (cudaMallocPitch(&data, pitch, width * sizeof(T), height) != cudaSuccess) {
        throw std::runtime_error("Unable to allocate buffer in device memory");
    }
    return data;
}

// just use one buffer for each different type
template<typename T>
auto get_data(NVVL::LayerDesc &layer_desc) {
    size_t pitch;
    auto data = std::unique_ptr<T, decltype(&cudaFree)>{
        new_data<T>(&pitch, layer_desc.width,
                    layer_desc.height * layer_desc.count * layer_desc.test_crops * 3),
        cudaFree};
    layer_desc.stride.y = pitch / sizeof(T);
    layer_desc.stride.x = 1;
    layer_desc.stride.c = layer_desc.stride.y * layer_desc.height;
    layer_desc.stride.n = layer_desc.stride.c * layer_desc.channels;
    return data.get();
}

template<typename T>
class SingleVideoLoader {
  private:
    NVVL::LayerDesc layer_desc;
    uint16_t count;
    uint16_t sequence_count;
    uint16_t start_frame;
    uint16_t interval;
    uint16_t key_base;
    uint16_t max_loader;
    uint16_t clear_freq;
    uint16_t device_id;
    LogLevel log_level;
    std::unique_ptr<std::unordered_map<
        uint32_t, uint16_t> > loader_sz2idx;
    std::unique_ptr<std::vector<
        std::shared_ptr<NVVL::VideoLoader> > > loader_pool;
    std::unique_ptr<std::vector<float> > hit_counter;
    uint16_t num_called;
    uint16_t num_loader;
  public:
    SingleVideoLoader(
        uint16_t count = 4,
        uint16_t start_frame = 0,
        uint16_t interval = 1,
        uint16_t key_base = 0,
        uint16_t max_loader = 20,
        uint16_t clear_freq = 500,
        uint16_t device_id = 0,
        LogLevel log_level = LogLevel_Error);
    void create_layer_desc(
        uint16_t width = 224,
        uint16_t height = 224,
        uint16_t scale_width = 0,
        uint16_t scale_height = 0,
        uint16_t scale_shorter_side = 256,
        uint16_t crop_x = 0,
        uint16_t crop_y = 0,
        uint16_t test_crops = 1,
        float mean_r = 0.5,
        float mean_g = 0.5,
        float mean_b = 0.5,
        float std_r = 1.0,
        float std_g = 1.0,
        float std_b = 1.0,
        bool center_crop = true,
        bool normalized = true,
        bool horiz_flip = false,
        NVVL::ColorSpace color_space = ColorSpace_RGB,
        NVVL::ChromaUpMethod chroma_up_method = ChromaUpMethod_Linear,
        NVVL::ScaleMethod scale_method = ScaleMethod_Linear);
    auto video_frames(const char *filename);
};
