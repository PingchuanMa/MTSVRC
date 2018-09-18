#include <iostream>
#include <unordered_map>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvvl_loader.h"
#include "VideoLoader.h"

using PictureSequence = NVVL::PictureSequence;
using LayerDesc = NVVL::LayerDesc;

template<typename T>
SingleVideoLoader<T>::SingleVideoLoader(
        uint16_t count,
        uint16_t start_frame,
        uint16_t interval,
        uint16_t key_base,
        uint16_t max_loader,
        uint16_t clear_freq,
        uint16_t device_id,
        LogLevel log_level)
    : count{count}, start_frame{start_frame}, interval{interval},
      key_base{key_base}, max_loader{max_loader},
      clear_freq{clear_freq}, device_id{device_id}, log_level{log_level},
      num_called{0}, num_loader{0}, loader_sz2idx{
          std::make_unique<std::unordered_map<
              uint32_t, uint16_t> >()}, loader_pool{
          std::make_unique<std::vector<
              std::shared_ptr<NVVL::VideoLoader> > >(
                  max_loader, std::make_shared<NVVL::VideoLoader>(
                      device_id, log_level
      ))}, hit_counter{
          std::make_unique<std::vector<float> >(max_loader, 0)} {
}

// Support NCHW Only
template<typename T>
void SingleVideoLoader<T>::create_layer_desc(
        uint16_t width, uint16_t height,
        uint16_t scale_width, uint16_t scale_height,
        uint16_t scale_shorter_side,
        uint16_t crop_x, uint16_t crop_y,
        uint16_t test_crops,
        float mean_r, float mean_g, float mean_b,
        float std_r, float std_g, float std_b,
        bool center_crop, bool normalized, bool horiz_flip,
        NVVL::ColorSpace color_space,
        NVVL::ChromaUpMethod chroma_up_method,
        NVVL::ScaleMethod scale_method) {

    uint16_t channels = 0;
    switch (color_space) {
    case ColorSpace_RGB:
        channels = 3;
        break;
    case ColorSpace_YCbCr:
        channels = 3;
        break;
    default:
        // More color spaces to be done
        throw std::runtime_error("Unknown Color Space.");
    }
    layer_desc.count = count;
    layer_desc.channels = channels;
    layer_desc.width = width;
    layer_desc.height = height;
    layer_desc.scale_width = scale_width;
    layer_desc.scale_height = scale_height;
    layer_desc.scale_shorter_side = scale_shorter_side;
    layer_desc.crop_x = crop_x;
    layer_desc.crop_y = crop_y;
    layer_desc.test_crops = test_crops;
    layer_desc.mean.r = mean_r;
    layer_desc.mean.g = mean_g;
    layer_desc.mean.b = mean_b;
    layer_desc.std.r = std_r;
    layer_desc.std.g = std_g;
    layer_desc.std.b = std_b;
    layer_desc.center_crop = center_crop;
    layer_desc.horiz_flip = horiz_flip;
    layer_desc.normalized = normalized;
    layer_desc.color_space = color_space;
    layer_desc.chroma_up_method = chroma_up_method;
    layer_desc.scale_method = scale_method;
    // We don't need to set stride since it will be done
    // when pitch-wise memory allocation
    //
    // layer_desc.stride.x = 1;
    // layer_desc.stride.y = width * layer_desc.stride.x;
    // layer_desc.stride.c = height * layer_desc.stride.y;
    // layer_desc.stride.n = channes * layer_desc.stride.c;
}

template<typename T>
auto SingleVideoLoader<T>::video_frames(const char *filename) {
    // auto video_size = nvvl_video_size_from_file(filename);
    // uint32_t video_size_pair = ((uint32_t)video_size.width << 16) + (uint32_t)video_size.height;
    // if (loader_sz2idx->find(video_size_pair) == loader_sz2idx->end()) {
    //     if (num_loader < max_loader) {
    //         loader_sz2idx->insert(std::make_pair(video_size_pair, num_loader));
    //         loader_pool->at(num_loader++) = std::make_shared<NVVL::VideoLoader>(
    //             device_id, log_level);
    //     } else {
    //         auto drop_index = std::distance(hit_counter->begin(),
    //                                         std::min_element(hit_counter->begin(), 
    //                                                          hit_counter->end()));
    //         delete loader_pool->at(drop_index).get();
    //         loader_pool->at(drop_index) = std::make_shared<NVVL::VideoLoader>(
    //             device_id, log_level);
    //         for (auto it = loader_sz2idx->begin(); it != loader_sz2idx->end(); ++it) {
    //             if (it->second == drop_index) {
    //                 loader_sz2idx->erase(it);
    //                 break;
    //             }
    //         }
    //         loader_sz2idx->insert(std::make_pair(video_size_pair, drop_index));
    //         hit_counter->at(drop_index) = 0.0;
    //     }
    // } else {
    //     hit_counter->at(loader_sz2idx->at(video_size_pair)) += 5.0;
    // }
    // if (num_loader >= max_loader) {
    //     std::for_each(hit_counter->begin(),
    //                   hit_counter->end(), [](float &i) { i -= 1.0; });
    // }
    // auto loader = loader_pool->at(loader_sz2idx->at(video_size_pair));
    auto loader = std::make_shared<NVVL::VideoLoader>(device_id, log_level);
    auto seq = std::make_shared<PictureSequence>(count, device_id);
    auto layer = PictureSequence::Layer<T>();
    layer.desc = layer_desc;
    layer.data = get_data<T>(layer.desc);
    loader->read_sequence(filename, start_frame, count, interval, key_base);
    seq->set_layer("data", layer);
    if (++num_called == clear_freq) {
        num_called = 0;
        std::for_each(hit_counter->begin(),
                      hit_counter->end(), [this](float &i) { i /= this->clear_freq; });
    }
    seq->wait();
    seq = nullptr;
    loader = nullptr;
    std::cout << layer.desc.stride.n << " " <<
            layer.desc.stride.c << " " <<
            layer.desc.stride.y << " " <<
            layer.desc.stride.x << std::endl;
    return layer.data;
}

int main() {
    SingleVideoLoader<float> loader;
    loader.create_layer_desc();
    for (auto i = 0; i < 100; ++i) {
        auto a = loader.video_frames("/data/1000661895.mp4");
    }
    return 0;
}