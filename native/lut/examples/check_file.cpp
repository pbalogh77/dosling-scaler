#include "dosling_scaler.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error(
          "Usage: lut-check model.bin frame.bin output.rgb");
    if (std::filesystem::exists(argv[3]))
      throw std::runtime_error("Output already exists");
    std::ifstream f(argv[2], std::ios::binary);
    uint32_t w, h, nr, seed;
    f.read((char *)&w, 4);
    f.read((char *)&h, 4);
    f.read((char *)&nr, 4);
    f.read((char *)&seed, 4);
    if (!f || !w || !h || w > 4096 || h > 4096 || nr > 65536)
      throw std::runtime_error("Invalid frame");
    std::vector<uint8_t> source(size_t(w) * h * 3), out(size_t(w) * h * 27);
    std::vector<dl_rect> rs(nr);
    f.read((char *)source.data(), source.size());
    f.seekg(std::streamoff(size_t(w) * h * 27), std::ios::cur);
    f.read((char *)rs.data(), rs.size() * sizeof(dl_rect));
    if (!f)
      throw std::runtime_error("Truncated frame");
    std::unique_ptr<dl_context, decltype(&dl_destroy)> c(dl_create(argv[1]),
                                                         dl_destroy);
    if (!c)
      throw std::runtime_error("Invalid model");
    dl_stats stats{};
    int status =
        dl_scale3(c.get(), source.data(), w, h, size_t(w) * 3, out.data(),
                  size_t(w) * 9, rs.data(), rs.size(), &stats);
    if (status)
      throw std::runtime_error("Scaler status " + std::to_string(status));
    std::ofstream dst(argv[3], std::ios::binary);
    dst.write((char *)out.data(), out.size());
    if (!dst)
      throw std::runtime_error("Output write failed");
    std::cout << w << 'x' << h << " -> " << w * 3 << 'x' << h * 3
              << " model=" << dl_model_bytes(c.get())
              << " pixels=" << stats.pixels << " extended=" << stats.extended
              << " fallback=" << stats.hq_fallback << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
