#pragma once

#include <string>
namespace whisper {

class MmapFile {
 public:
  MmapFile() = default;
  explicit MmapFile(const std::string& filepath);
  ~MmapFile();

  void* data() const { return data_; }
  size_t size() const { return size_; }

  // Disable copy and assignment
  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;

  MmapFile(MmapFile&& from) noexcept;

  MmapFile& operator=(MmapFile&& from) noexcept;

 private:
  void consume(MmapFile& from);
  void reset();

  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
};
}  // namespace whisper
