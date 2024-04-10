#include "mmap_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <stdexcept>

namespace whisper {

MmapFile::MmapFile(const std::string& filepath) {
  fd_ = open(filepath.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw std::runtime_error("Failed to open file: " + filepath);
  }

  struct stat st;
  if (fstat(fd_, &st) == -1) {
    close(fd_);
    throw std::runtime_error("Failed to get file size: " + filepath);
  }
  size_ = st.st_size;

  data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (data_ == MAP_FAILED) {  // NOLINT
    close(fd_);
    throw std::runtime_error("Failed to mmap file: " + filepath);
  }
}

MmapFile::~MmapFile() {
  if (data_ != nullptr) {
    munmap(data_, size_);
  }
  if (fd_ != -1) {
    close(fd_);
  }
}

MmapFile::MmapFile(MmapFile&& from) noexcept
    : fd_(from.fd_), data_(from.data_), size_(from.size_) {
  from.reset();
}

MmapFile& MmapFile::operator=(MmapFile&& from) noexcept {
  if (this == &from) {
    return *this;
  }
  consume(from);
  return *this;
}

void MmapFile::consume(MmapFile& from) {
  fd_ = (from.fd_);
  data_ = (from.data_);
  size_ = (from.size_);
  from.reset();
}

void MmapFile::reset() {
  fd_ = -1;
  data_ = nullptr;
  size_ = 0;
}
}  // namespace whisper
