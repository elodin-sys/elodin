#pragma once
#include "rust/cxx.h"
#include <memory>

namespace elodin_cesium {
struct View;
struct Frame;
class Session {
public:
  explicit Session(rust::Str url);
  Session(int64_t asset, rust::Str token);
  ~Session();
  Frame update(rust::Slice<const View> views, float seconds);
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
std::unique_ptr<Session> open_url(rust::Str url);
std::unique_ptr<Session> open_ion(int64_t asset, rust::Str token);
}
