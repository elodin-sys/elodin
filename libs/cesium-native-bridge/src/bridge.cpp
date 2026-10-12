#include "cesium-native-bridge/src/lib.rs.h"
#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <Cesium3DTilesSelection/ViewState.h>
#include <Cesium3DTilesSelection/ViewUpdateResult.h>
#include <CesiumAsync/ITaskProcessor.h>
#include <CesiumCurl/CurlAssetAccessor.h>
#include <CesiumGltf/AccessorUtility.h>
#include <CesiumGltfContent/GltfUtilities.h>
#include <CesiumUtility/CreditSystem.h>
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/sinks/null_sink.h>
#include <condition_variable>
#include <atomic>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>

namespace elodin_cesium {
namespace {
using namespace Cesium3DTilesSelection;

// Bounded worker count; destruction drains work and joins every worker.
class Workers final : public CesiumAsync::ITaskProcessor {
  std::mutex mutex;
  std::condition_variable wake;
  std::deque<std::function<void()>> jobs;
  bool stopping = false;
  std::vector<std::thread> threads;
public:
  Workers() {
    for (int i = 0; i < 4; ++i) threads.emplace_back([this] {
      for (;;) {
        std::function<void()> job;
        {
          std::unique_lock lock(mutex);
          wake.wait(lock, [this] { return stopping || !jobs.empty(); });
          if (jobs.empty() && stopping) return;
          job = std::move(jobs.front()); jobs.pop_front();
        }
        job();
      }
    });
  }
  ~Workers() override {
    { std::lock_guard lock(mutex); stopping = true; }
    wake.notify_all();
    for (auto& thread : threads) thread.join();
  }
  void startTask(std::function<void()> task) override {
    { std::lock_guard lock(mutex); jobs.push_back(std::move(task)); }
    wake.notify_one();
  }
};

Tile extract(CesiumGltf::Model& model, const glm::dmat4& tileTransform) {
  Tile output{};
  model.generateMissingNormalsSmooth();
  const auto root = CesiumGltfContent::GltfUtilities::applyGltfUpAxisTransform(
      model, CesiumGltfContent::GltfUtilities::applyRtcCenter(model, tileTransform));
  model.forEachPrimitiveInScene(-1, [&](const CesiumGltf::Model& gltf,
      const CesiumGltf::Node&, const CesiumGltf::Mesh&,
      const CesiumGltf::MeshPrimitive& primitive, const glm::dmat4& node) {
    if (primitive.mode != 4) return; // First prototype: triangle surfaces.
    const auto positions = CesiumGltf::getPositionAccessorView(gltf, primitive);
    const auto normals = CesiumGltf::getNormalAccessorView(gltf, primitive);
    const auto uvs = CesiumGltf::getTexCoordAccessorView(gltf, primitive, 0);
    const auto count = std::visit(CesiumGltf::CountFromAccessor{}, positions);
    if (count <= 0) return;
    Primitive p{};
    auto matrix = root * node;
    std::copy_n(glm::value_ptr(matrix), 16, p.transform.begin());
    p.base_color = {1.f, 1.f, 1.f, 1.f};
    for (int64_t i = 0; i < count; ++i) {
      const auto position = std::visit(CesiumGltf::PositionFromAccessor{i}, positions);
      if (!position) return;
      const auto normal = std::visit(CesiumGltf::NormalFromAccessor{i}, normals);
      const auto uv = std::visit(CesiumGltf::TexCoordFromAccessor{i}, uvs);
      for (int axis = 0; axis < 3; ++axis) {
        p.positions.push_back(float((*position)[axis]));
        p.normals.push_back(normal ? float((*normal)[axis]) : float(axis == 1));
      }
      p.uvs.push_back(uv ? float(uv->x) : 0.f);
      p.uvs.push_back(uv ? float(uv->y) : 0.f);
    }
    if (primitive.indices >= 0) {
      auto indices = CesiumGltf::getIndexAccessorView(gltf, primitive);
      auto n = std::visit(CesiumGltf::CountFromAccessor{}, indices);
      for (int64_t i = 0; i < n; ++i) {
        auto index = std::visit(CesiumGltf::IndexFromAccessor{i}, indices);
        if (index < 0 || index >= count) return;
        p.indices.push_back(uint32_t(index));
      }
    } else {
      for (int64_t i = 0; i < count; ++i) p.indices.push_back(uint32_t(i));
    }
    if (p.indices.size() % 3 != 0) return;
    if (primitive.material >= 0 && size_t(primitive.material) < gltf.materials.size()) {
      const auto& material = gltf.materials[primitive.material];
      p.double_sided = material.doubleSided;
      if (material.pbrMetallicRoughness) {
        const auto& pbr = *material.pbrMetallicRoughness;
        if (pbr.baseColorFactor.size() == 4)
          for (int i = 0; i < 4; ++i) p.base_color[i] = float(pbr.baseColorFactor[i]);
        if (pbr.baseColorTexture && pbr.baseColorTexture->index >= 0 &&
            size_t(pbr.baseColorTexture->index) < gltf.textures.size()) {
          auto source = gltf.textures[pbr.baseColorTexture->index].source;
          if (source >= 0 && size_t(source) < gltf.images.size()) {
            auto image = gltf.images[source].pAsset;
            if (image && image->width > 0 && image->height > 0 &&
                image->bytesPerChannel == 1 && image->channels >= 1 && image->channels <= 4 &&
                image->compressedPixelFormat == CesiumImage::GpuCompressedPixelFormat::NONE) {
              auto pixels = size_t(image->width) * size_t(image->height);
              if (image->pixelData.size() >= pixels * image->channels) {
                p.width = image->width; p.height = image->height;
                p.pixels.reserve(pixels * 4);
                for (size_t i = 0; i < pixels; ++i) {
                  auto read = [&](int c) { return uint8_t(image->pixelData[i * image->channels + c]); };
                  bool gray = image->channels < 3;
                  p.pixels.push_back(read(0));
                  p.pixels.push_back(read(gray ? 0 : 1));
                  p.pixels.push_back(read(gray ? 0 : 2));
                  p.pixels.push_back(image->channels == 2 ? read(1) : image->channels == 4 ? read(3) : 255);
                }
              }
            }
          }
        }
      }
    }
    output.primitives.push_back(std::move(p));
  });
  return output;
}

class Resources final : public IPrepareRendererResources {
public:
  uint64_t nextId = 1;
  rust::Vec<Tile> loaded;
  rust::Vec<uint64_t> removed;
  CesiumAsync::Future<TileLoadResultAndRenderResources> prepareInLoadThread(
      const CesiumAsync::AsyncSystem& async, TileLoadResult&& result,
      const glm::dmat4& transform, const std::any&) override {
    auto packet = std::make_unique<Tile>();
    if (auto* model = std::get_if<CesiumGltf::Model>(&result.contentKind))
      *packet = extract(*model, transform);
    return async.createResolvedFuture(TileLoadResultAndRenderResources{std::move(result), packet.release()});
  }
  void* prepareInMainThread(Cesium3DTilesSelection::Tile&, void* data) override {
    std::unique_ptr<Tile> packet(static_cast<Tile*>(data));
    auto id = nextId++;
    packet->id = id;
    loaded.push_back(std::move(*packet));
    return new uint64_t(id);
  }
  void free(Cesium3DTilesSelection::Tile&, void* loading, void* ready) noexcept override {
    delete static_cast<Tile*>(loading);
    if (ready) {
      auto* id = static_cast<uint64_t*>(ready);
      removed.push_back(*id);
      delete id;
    }
  }
  void* prepareRasterInLoadThread(CesiumImage::ImageAsset&, const std::any&) override { return nullptr; }
  void* prepareRasterInMainThread(CesiumRasterOverlays::RasterOverlayTile&, void*) override { return nullptr; }
  void freeRaster(const CesiumRasterOverlays::RasterOverlayTile&, void*, void*) noexcept override {}
  void attachRasterInMainThread(const Cesium3DTilesSelection::Tile&, int32_t,
      const CesiumRasterOverlays::RasterOverlayTile&, void*, const glm::dvec2&, const glm::dvec2&) override {}
  void detachRasterInMainThread(const Cesium3DTilesSelection::Tile&, int32_t,
      const CesiumRasterOverlays::RasterOverlayTile&, void*) noexcept override {}
};
}

struct Session::Impl {
  std::shared_ptr<Workers> workers = std::make_shared<Workers>();
  CesiumAsync::AsyncSystem async{workers};
  static std::shared_ptr<CesiumCurl::CurlAssetAccessor> createAccessor() {
    CesiumCurl::CurlAssetAccessorOptions options;
    if (auto* file = std::getenv("SSL_CERT_FILE")) options.certificateFile = file;
    else if (auto* file = std::getenv("NIX_SSL_CERT_FILE")) options.certificateFile = file;
    return std::make_shared<CesiumCurl::CurlAssetAccessor>(options);
  }
  std::shared_ptr<CesiumCurl::CurlAssetAccessor> accessor = createAccessor();
  std::shared_ptr<Resources> resources = std::make_shared<Resources>();
  std::shared_ptr<CesiumUtility::CreditSystem> credits = std::make_shared<CesiumUtility::CreditSystem>();
  std::unique_ptr<Tileset> tileset;
  std::shared_ptr<std::atomic<bool>> loadFailed = std::make_shared<std::atomic<bool>>(false);
  TilesetOptions options() {
    TilesetOptions options;
    auto failed = loadFailed;
    options.loadErrorCallback = [failed](const auto&) { failed->store(true); };
    return options;
  }
  TilesetExternals externals() {
    // Native errors may contain signed asset URLs. Never forward those to logs.
    auto logger = std::make_shared<spdlog::logger>("cesium", std::make_shared<spdlog::sinks::null_sink_mt>());
    return {accessor, resources, async, credits, logger};
  }
  ~Impl() {
    if (!tileset) return;
    auto finished = tileset->getAsyncDestructionCompleteEvent();
    tileset.reset();
    while (!finished.isReady()) {
      accessor->tick(); async.dispatchMainThreadTasks();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
};
Session::Session(rust::Str url) : impl(std::make_unique<Impl>()) {
  Cesium3DTilesContent::registerAllTileContentTypes();
  impl->tileset = std::make_unique<Tileset>(impl->externals(), std::string(url), impl->options());
}
Session::Session(int64_t asset, rust::Str token) : impl(std::make_unique<Impl>()) {
  Cesium3DTilesContent::registerAllTileContentTypes();
  impl->tileset = std::make_unique<Tileset>(impl->externals(), asset, std::string(token), impl->options());
}
Session::~Session() = default;
std::unique_ptr<Session> open_url(rust::Str url) { return std::make_unique<Session>(url); }
std::unique_ptr<Session> open_ion(int64_t asset, rust::Str token) { return std::make_unique<Session>(asset, token); }
Frame Session::update(rust::Slice<const View> input, float seconds) {
  std::vector<ViewState> views;
  for (const auto& view : input) {
    auto vec = [](const std::array<double, 3>& a) { return glm::dvec3(a[0], a[1], a[2]); };
    views.emplace_back(vec(view.position), vec(view.direction), vec(view.up),
        glm::dvec2(view.width, view.height), view.horizontal_fov, view.vertical_fov);
  }
  impl->accessor->tick();
  impl->async.dispatchMainThreadTasks();
  auto& tileset = *impl->tileset;
  const auto& result = tileset.updateViewGroup(tileset.getDefaultViewGroup(), views, seconds);
  tileset.loadTiles();
  Frame frame{};
  frame.loaded = std::move(impl->resources->loaded);
  frame.removed = std::move(impl->resources->removed);
  for (const auto& tile : result.tilesToRenderThisFrame) {
    auto* content = tile->getContent().getRenderContent();
    if (content && content->getRenderResources())
      frame.visible.push_back(*static_cast<const uint64_t*>(content->getRenderResources()));
  }
  for (auto credit : impl->credits->getSnapshot().currentCredits)
    frame.credits.push_back(impl->credits->getHtml(credit));
  frame.pending = result.workerThreadTileLoadQueueLength + result.mainThreadTileLoadQueueLength;
  frame.load_failed = impl->loadFailed->load();
  return frame;
}
}
