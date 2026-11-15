"""Prepare explicit Photo 3D tilesets for declared regions, without running a simulation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np

ROOT = "https://tile.googleapis.com/v1/3dtiles/root.json"
IDENTITY = np.eye(4)
MAX_RESOURCE_BYTES = 256 * 1024 * 1024
MAX_PACKAGE_BYTES = 4 * 1024**3


def canonical(url):
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.netloc != "tile.googleapis.com"
        or not parts.path.startswith("/v1/3dtiles/")
    ):
        raise ValueError("Only Google Photo 3D resources are supported")
    query = sorted((k, v) for k, v in parse_qsl(parts.query) if k not in {"key", "session"})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def read_json(path):
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Malformed terrain manifest") from error
    if not isinstance(value, dict):
        raise ValueError("Terrain manifest must be an object")
    return value


def validate_package(directory, request):
    """Verify declared coverage and every prepared file before an offline run."""
    manifest = read_json(directory / "prepared-region.json")
    from .terrain import TerrainRegion

    try:
        previous_regions = manifest["regions"]
        if not isinstance(previous_regions, list) or not previous_regions:
            raise ValueError("Missing terrain regions")
        for old in previous_regions:
            TerrainRegion(**old)
        files = manifest["files"]
        if not isinstance(files, dict) or not all(
            isinstance(name, str) and isinstance(digest, str) for name, digest in files.items()
        ):
            raise ValueError("Invalid terrain file manifest")
        if not isinstance(manifest["tiles"], int) or not isinstance(manifest["bytes"], int):
            raise ValueError("Invalid terrain package counts")
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed terrain package manifest") from error
    regions = json.loads(json.dumps(request["regions"]))
    if manifest.get("version") != 1 or manifest.get("asset_id") != request["asset_id"]:
        raise ValueError("Terrain package provider/version mismatch")
    for region in regions:
        w, s, e, n = region["bounds"]
        if not any(
            old["bounds"][0] <= w
            and old["bounds"][1] <= s
            and old["bounds"][2] >= e
            and old["bounds"][3] >= n
            and old["max_error"] <= region["max_error"]
            for old in previous_regions
        ):
            raise ValueError("Prepared terrain does not cover the requested region/detail")
    if "tileset.json" not in files or "package.json" not in files:
        raise ValueError("Terrain package has no verified entrypoint")
    for relative, expected in files.items():
        path = directory / relative
        if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
            raise ValueError("Terrain package reference escapes its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Terrain file changed or is corrupt: {relative}")
    return manifest


def trig_interval(lo, hi, cosine=False):
    if cosine:
        lo += math.pi / 2
        hi += math.pi / 2
    values = [math.sin(lo), math.sin(hi)]
    for k in range(
        math.ceil((lo - math.pi / 2) / math.pi), math.floor((hi - math.pi / 2) / math.pi) + 1
    ):
        values.append(math.sin(math.pi / 2 + k * math.pi))
    return min(values), max(values)


def product(*intervals):
    low, high = 1.0, 1.0
    for a, b in intervals:
        values = [low * a, low * b, high * a, high * b]
        low, high = min(values), max(values)
    return low, high


def region_aabb(bounds, heights=(-12000.0, 100000.0)):
    """Conservative ECEF box: interval arithmetic includes the curved WGS84 surface."""
    west, south, east, north = np.radians(bounds)
    # WGS84 prime-vertical radius is between these limits at every latitude.
    radial = (6378137.0 + heights[0], 6399594.0 + heights[1])
    vertical = (
        6378137.0 * (1 - 0.0066943799901413165) + heights[0],
        6399594.0 * (1 - 0.0066943799901413165) + heights[1],
    )
    lat_cos = trig_interval(south, north, True)
    return np.array(
        [
            product(radial, lat_cos, trig_interval(west, east, True)),
            product(radial, lat_cos, trig_interval(west, east)),
            product(vertical, trig_interval(south, north)),
        ]
    )


def tile_aabb(volume, transform):
    if "region" in volume:
        west, south, east, north, low, high = volume["region"]
        if west > east:  # Antimeridian-spanning coarse parents: do not cull them.
            west, east = -math.pi, math.pi
        # 3D Tiles geographic regions are already Earth-fixed; ignore transform.
        return region_aabb(np.degrees([west, south, east, north]), (low, high))
    if "box" in volume:
        values = np.asarray(volume["box"], dtype=float)
        if values.shape != (12,) or not np.all(np.isfinite(values)):
            raise ValueError("Invalid tile bounding box")
        center = transform[:3, :3] @ values[:3] + transform[:3, 3]
        axes = transform[:3, :3] @ values[3:].reshape((3, 3), order="F")
        radius = np.abs(axes).sum(axis=1)
    elif "sphere" in volume:
        values = np.asarray(volume["sphere"], dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)) or values[3] < 0:
            raise ValueError("Invalid tile bounding sphere")
        center = transform[:3, :3] @ values[:3] + transform[:3, 3]
        radius = np.linalg.norm(transform[:3, :3], axis=1) * values[3]
    else:
        raise ValueError("Unsupported tile bounding volume")
    return np.column_stack((center - radius, center + radius))


def intersects(a, b):
    return bool(np.all(a[:, 0] <= b[:, 1]) and np.all(b[:, 0] <= a[:, 1]))


def intersects_geographic(bounds, geographic):
    """Conservative angular culling, avoiding the long radial extent of ECEF boxes."""
    center = bounds.mean(axis=1)
    radius = float(np.linalg.norm((bounds[:, 1] - bounds[:, 0]) / 2))
    distance = float(np.linalg.norm(center))
    if distance <= radius:
        return True
    alpha = math.asin(min(1.0, radius / distance))
    latitude = math.atan2(center[2], math.hypot(center[0], center[1]))
    longitude = math.atan2(center[1], center[0])
    west, south, east, north = np.radians(geographic)

    def geocentric(lat, height):
        radius = 6378137 / math.sqrt(1 - 0.0066943799901413165 * math.sin(lat) ** 2)
        return math.atan2(
            (radius * (1 - 0.0066943799901413165) + height) * math.sin(lat),
            (radius + height) * math.cos(lat),
        )

    bottom = min(geocentric(south, height) for height in (-12000, 100000))
    top = max(geocentric(north, height) for height in (-12000, 100000))
    if latitude + alpha < bottom or latitude - alpha > top:
        return False
    if abs(latitude) + alpha >= math.pi / 2:
        return True
    span = math.asin(min(1.0, math.sin(alpha) / math.cos(latitude)))
    delta = (longitude - (west + east) / 2 + math.pi) % (2 * math.pi) - math.pi
    return abs(delta) <= (east - west) / 2 + span


def validate_glb(data):
    if len(data) < 20:
        raise ValueError("Truncated terrain GLB")
    magic, version, size, json_size, kind = struct.unpack_from("<4sIIII", data)
    if (
        magic != b"glTF"
        or version != 2
        or size != len(data)
        or kind != 0x4E4F534A
        or json_size > len(data) - 20
    ):
        raise ValueError("Invalid terrain GLB")
    document = json.loads(data[20 : 20 + json_size])
    for group in ("buffers", "images"):
        for item in document.get(group, []):
            if "uri" in item and not item["uri"].startswith("data:"):
                raise ValueError("Terrain GLB has an external dependency")


class GoogleSource:
    def __init__(self, token, cache, existing=None):
        self.cache = cache
        self.existing = existing
        self.existing_resources = {}
        self.existing_hashes = {}
        if existing is not None and (existing / "prepared-region.json").is_file():
            try:
                self.existing_resources = read_json(existing / "package.json")["resources"]
                self.existing_hashes = read_json(existing / "prepared-region.json")["files"]
                if not isinstance(self.existing_resources, dict) or not isinstance(
                    self.existing_hashes, dict
                ):
                    raise ValueError("Invalid terrain package index")
            except (ValueError, OSError, KeyError):
                # A damaged package can be repaired from the provider/cache.
                self.existing_resources = {}
                self.existing_hashes = {}
        self.credentials = {}
        self.downloads = 0
        self.hits = 0
        self.bytes = 0
        self.documents = {}
        endpoint = "https://api.cesium.com/v1/assets/2275207/endpoint"
        grant = json.loads(
            self.download(Request(endpoint, headers={"Authorization": f"Bearer {token}"}))
        )
        if grant.get("externalType") != "3DTILES":
            raise ValueError("Expected an external Photo 3D ion asset")
        url = grant.get("options", {}).get("url", grant.get("url", ""))
        if canonical(url) != ROOT:
            raise ValueError("Unexpected Photo 3D entrypoint")
        self.credentials = dict(parse_qsl(urlsplit(url).query))
        self.attributions = grant.get("attributions", [])
        self.cache.mkdir(parents=True, exist_ok=True)

    def download(self, request):
        try:
            with urlopen(request, timeout=60) as response:
                data = response.read(MAX_RESOURCE_BYTES + 1)
        except HTTPError as error:
            raise ValueError(
                f"Terrain download failed (HTTP {error.code}); rerun preparation to retry"
            ) from None
        except (URLError, TimeoutError):
            raise ValueError(
                "Terrain download failed or timed out; rerun preparation to retry"
            ) from None
        if len(data) > MAX_RESOURCE_BYTES:
            raise ValueError("Terrain resource exceeds the preparation size limit")
        self.bytes += len(data)
        if self.bytes > MAX_PACKAGE_BYTES:
            raise ValueError("Terrain preparation exceeded 4 GiB; reduce region/detail")
        self.downloads += 1
        return data

    def get(self, url):
        key = canonical(url)
        if key in self.documents:
            return self.documents[key]
        relative = self.existing_resources.get(key)
        if self.existing is not None and isinstance(relative, str) and relative.endswith(".glb"):
            path = self.existing / relative
            if path.resolve().is_relative_to(self.existing.resolve()) and path.is_file():
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() == self.existing_hashes.get(relative):
                    validate_glb(data)
                    self.hits += 1
                    return data
        cache_path = self.cache / hashlib.sha256(key.encode()).hexdigest()
        # Refresh hierarchy documents to obtain a current session and discover changes.
        if cache_path.exists() and not urlsplit(key).path.endswith(".json"):
            self.hits += 1
            return cache_path.read_bytes()
        parts = urlsplit(url)
        query = {**dict(parse_qsl(parts.query)), **self.credentials}
        authorized = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
        data = self.download(Request(authorized))
        if not data.startswith(b"glTF"):
            # Discover the session from the fresh root before fetching its descendants.
            document = json.loads(data)

            def sessions(value):
                if isinstance(value, dict):
                    for name, child in value.items():
                        if name in {"uri", "url"} and isinstance(child, str):
                            session = dict(parse_qsl(urlsplit(child).query)).get("session")
                            if session:
                                self.credentials["session"] = session
                                return True
                        if sessions(child):
                            return True
                elif isinstance(value, list):
                    return any(sessions(child) for child in value)
                return False

            if key == ROOT:
                sessions(document)
            self.documents[key] = data
        else:
            validate_glb(data)
            temporary = cache_path.with_suffix(".part")
            temporary.write_bytes(data)
            temporary.replace(cache_path)
        return data


class RegionPackage:
    def __init__(self, source, directory, regions):
        self.source = source
        self.directory = directory
        self.regions = [
            (region_aabb(region["bounds"]), region["bounds"], region["max_error"])
            for region in regions
        ]
        self.resources = {}
        self.files = {}
        self.visiting = set()
        self.nodes = 0
        self.tiles = 0
        self.coarser_leaf_tiles = 0
        self.largest_leaf_error = 0.0
        self.byte_count = 0

    def write(self, name, data):
        if any(
            value.encode() in data
            for value in getattr(self.source, "credentials", {}).values()
            if len(value) >= 8
        ):
            raise ValueError("Provider credentials remain in a prepared resource")
        self.byte_count += len(data)
        if self.byte_count > MAX_PACKAGE_BYTES:
            raise ValueError("Prepared terrain exceeds 4 GiB; reduce region/detail")
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files[name] = hashlib.sha256(data).hexdigest()

    def json(self, name, value):
        self.write(name, json.dumps(value, separators=(",", ":"), allow_nan=False).encode())

    def source_node(self, node, base):
        for content in [node["content"]] if "content" in node else node.get("contents", []):
            uri = content.pop("url", content.get("uri"))
            content["uri"] = canonical(urljoin(base, uri))
        for child in node.get("children", []):
            self.source_node(child, base)

    def resource(self, url, transform, depth=0):
        key = canonical(url)
        if key in self.visiting or depth > 64:
            raise ValueError("Cyclic or excessively deep terrain hierarchy")
        if len(self.resources) >= 100000:
            raise ValueError("Terrain preparation has too many resources")
        # Different inherited transforms must be traversed independently, even if the URI is shared.
        data = self.source.get(url)
        digest = hashlib.sha256(key.encode()).hexdigest()
        if data.startswith(b"glTF"):
            validate_glb(data)
            name = f"tiles/{digest}.glb"
            if name not in self.files:
                self.write(name, data)
                self.tiles += 1
                if self.tiles % 100 == 0:
                    print(
                        f"Preparing terrain: {self.tiles} tiles, {self.byte_count / 1e6:.1f} MB",
                        flush=True,
                    )
            self.resources[key] = name
            return name, False
        document = json.loads(data)
        if not isinstance(document, dict) or "root" not in document:
            raise ValueError("Expected an explicit 3D Tiles tileset")
        self.visiting.add(key)
        try:
            original = copy.deepcopy(document)
            self.source_node(original["root"], key)
            source_name = f"source/{digest}.json"
            self.json(source_name, original)
            self.resources[key] = source_name
            document["root"] = self.node(document["root"], url, transform, depth + 1)
            local_digest = hashlib.sha256(key.encode() + transform.tobytes()).hexdigest()
            name = "tileset.json" if key == ROOT else f"tiles/{local_digest}.json"
            self.json(name, document)
            return name, True
        finally:
            self.visiting.remove(key)

    def node(self, node, base, parent, depth):
        self.nodes += 1
        if self.nodes > 500000 or depth > 64:
            raise ValueError("Terrain preparation hierarchy exceeds its traversal limit")
        if "implicitTiling" in node or node.get("extensions"):
            raise ValueError("Terrain preparation requires explicit tiles without tile extensions")
        transform = np.asarray(node.get("transform", IDENTITY.flatten(order="F")), dtype=float)
        if transform.shape != (16,) or not np.all(np.isfinite(transform)):
            raise ValueError("Invalid tile transform")
        transform = parent @ transform.reshape((4, 4), order="F")
        bounds = tile_aabb(node["boundingVolume"], transform)
        matching = [
            error
            for region, geographic, error in self.regions
            if intersects(bounds, region) and intersects_geographic(bounds, geographic)
        ]
        if not matching:
            # An empty outside node retains bounds/refinement without a network reference.
            node.pop("content", None)
            node.pop("contents", None)
            node.pop("children", None)
            return node
        error = float(node["geometricError"]) * np.linalg.norm(transform[:3, :3], ord=2)
        if not math.isfinite(error) or error < 0:
            raise ValueError("Invalid terrain geometric error")
        contents = [node["content"]] if "content" in node else node.get("contents", [])
        has_geometry = False
        has_external = False
        for content in contents:
            uri = content.pop("url", content.get("uri"))
            target, external = self.resource(urljoin(base, uri), transform, depth)
            content["uri"] = ("../" if canonical(base) != ROOT else "") + target
            has_geometry |= not external
            has_external |= external
        if has_geometry and error <= min(matching):
            node.pop("children", None)
        else:
            if (
                has_geometry
                and not has_external
                and not node.get("children")
                and error > min(matching)
            ):
                # A provider leaf has no finer geometry to fetch, even online.
                # Keep its best available content and report this limit explicitly.
                self.coarser_leaf_tiles += 1
                self.largest_leaf_error = max(self.largest_leaf_error, error)
            node["children"] = [
                self.node(child, base, transform, depth + 1) for child in node.get("children", [])
            ]
        return node


def prepare(request, source):
    from .terrain import TerrainRegion

    if (
        request.get("version") != 1
        or request.get("asset_id") != 2275207
        or not request.get("regions")
    ):
        raise ValueError("Declare Photo 3D terrain regions before preparation")
    for region in request["regions"]:
        TerrainRegion(**region)
    output = Path(request["output"])
    if output.is_symlink():
        raise ValueError("Terrain output must not be a symbolic link")
    if output.exists():
        try:
            result = validate_package(output, request)
            print(
                f"Terrain ready: {result['tiles']} tiles, {result['bytes'] / 1e6:.1f} MB (reused)",
                flush=True,
            )
            return result
        except (ValueError, FileNotFoundError):
            if not (output / "prepared-region.json").is_file():
                raise ValueError(
                    "Refusing to replace a directory not created by terrain prepare"
                ) from None
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.prepare.lock")
    with lock.open("x"):
        pass
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    backup = output.with_name(f".{output.name}.previous")
    try:
        package = RegionPackage(source, stage, request["regions"])
        package.resource(ROOT, IDENTITY)
        if package.tiles == 0:
            raise ValueError("No Photo 3D geometry was found for the declared regions")
        package.json(
            "package.json",
            {"version": 1, "ion_asset_id": 2275207, "root": ROOT, "resources": package.resources},
        )
        package.json("attribution.json", source.attributions)
        if "basemap" in request:
            package.write("earth-lowres.glb", Path(request["basemap"]).read_bytes())
        report = {
            "version": 1,
            "asset_id": 2275207,
            "regions": request["regions"],
            "tiles": package.tiles,
            "coarser_leaf_tiles": package.coarser_leaf_tiles,
            "largest_leaf_error": package.largest_leaf_error,
            "bytes": package.byte_count,
            "files": package.files,
        }
        (stage / "prepared-region.json").write_text(json.dumps(report, indent=2) + "\n")
        for path in [
            stage / name
            for name in package.files
            if name.endswith(".json")
            and not name.startswith("source/")
            and name not in {"package.json", "attribution.json"}
        ]:

            def check_references(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key.lower() in {"uri", "url", "schemauri"} and isinstance(item, str):
                            parts = urlsplit(item)
                            target = (path.parent / parts.path).resolve()
                            if (
                                parts.scheme
                                or parts.netloc
                                or parts.query
                                or parts.fragment
                                or not target.is_relative_to(stage.resolve())
                                or not target.is_file()
                            ):
                                raise ValueError(
                                    "A prepared terrain resource is missing or external"
                                )
                        check_references(item)
                elif isinstance(value, list):
                    for item in value:
                        check_references(item)

            check_references(read_json(path))
        validate_package(stage, request)
        if backup.exists():
            raise ValueError("Previous terrain backup exists; inspect it before preparing again")
        if output.exists():
            old = read_json(output / "prepared-region.json")
            allowed = set(old["files"]) | {"prepared-region.json"}
            if any(
                str(path.relative_to(output)) not in allowed
                for path in output.rglob("*")
                if path.is_file()
            ):
                raise ValueError("Terrain directory contains files not owned by preparation")
            output.rename(backup)
        try:
            stage.rename(output)
        except OSError:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        print(
            f"Terrain ready: {package.tiles} tiles, {package.byte_count / 1e6:.1f} MB", flush=True
        )
        if package.coarser_leaf_tiles:
            print(
                f"Provider detail limit: {package.coarser_leaf_tiles} leaf tiles are coarser than the requested target (largest geometric error {package.largest_leaf_error:.1f} m)",
                flush=True,
            )
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        lock.unlink(missing_ok=True)


def main():
    request = read_json(Path(sys.argv[1]))
    # A valid package is usable without any provider connection or credentials.
    try:
        report = validate_package(Path(request["output"]), request)
    except (FileNotFoundError, ValueError):
        cache = (
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "elodin/terrain-photo3d"
        )
        source = GoogleSource(os.environ["ELODIN_CESIUM_TOKEN"], cache, Path(request["output"]))
        prepare(request, source)
    else:
        print(f"Terrain ready: {report['tiles']} tiles, {report['bytes'] / 1e6:.1f} MB (reused)")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(f"Terrain preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
