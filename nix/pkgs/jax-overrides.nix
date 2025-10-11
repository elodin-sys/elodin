# Shared JAX/jaxlib version overrides for consistency across the project
# These overrides ensure we use JAX 0.4.31 everywhere to match our pyproject.toml
{pkgs}:
self: super: {
  # Override jaxlib to use version 0.4.31
  jaxlib = super.jaxlib-bin.overridePythonAttrs (old: rec {
    version = "0.4.31";
    src = let
      system = pkgs.stdenv.system;
      platform =
        if system == "x86_64-linux"
        then "manylinux2014_x86_64"
        else if system == "aarch64-linux"
        then "manylinux2014_aarch64"
        else "macosx_11_0_arm64";
      wheelName = "jaxlib-${version}-cp312-cp312-${platform}.whl";
      baseUrl = "https://files.pythonhosted.org/packages";
      # These are the specific paths for each platform's wheel
      wheelUrls = {
        "manylinux2014_x86_64" = "${baseUrl}/b1/09/58d35465d48c8bee1d9a4e7a3c5db2edaabfc7ac94f4576c9f8c51b83e70/${wheelName}";
        "manylinux2014_aarch64" = "${baseUrl}/e0/af/10b49f8de2acc7abc871478823579d7241be52ca0d6bb0d2b2c476cc1b68/${wheelName}";
        "macosx_11_0_arm64" = "${baseUrl}/68/cf/28895a4a89d88d18592507d7a35218b6bb2d8bced13615065c9f925f2ae1/${wheelName}";
      };
    in
      pkgs.fetchurl {
        url = wheelUrls.${platform} or (throw "Unsupported platform: ${platform}");
        hash =
          if system == "x86_64-linux"
          then "sha256-Hxr6X9WKYPZ/DKWG4mcUrs5i6qLIM0wk0OgoWvxKfM0="
          else if system == "aarch64-linux"
          then "sha256-TYZ6GgVlsxz9qrvsgeAwLGRhuyrEuSwEZwMo15WBmAM="
          else "sha256-yficGFKH5A7oFzpxQtZJUxHncs0TmpPcqT8NmcGHKDI="; # macosx_11_0_arm64
      };
  });
  
  # Make jaxlib-bin point to our overridden jaxlib
  jaxlib-bin = self.jaxlib;

  # Override JAX to use version 0.4.31
  jax = super.jax.overridePythonAttrs (old: rec {
    version = "0.4.31";
    src = super.fetchPypi {
      inherit (old) pname;
      inherit version;
      hash = "sha256-/S1HBkOgBz2CJzfweI9xORZWr35izFsueZXuOQzqwoc=";
    };
    # Skip version check during build
    pythonImportsCheck = [];
    doCheck = false;
  });
}
