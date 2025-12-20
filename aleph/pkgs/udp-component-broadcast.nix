{
  pkgs,
  rustToolchain,
  lib,
  ...
}: let
  pname = "udp-component-broadcast";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;
  # Convert version format from 0.15.0-alpha.1 to 0.15.0a1 for wheel filename (PEP 440)
  wheelVersion = lib.strings.replaceStrings ["-alpha."] ["a"] version;
  wheelName = "impeller_py";
  wheelSuffix = "cp312-cp312-linux_aarch64";

  common = import ./common.nix {inherit lib;};
  src = common.src;

  python = pkgs.python312;

  # Build impeller_py wheel using maturin
  # Note: impeller_py is part of the main workspace
  impellerPyWheel = pkgs.rustPlatform.buildRustPackage {
    pname = "impeller_py";
    inherit version src;

    cargoLock = {
      lockFile = ../../Cargo.lock;
      allowBuiltinFetchGit = true;
    };

    buildAndTestSubdir = "fsw/udp_component_broadcast/impeller_py";

    nativeBuildInputs = with pkgs; [
      (rustToolchain pkgs)
      maturin
      python
      which
    ];

    buildInputs = [
      python
    ];

    doCheck = false;

    # Override the build phase to use maturin
    buildPhase = ''
      runHook preBuild

      # Build the wheel with maturin
      maturin build --offline --target-dir ./target -m fsw/udp_component_broadcast/impeller_py/Cargo.toml --release

      runHook postBuild
    '';

    # Install the wheel
    installPhase = ''
      runHook preInstall

      mkdir -p $out
      cp target/wheels/${wheelName}-${wheelVersion}-${wheelSuffix}.whl $out/

      runHook postInstall
    '';
  };

  # Install impeller_py wheel as a Python package
  impellerPy = python.pkgs.buildPythonPackage {
    pname = wheelName;
    inherit version;
    format = "wheel";
    src = "${impellerPyWheel}/${wheelName}-${wheelVersion}-${wheelSuffix}.whl";
    doCheck = false;
  };

  # Python environment with all dependencies
  pythonEnv = python.withPackages (ps: [
    ps.protobuf
    ps.numpy
    ps.netifaces
    impellerPy
  ]);

  # Package containing both scripts and wrapper binaries
  pkg = pkgs.stdenv.mkDerivation {
    inherit pname version;

    src = ../../fsw/udp_component_broadcast;

    dontBuild = true;

    installPhase = ''
      runHook preInstall

      # Create directory structure
      mkdir -p $out/bin
      mkdir -p $out/lib/udp-component-broadcast

      # Copy Python files
      cp broadcast_component.py $out/lib/udp-component-broadcast/
      cp receive_broadcast.py $out/lib/udp-component-broadcast/
      cp component_broadcast_pb2.py $out/lib/udp-component-broadcast/

      # Create wrapper script for broadcaster
      cat > $out/bin/udp-broadcast <<EOF
      #!${pkgs.stdenv.shell}
      export PYTHONPATH=$out/lib/udp-component-broadcast:${pythonEnv}/${python.sitePackages}
      exec ${pythonEnv}/bin/python $out/lib/udp-component-broadcast/broadcast_component.py "\$@"
      EOF
      chmod +x $out/bin/udp-broadcast

      # Create wrapper script for receiver
      cat > $out/bin/udp-receive <<EOF
      #!${pkgs.stdenv.shell}
      export PYTHONPATH=$out/lib/udp-component-broadcast:${pythonEnv}/${python.sitePackages}
      exec ${pythonEnv}/bin/python $out/lib/udp-component-broadcast/receive_broadcast.py "\$@"
      EOF
      chmod +x $out/bin/udp-receive

      runHook postInstall
    '';

    meta = with lib; {
      description = "UDP Component Broadcast for Elodin-DB - broadcasts component data between flight computers";
      homepage = "https://github.com/elodin-sys/elodin";
      license = licenses.mit;
      maintainers = [];
      mainProgram = "udp-broadcast";
    };
  };
in
  pkg
