{
  pkgs,
  rustToolchain,
  lib,
  ...
}: let
  pname = "udp-component-broadcast";
  workspaceToml = builtins.fromTOML (builtins.readFile ../../Cargo.toml);
  version = workspaceToml.workspace.package.version;

  # The scripts use the first-class `elodin.db` client shipped in the elodin
  # wheel (jax is forced CPU by the aleph overlay, as in elodin-cli.nix).
  elodinPy = pkgs.callPackage ../../nix/pkgs/elodin-py.nix {
    inherit rustToolchain;
    pythonPackages = pkgs.python313Packages;
    python = pkgs.python313;
  };
  python = elodinPy.python;

  # Python environment with all dependencies
  pythonEnv = python.withPackages (ps: [
    ps.protobuf
    ps.numpy
    ps.netifaces
    elodinPy.py
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
