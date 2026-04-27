{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.elodin-py;
  pythonWithElodin = pkgs.python313.withPackages (_: [pkgs.elodin.elodin-py.py]);
  elodinPython = pkgs.writeShellScriptBin "elodin-python" ''
    export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16384
    exec ${lib.getExe pythonWithElodin} "$@"
  '';
in {
  options.services.elodin-py = {
    enable = lib.mkEnableOption ''
      Elodin Python SDK: Python 3.13 with the `elodin` package (nox-py extension + JAX stack).
      Installs `python` and `elodin-python` with glibc static-TLS tuning for the extension.
    '';
    editor = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Also install the Elodin editor CLI (`elodin` binary, Bevy + GPU stack).
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages =
      [
        pythonWithElodin
        elodinPython
        (pkgs.writeShellScriptBin "python" ''
          export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=16384
          exec ${lib.getExe pythonWithElodin} "$@"
        '')
      ]
      ++ lib.optional cfg.editor pkgs.elodin.elodin-cli;
  };
}
