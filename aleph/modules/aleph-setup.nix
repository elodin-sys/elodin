{
  config,
  lib,
  pkgs,
  ...
}: {
  config = lib.mkIf (!config.aleph.sd.enable) {
    environment.systemPackages = with pkgs; [aleph-setup];
    environment.interactiveShellInit = ''
      if [ "$(id -u)" -eq 0 ] && [ ! -f "/root/.aleph-setup" ]; then
        aleph-setup
        touch /root/.aleph-setup
      fi;
    '';
  };
}
