{pkgs, ...}: {
  environment.systemPackages = with pkgs; [aleph-setup];
  environment.interactiveShellInit = ''
     if [ "$(id -u)" -eq 0 ] && [ ! -f ".aleph-setup" ]; then
       aleph-setup
       touch /root/.aleph-setup
    fi;
  '';
}
