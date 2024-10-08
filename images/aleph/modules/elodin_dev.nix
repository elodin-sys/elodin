let
  dev_user = {
    username,
    sha256,
  }: {
    openssh.authorizedKeys.keys = [
      (builtins.readFile (builtins.fetchurl {
        url = "https://github.com/${username}.keys";
        inherit sha256;
        #sha256 = "0951ac78df9d1febbbf5a69c8bfbb536dcb1c1174e956a48f2af72402fe246f7";
      }))
    ];
    isNormalUser = true;
    extraGroups = ["wheel"];
    initialPassword = "nixos";
  };
in
  {lib, ...}: {
    networking.wireless.networks = {
      elodin = {
        psk = "kvothe123";
      };
      aoraki = {
        psk = "idclimbthat";
      };
    };
    networking.wireless.enable = true;
    networking.dhcpcd.enable = true;
    users.users.sphw = dev_user {
      username = "sphw";
      sha256 = "0951ac78df9d1febbbf5a69c8bfbb536dcb1c1174e956a48f2af72402fe246f7";
    };
    nix.settings.trusted-users = ["root" "@wheel"];
    nix.settings.trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      (builtins.readFile ../dev-public.key)
    ];
  }
