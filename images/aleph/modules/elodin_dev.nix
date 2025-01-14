let
  dev_user = {
    username,
    sha256,
    lib,
  }: {
    openssh.authorizedKeys.keys =
      lib.splitString "\n"
      (builtins.readFile (builtins.fetchurl {
        url = "https://github.com/${username}.keys";
        inherit sha256;
        #sha256 = "0951ac78df9d1febbbf5a69c8bfbb536dcb1c1174e956a48f2af72402fe246f7";
      }));
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
      inherit lib;
    };
    users.users.akhilles = dev_user {
      username = "akhilles";
      sha256 = "0kkcfsl7jy3j7lm1hj2rvfn21agcna7chqsvxx0kq723z49x0jwi";
      inherit lib;
    };
    users.users.msvandad = dev_user {
      username = "msvandad";
      sha256 = "12r06l5kdbyi38zakhc0dykz9qn0dpwfgs0lrmdlg4ylwq49inak";
      inherit lib;
    };
    users.users.x46085 = dev_user {
      username = "x46085";
      sha256 = "1b79lpdq89dlcb7vva667j5ljbxw4qriqvacxssvv2k0dfy1chxr";
      inherit lib;
    };
    nix.settings.trusted-users = ["root" "@wheel"];
    nix.settings.trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      (builtins.readFile ../dev-public.key)
    ];
  }
