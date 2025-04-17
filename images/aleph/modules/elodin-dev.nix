let
  dev_user = {
    lib,
    username,
    sha256,
  }: {
    openssh.authorizedKeys.keys =
      lib.splitString "\n"
      (builtins.readFile (builtins.fetchurl {
        url = "https://github.com/${username}.keys";
        inherit sha256;
        #sha256 = "0951ac78df9d1febbbf5a69c8bfbb536dcb1c1174e956a48f2af72402fe246f7";
      }));
    isNormalUser = true;
    extraGroups = ["wheel" "video" "dialout"];
    initialPassword = "nixos";
  };
in
  {lib, ...}: {
    users.users.sphw = dev_user {
      username = "sphw";
      sha256 = "0951ac78df9d1febbbf5a69c8bfbb536dcb1c1174e956a48f2af72402fe246f7";
      inherit lib;
    };
    users.users.akhilles = dev_user {
      username = "akhilles";
      sha256 = "0r6bjy033a94kqr5pzzckfxxly4n2zdx6jdds0y04dvppdisqw38";
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
  }
