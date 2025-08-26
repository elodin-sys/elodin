{pkgs ? import <nixpkgs> {}}:
pkgs.callPackage ./nix/pkgs/xla-ext.nix {}
