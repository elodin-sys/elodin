let
  aleph-7427 = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIN8prWlP7DynQYV/YZRnW7IxoIHYEHhH2/EgMBSTZ776";
in {
  "tunnel-creds.age".publicKeys = [aleph-7427];
  "tunnel-cert.age".publicKeys = [aleph-7427];
}
