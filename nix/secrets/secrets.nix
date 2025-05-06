let
  aleph-7427 = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBqXJooZYJIyxkpvpnHMaKNmWVE3iqNZnAKUsVoRKkCR";
  akhilles = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOh7/655LhcognTvqjkXNuP7HIS27IVeMhJUfp9oRd+o";
in {
  "tunnel-creds.age".publicKeys = [aleph-7427 akhilles];
  "tunnel-cert.age".publicKeys = [aleph-7427 akhilles];
}
