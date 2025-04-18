{lib, ...}: {
  # To add a user:
  # - Uncomment the following lines
  # - Replace <username> with your desired username
  # - Replace <ssh-pub-key> with your desired public SSH key

  # users.users.<username> = {
  #   isNormalUser = true;
  #   extraGroups = ["wheel" "video" "dialout"];
  #   openssh.authorizedKeys.keys = [
  #     "<ssh-pub-key>"
  #   ];
  # };
}
