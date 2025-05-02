{config, ...}: {
  age.secrets.tunnel-creds = {
    file = ../secrets/tunnel-creds.age;
    mode = "770";
    owner = "cloudflared";
    group = "cloudflared";
  };
  age.secrets.tunnel-cert = {
    file = ../secrets/tunnel-cert.age;
    path = "/etc/cloudflared/cert.pem";
    mode = "770";
    owner = "cloudflared";
    group = "cloudflared";
  };
  services.cloudflared = {
    enable = true;
    tunnels = {
      dev-docs = {
        credentialsFile = "${config.age.secrets.tunnel-creds.path}";
        default = "http_status:503";
      };
    };
  };
}
