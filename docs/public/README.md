# Public Docs

Elodin Documentation is statically built using:    

[Zola](https://www.getzola.org/): Static site generator and server
[Tera](https://keats.github.io/tera/): Templating engine for Zola
[AdiDoks](https://adidoks.org/): This was initially built on top of AdiDoks, a modern documentation theme, which is a port of the Hugo theme [Doks](https://github.com/h-enk/doks) for Zola.
[Bootstrap](https://icons.getbootstrap.com/): AdiDoks is built on Bootstrap
[Static Web Server](https://static-web-server.net/): Rust-based SWS used when deployed, see the [docs.nix](../../nix/docs.nix) file for usage.

*note: The styles and templates have been heavily modified since AdiDoks, and will not always be recognizable.*

Inspired by the presentation of these documentation sites, albeit a work in progress:     
[Oxide](https://docs.oxide.computer/guides/introduction)    
[Rerun](https://rerun.io/docs/getting-started/what-is-rerun)

### Development

```sh
brew install zola
cd docs/public
zola serve
```
