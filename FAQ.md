# Frequently Asked Questions (FAQ)

## When running `nix develop`, it warns that it's "ignoring untrustd substituter". How do I fix that?

If you run `nix develop` and it looks like this:
```sh
$ nix develop
warning: Git tree '/Users/shane/Projects/elodin' has uncommitted changes
warning: ignoring untrusted substituter 'https://elodin-nix-cache.s3.us-west-2.amazonaws.com', you are not a trusted user.
...
```
Edit the following files to look like this:

```sh
$ cat /etc/nix/nix.custom.conf
trusted-users = root YOUR-USER-NAME-HERE
extra-trusted-substituters = https://elodin-nix-cache.s3.us-west-2.amazonaws.com
extra-trusted-public-keys = elodin-cache-1:vvbmIQvTOjcBjIs8Ri7xlT2I3XAmeJyF5mNlWB+fIwM=
```

Once those edits are made, restart the nix daemon.

### macOS
```sh
sudo launchctl kickstart -k system/systems.determinate.nix-daemon
```
### Linux
```sh
sudo systemctl restart nix-daemon
```

