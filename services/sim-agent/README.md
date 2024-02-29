# Elodin Web Runner

## Build Docker Image

To build the Docker image for this service you will need Nix installed. Then you can run:

``` sh
docker load -i $(nix build .#packages.aarch64-linux.docker.image --print-out-paths) # for arm
docker load -i $(nix build .#packages.x86_64-linux.docker.image --print-out-paths) # for x86
```

To build on macOS you will need to follow the instructions in [docs/nix.md](../../docs/nix.md) under macOS VM

## Build and run sandbox micro VM

NOTE: Only works on Linux due to host kernel dependency

```
nix build .#sandbox-vm
qemu-system-x86_64 -M microvm -kernel /boot/vmlinuz-(uname -r) -append 'quiet root=/dev/vda ro init=/init' -cpu host -enable-kvm -m 1G -nographic -drive file=result/root.squashfs,format=raw,id=root,if=none,readonly=on -device virtio-blk-device,drive=root
```

- `-kernel /boot/vmlinuz-(uname -r)`
    - Use host kernel. This is fine for now, but eventually we should bundle a smaller, more locked down kernel.
- `-append 'quiet root=/dev/vda init=/init' ro`
    - Replace "quiet" with "debug" to view more kernel logs, which is especially useful for debugging the init process.
    - `root=/dev/vda init=/init` is used to skip initrd/initramfs and jump directly to the `/init` process in the attached squashfs drive.
- `-drive file=result/root.squashfs,format=raw,id=root,if=none,readonly=on`
    - Use a read-only squashfs root file-system. An immutable file-system should be compatible with our use cases. If not, we can just mount an overlayfs on top.
