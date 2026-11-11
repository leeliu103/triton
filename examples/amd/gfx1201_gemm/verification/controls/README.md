# Negative controls: deliberately broken kernels

The three `kernel.amdgcn` files here are copies of the double-bank kernel with
one piece of synchronization removed. **They produce wrong results by design.**
They exist only to show that the bit-exact check detects an LDS race. They have
no HSACO, are not in `manifest.json`, and must not be used as candidates.
