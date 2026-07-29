"""
Cortex VM Infrastructure — The AI that builds and runs computers.

Four capabilities:
  1. Self-replication: boot Cortex clones in VMs for testing/experimentation
  2. Sandboxed execution: disposable VMs for untrusted code
  3. Multi-tenant: isolated Cortex instances per user/context
  4. Full OS control: provision and manage entire operating systems

Platform backends:
  - macOS: Virtualization.framework (Apple Silicon native)
  - Linux: libvirt/qemu-kvm
  - Windows: Hyper-V (PowerShell)
  - Fallback: Docker/Podman containers (not true VMs but practical)
"""
