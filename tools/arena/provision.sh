#!/usr/bin/env bash
# Prepare one competitor box (Ubuntu 24.04, x86_64), as root: the AMD ROCm
# compiler toolkit, the usual developer tools, Node.js LTS, uv, swap, key-only
# SSH, automatic security updates, and an `agent` user who owns the workspace.
# The agent gets passwordless sudo: the whole box is its sandbox, and it holds
# nothing but that agent's own login. Safe to run again.
#
# Usage: ssh root@BOX 'bash -s' < tools/arena/provision.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_SUSPEND=1  # quiet: nothing here needs a service restart

ROCM_VERSION=7.2.4
NODE_MAJOR=24
UV_VERSION=0.10.9
UV_SHA256=433e56874739e92c7cfd661ba9e5f287b376ca612c08c8194a41a98a13158aea  # x86_64 musl

. /etc/os-release
[ "${VERSION_CODENAME:-}" = noble ] || { echo "needs Ubuntu 24.04 (noble), not ${PRETTY_NAME:-unknown}" >&2; exit 1; }
[ "$(uname -m)" = x86_64 ] || { echo "needs x86_64: AMD's compiler ships for nothing else" >&2; exit 1; }
step() { printf '== %s\n' "$*"; }

step "swap: 4 GB"
if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
    [ -f /swapfile ] || { fallocate -l 4G /swapfile; chmod 600 /swapfile; mkswap /swapfile > /dev/null; }
    swapon /swapfile
fi
grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab

step "developer tools"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    build-essential cmake ninja-build pkg-config git git-lfs curl wget ca-certificates gnupg \
    jq ripgrep fd-find tree tmux screen htop unzip zip xz-utils rsync file bc less man-db time \
    vim nano bsdextrautils python3 python3-venv python3-pip python3-dev pipx python3-numpy \
    python3-sympy gdb strace ltrace valgrind shellcheck sqlite3 libsodium-dev libssl-dev \
    unattended-upgrades > /dev/null
ln -sf "$(command -v fdfind)" /usr/local/bin/fd
printf 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Unattended-Upgrade "1";\n' \
    > /etc/apt/apt.conf.d/20auto-upgrades  # Ubuntu security updates only

install -d -m 755 /etc/apt/keyrings
step "Node.js ${NODE_MAJOR} LTS"
if ! node --version 2>/dev/null | grep -q "^v${NODE_MAJOR}\."; then
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list
    apt-get update -qq
    apt-get install -y -qq nodejs > /dev/null
fi

step "ROCm ${ROCM_VERSION}: rocm-dev"
# The pin matters: unpinned, Ubuntu's own hipcc wins on version number, and it
# runs a stock clang that does not know gfx950.
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor --yes -o /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
    > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
apt-get update -qq
apt-get install -y -qq rocm-dev > /dev/null
# Every SSH session reads /etc/environment, including the agents' tool calls.
grep -q /opt/rocm/bin /etc/environment \
    || sed -i 's#^PATH="\(.*\)"$#PATH="\1:/opt/rocm/bin"#' /etc/environment

step "uv ${UV_VERSION}"
if ! /usr/local/bin/uv --version 2>/dev/null | grep -q " ${UV_VERSION}"; then
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/uv.tar.gz" \
        "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-musl.tar.gz"
    echo "${UV_SHA256}  $tmp/uv.tar.gz" | sha256sum -c --quiet
    tar -xzf "$tmp/uv.tar.gz" -C /usr/local/bin --strip-components=1
    rm -rf "$tmp"
fi

step "agent user"
id agent > /dev/null 2>&1 || useradd -m -s /bin/bash agent
echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-agent
chmod 440 /etc/sudoers.d/90-agent
install -d -m 700 -o agent -g agent /home/agent/.ssh
install -m 600 -o agent -g agent /root/.ssh/authorized_keys /home/agent/.ssh/authorized_keys

step "SSH: keys only"
# Lexically first drop-in wins in sshd, so this beats cloud-init's 50-*.conf.
printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin prohibit-password\n' \
    > /etc/ssh/sshd_config.d/10-keys-only.conf
sshd -t
systemctl reload ssh

step "versions"
echo "hipcc $(dpkg-query -W -f='${Version}' hipcc), node $(node --version), uv $(/usr/local/bin/uv --version | cut -d' ' -f2), python $(python3 --version | cut -d' ' -f2)"
echo "provision: OK"
