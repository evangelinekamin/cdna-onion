# Source me (`. tools/sol/env.sh`): puts the repo-local uv from tools/sol/setup.sh
# on PATH and keeps its cache and Python downloads inside the repo.
_onionrig_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PATH="$_onionrig_root/.uv/bin:$PATH"
export UV_CACHE_DIR="$_onionrig_root/.uv/cache"
export UV_PYTHON_INSTALL_DIR="$_onionrig_root/.uv/python"
unset _onionrig_root
