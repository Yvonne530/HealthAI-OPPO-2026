#!/usr/bin/env bash
# setup_mnn_290.sh
# ================
# Downloads MNN 2.9.0 AAR and extracts:
#   - C++ headers  → app/src/main/cpp/include/MNN/
#   - libMNN.so    → app/src/main/jniLibs/arm64-v8a/
#   - libc++_shared.so  → app/src/main/jniLibs/arm64-v8a/
#   - MNN AAR      → app/libs/  (supplies Java classes if needed)
#
# Run from the project root:
#   bash scripts/setup_mnn_290.sh
#
# Requirements: curl, unzip (both available on macOS and Ubuntu)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$PROJECT_ROOT/app"

MNN_VERSION="2.9.0"
# Official MNN 2.9.0 Android AAR from GitHub Releases
# If this URL changes, find the latest at: https://github.com/alibaba/MNN/releases
MNN_AAR_URL="https://github.com/alibaba/MNN/releases/download/${MNN_VERSION}/MNN-Android-${MNN_VERSION}.aar"
AAR_FILE="$APP_DIR/libs/MNN-Android-${MNN_VERSION}.aar"

# Target directories
CPP_INCLUDE="$APP_DIR/src/main/cpp/include"
JNI_LIBS="$APP_DIR/src/main/jniLibs/arm64-v8a"
LIBS_DIR="$APP_DIR/libs"

echo "======================================================="
echo "  MNN $MNN_VERSION Android Setup"
echo "======================================================="

# ── Step 1: Create directories ─────────────────────────────────────────────
mkdir -p "$CPP_INCLUDE" "$JNI_LIBS" "$LIBS_DIR"

# ── Step 2: Download AAR if not cached ─────────────────────────────────────
if [ -f "$AAR_FILE" ]; then
    echo "[✓] AAR already present: $AAR_FILE"
else
    echo "[↓] Downloading MNN $MNN_VERSION AAR..."
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$AAR_FILE" "$MNN_AAR_URL"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$AAR_FILE" "$MNN_AAR_URL"
    else
        echo "[ERROR] Neither curl nor wget found."
        echo "        Manually download:"
        echo "        $MNN_AAR_URL"
        echo "        → $AAR_FILE"
        exit 1
    fi
    echo "[✓] Downloaded to $AAR_FILE"
fi

# ── Step 3: Extract from AAR ────────────────────────────────────────────────
# An AAR is a ZIP containing:
#   jni/arm64-v8a/libMNN.so
#   jni/arm64-v8a/libc++_shared.so
#   headers/  (may be absent — then we fetch headers separately)
TMP_DIR="$(mktemp -d)"
trap "rm -rf $TMP_DIR" EXIT

echo "[→] Extracting AAR..."
unzip -q "$AAR_FILE" -d "$TMP_DIR"

# Extract libMNN.so (arm64-v8a)
SO_PATHS=("$TMP_DIR/jni/arm64-v8a/libMNN.so"
           "$TMP_DIR/jni/arm64-v8a/libMNN_CL.so"
           "$TMP_DIR/jni/arm64-v8a/libc++_shared.so")

for SO in "${SO_PATHS[@]}"; do
    if [ -f "$SO" ]; then
        cp "$SO" "$JNI_LIBS/"
        echo "[✓] Copied $(basename $SO)"
    fi
done

if [ ! -f "$JNI_LIBS/libMNN.so" ]; then
    echo "[ERROR] libMNN.so not found in AAR."
    echo "        Expected path: jni/arm64-v8a/libMNN.so"
    echo "        Contents of AAR:"
    ls "$TMP_DIR/jni/" 2>/dev/null || true
    exit 1
fi

# ── Step 4: Extract / download C++ headers ─────────────────────────────────
HEADER_SRC="$TMP_DIR/headers"
if [ -d "$HEADER_SRC" ]; then
    cp -r "$HEADER_SRC/." "$CPP_INCLUDE/"
    echo "[✓] Headers extracted from AAR"
else
    # Headers may not be in the AAR — fetch from source tarball
    echo "[→] Headers not in AAR — fetching from GitHub source..."
    TARBALL_URL="https://github.com/alibaba/MNN/archive/refs/tags/${MNN_VERSION}.tar.gz"
    TARBALL="$TMP_DIR/mnn-src.tar.gz"

    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$TARBALL" "$TARBALL_URL"
    else
        wget --show-progress -O "$TARBALL" "$TARBALL_URL"
    fi

    tar -xzf "$TARBALL" -C "$TMP_DIR" --strip-components=1 \
        "MNN-${MNN_VERSION}/include/MNN"

    if [ -d "$TMP_DIR/include/MNN" ]; then
        mkdir -p "$CPP_INCLUDE/MNN"
        cp -r "$TMP_DIR/include/MNN/." "$CPP_INCLUDE/MNN/"
        echo "[✓] Headers extracted from source tarball"
    else
        echo "[ERROR] Could not find MNN headers."
        echo "        Manually copy the include/MNN/ folder from the MNN source"
        echo "        to: $CPP_INCLUDE/MNN/"
        exit 1
    fi
fi

# ── Step 5: Validation ──────────────────────────────────────────────────────
echo ""
echo "── Validation ──────────────────────────────────────────"

REQUIRED_HEADERS=("Interpreter.hpp" "Tensor.hpp" "MNNDefine.h" "ErrorCode.hpp")
ALL_OK=true
for H in "${REQUIRED_HEADERS[@]}"; do
    if [ -f "$CPP_INCLUDE/MNN/$H" ]; then
        echo "  [✓] $H"
    else
        echo "  [✗] MISSING: $H"
        ALL_OK=false
    fi
done

if [ -f "$JNI_LIBS/libMNN.so" ]; then
    SIZE=$(du -h "$JNI_LIBS/libMNN.so" | cut -f1)
    echo "  [✓] libMNN.so  ($SIZE)"
else
    echo "  [✗] MISSING: libMNN.so"
    ALL_OK=false
fi

echo ""
if [ "$ALL_OK" = true ]; then
    echo "======================================================="
    echo "  ✅ Setup complete! Now build the project:"
    echo "     ./gradlew assembleDebug"
    echo ""
    echo "  Verify JNI is active by checking logcat for:"
    echo "     ✅ REAL JNI INFERENCE ACTIVE — MNN $MNN_VERSION"
    echo "======================================================="
else
    echo "[ERROR] Some files are missing. Check errors above."
    exit 1
fi
