#!/usr/bin/env bash
# ============================================================
# Download MNN Android AAR from GitHub releases
# Run ONCE before building: bash scripts/download_mnn_aar.sh
# ============================================================
set -e

MNN_VERSION="2.9.0"
AAR_NAME="MNN-${MNN_VERSION}-release.aar"
DOWNLOAD_URL="https://github.com/alibaba/MNN/releases/download/${MNN_VERSION}/${AAR_NAME}"
DEST="app/libs/${AAR_NAME}"

# Also try older naming convention
ALT_URL="https://github.com/alibaba/MNN/releases/download/2.8.2/MNN-2.8.2-release.aar"
ALT_DEST="app/libs/MNN-2.8.2-release.aar"

echo "Downloading MNN Android AAR..."
cd "$(dirname "$0")/.."

if curl -L --fail --max-time 120 -o "$DEST" "$DOWNLOAD_URL" 2>/dev/null; then
    echo "✅ Downloaded: $DEST"
elif curl -L --fail --max-time 120 -o "$ALT_DEST" "$ALT_URL" 2>/dev/null; then
    echo "✅ Downloaded: $ALT_DEST"
    # Update build.gradle reference
    sed -i "s|MNN-${MNN_VERSION}-release.aar|MNN-2.8.2-release.aar|g" app/build.gradle
    echo "   Updated build.gradle to reference MNN-2.8.2-release.aar"
else
    echo "❌ Auto-download failed. Manual steps:"
    echo "   1. Go to: https://github.com/alibaba/MNN/releases"
    echo "   2. Find latest release, download *-release.aar"
    echo "   3. Copy to: app/libs/"
    exit 1
fi

# Download .weight file companion if needed (MNN saveExternalData=1)
echo ""
echo "Note: MNN .mnn.weight files are your model weight files (already in assets/)."
echo "The AAR is the MNN runtime library. These are different."
echo ""
echo "✅ MNN runtime ready. Now run: ./gradlew assembleDebug"
