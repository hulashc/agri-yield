import os

# Override via BUILD_VERSION env var at container runtime.
# Set at build time via CI (e.g. docker build --build-arg BUILD_VERSION=${{ github.sha }}).
BUILD_VERSION = os.getenv("BUILD_VERSION", "dev")
