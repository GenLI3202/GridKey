#!/bin/bash
#
# GridKey Optimizer Service - Startup Script
# ==========================================
# Convenience script for local development and testing.
#
# Usage:
#   ./startup.sh          - Run with default configuration
#   ./startup.sh dev      - Run with hot-reload (uvicorn --reload)
#   ./startup.sh test     - Run tests instead of server
#

set -e

# Default configuration
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
WORKERS=${WORKERS:-"1"}
RELOAD=${RELOAD:-"false"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  GridKey Optimizer Service${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo -e "${RED}Error: uvicorn not found${NC}"
    echo "Install with: pip install uvicorn[standard]"
    exit 1
fi

# Handle command line arguments
case "${1:-}" in
    "dev")
        echo -e "${YELLOW}Mode: Development (hot-reload enabled)${NC}"
        RELOAD="true"
        ;;
    "test")
        echo -e "${YELLOW}Mode: Test Suite${NC}"
        echo ""
        pytest src/test/ -v --tb=short
        exit $?
        ;;
    *)
        echo -e "${YELLOW}Mode: Production${NC}"
        ;;
esac

# Display configuration
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Workers: ${WORKERS}"
echo "Reload: ${RELOAD}"
echo ""

# Build uvicorn command
UVICORN_CMD="uvicorn src.api.main:app"
UVICORN_CMD="${UVICORN_CMD} --host ${HOST} --port ${PORT}"

if [ "$RELOAD" = "true" ]; then
    UVICORN_CMD="${UVICORN_CMD} --reload"
fi

if [ "$WORKERS" -gt 1 ] && [ "$RELOAD" = "false" ]; then
    UVICORN_CMD="${UVICORN_CMD} --workers ${WORKERS}"
fi

# Start server
echo -e "${GREEN}Starting server...${NC}"
echo -e "${GREEN}API will be available at: http://${HOST}:${PORT}${NC}"
echo -e "${GREEN}Docs at: http://${HOST}:${PORT}/docs${NC}"
echo ""

eval $UVICORN_CMD
