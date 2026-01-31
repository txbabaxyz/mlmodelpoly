#!/bin/bash
# ============================================================
# Binance Collector - Smoke Test Script
# ============================================================
#
# Usage:
#   ./scripts/smoke_test.sh [host:port]
#
# Default: localhost:8000
#
# Exit codes:
#   0 - All tests passed
#   1 - One or more tests failed
#
# ============================================================

set -e

HOST="${1:-localhost:8000}"
BASE_URL="http://${HOST}"
FAILED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Binance Collector Smoke Tests"
echo "Target: ${BASE_URL}"
echo "=============================================="
echo ""

# Helper function
check() {
    local name="$1"
    local condition="$2"
    
    if eval "$condition"; then
        echo -e "${GREEN}✓${NC} $name"
        return 0
    else
        echo -e "${RED}✗${NC} $name"
        FAILED=1
        return 1
    fi
}

# ============================================
# Test 1: Health endpoint
# ============================================
echo "--- Test 1: Health Endpoint ---"

HEALTH=$(curl -s "${BASE_URL}/health" 2>/dev/null || echo "{}")
echo "Response: $HEALTH"

check "Health endpoint returns ok=true" \
    "echo '$HEALTH' | jq -e '.ok == true' > /dev/null 2>&1"

echo ""

# ============================================
# Test 2: State endpoint
# ============================================
echo "--- Test 2: State Endpoint ---"

STATE=$(curl -s "${BASE_URL}/state" 2>/dev/null || echo "{}")

check "State endpoint returns data" \
    "echo '$STATE' | jq -e '.server_time_ms' > /dev/null 2>&1"

check "State has ws_connected" \
    "echo '$STATE' | jq -e '.ws_connected' > /dev/null 2>&1"

check "State has uptime_ms" \
    "echo '$STATE' | jq -e '.uptime_ms > 0' > /dev/null 2>&1"

echo ""

# ============================================
# Test 3: Latest Features endpoint
# ============================================
echo "--- Test 3: Latest Features Endpoint ---"

FEATURES=$(curl -s "${BASE_URL}/latest/features" 2>/dev/null || echo "{}")

check "Features endpoint returns data" \
    "echo '$FEATURES' | jq -e '.schema_version' > /dev/null 2>&1"

check "Features has futures section" \
    "echo '$FEATURES' | jq -e '.futures' > /dev/null 2>&1"

check "Features has spot section" \
    "echo '$FEATURES' | jq -e '.spot' > /dev/null 2>&1"

check "Features has basis section" \
    "echo '$FEATURES' | jq -e '.basis' > /dev/null 2>&1"

check "Features has anchor section" \
    "echo '$FEATURES' | jq -e '.anchor' > /dev/null 2>&1"

check "Features has orderbook section" \
    "echo '$FEATURES' | jq -e '.orderbook' > /dev/null 2>&1"

check "Features has liquidations section" \
    "echo '$FEATURES' | jq -e '.liquidations' > /dev/null 2>&1"

check "Features has quality section" \
    "echo '$FEATURES' | jq -e '.quality' > /dev/null 2>&1"

check "Quality mode exists" \
    "echo '$FEATURES' | jq -e '.quality.mode' > /dev/null 2>&1"

echo ""

# ============================================
# Test 4: Data Validation
# ============================================
echo "--- Test 4: Data Validation ---"

# Check futures CVD exists (data is flowing)
FUTURES_CVD=$(echo "$FEATURES" | jq -r '.futures.cvd // empty')
check "Futures CVD is present" \
    "[ -n '$FUTURES_CVD' ]"

# Check spot CVD exists
SPOT_CVD=$(echo "$FEATURES" | jq -r '.spot.cvd // empty')
check "Spot CVD is present" \
    "[ -n '$SPOT_CVD' ]"

# Check anchor mode
ANCHOR_MODE=$(echo "$FEATURES" | jq -r '.anchor.mode // empty')
check "Anchor mode is AUTO_UTC or MANUAL" \
    "[ '$ANCHOR_MODE' = 'AUTO_UTC' ] || [ '$ANCHOR_MODE' = 'MANUAL' ]"

# Check AVWAP exists
AVWAP=$(echo "$FEATURES" | jq -r '.anchor.avwap_15m // empty')
check "AVWAP 15m is present" \
    "[ -n '$AVWAP' ]"

# Check quality mode is valid
QUALITY_MODE=$(echo "$FEATURES" | jq -r '.quality.mode // empty')
check "Quality mode is OK/DEGRADED/BAD" \
    "[ '$QUALITY_MODE' = 'OK' ] || [ '$QUALITY_MODE' = 'DEGRADED' ] || [ '$QUALITY_MODE' = 'BAD' ]"

echo ""

# ============================================
# Test 5: WebSocket Connectivity
# ============================================
echo "--- Test 5: WebSocket Connectivity ---"

WS_FUTURES=$(echo "$FEATURES" | jq -r '.quality.ws_futures // false')
WS_SPOT=$(echo "$FEATURES" | jq -r '.quality.ws_spot // false')

check "Futures WebSocket connected" \
    "[ '$WS_FUTURES' = 'true' ]"

check "Spot WebSocket connected" \
    "[ '$WS_SPOT' = 'true' ]"

echo ""

# ============================================
# Test 6: Bars (if available)
# ============================================
echo "--- Test 6: Bars ---"

BARS_TOTAL=$(echo "$STATE" | jq -r '.bars_total // {}')

check "Bars total section exists" \
    "echo '$STATE' | jq -e '.bars_total' > /dev/null 2>&1"

echo ""

# ============================================
# Summary
# ============================================
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All smoke tests PASSED${NC}"
    echo "=============================================="
    echo ""
    echo "Quick summary:"
    echo "  Quality mode: $QUALITY_MODE"
    echo "  Futures CVD:  $FUTURES_CVD"
    echo "  Spot CVD:     $SPOT_CVD"
    echo "  AVWAP 15m:    $AVWAP"
    echo "  WS Futures:   $WS_FUTURES"
    echo "  WS Spot:      $WS_SPOT"
    exit 0
else
    echo -e "${RED}Some smoke tests FAILED${NC}"
    echo "=============================================="
    exit 1
fi
