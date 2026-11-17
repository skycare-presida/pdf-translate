#!/usr/bin/env bash
# =============================================================================
# install.sh — SkyCare PDF Translation Service v3 installer
# Async job queue: submit → poll → fetch pattern eliminates timeout risk.
# Tested on Ubuntu 24.04 (Ancilla). Must be run as root (sudo ./install.sh).
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}── $* ${RESET}"; }

[[ $EUID -eq 0 ]] || error "Please run as root: sudo ./install.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="skycare-pdf-translate-v3"
INSTALL_DIR="/opt/skycare/pdf-translate-v3"
V1_INSTALL_DIR="/opt/skycare/pdf-translate"
V2_INSTALL_DIR="/opt/skycare/pdf-translate-v2"
LOG_DIR="/var/log/skycare"
JOB_DIR="/tmp/skycare-jobs"
SYSTEMD_UNIT="/etc/systemd/system/${SERVICE_NAME}.service"
DEFAULT_PORT=5107
PRESIDA_IP="51.77.119.156"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  SkyCare PDF Translation Service v3 — Installer         ║${RESET}"
echo -e "${BOLD}║  Async Job Queue (submit → poll → fetch)                ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Step 0: System packages ───────────────────────────────────────────────────
# Tesseract + Cyrillic language packs are used by the Mode B/C OCR fallback
# when the vision model over-produces on a non-Latin scan (Nov 2026 — Ukrainian
# oncology discharge form provoked a 129k-character hallucination from
# Mistral-Small; Tesseract read the same page cleanly).
step "0/8  System packages (tesseract + language packs)"

TESSERACT_PKGS=(
    tesseract-ocr
    tesseract-ocr-ukr tesseract-ocr-rus tesseract-ocr-bul
    tesseract-ocr-srp tesseract-ocr-mkd tesseract-ocr-bel tesseract-ocr-kaz
)
MISSING_PKGS=()
for pkg in "${TESSERACT_PKGS[@]}"; do
    dpkg -s "$pkg" &>/dev/null || MISSING_PKGS+=("$pkg")
done
if (( ${#MISSING_PKGS[@]} > 0 )); then
    info "Installing: ${MISSING_PKGS[*]}"
    apt-get update -qq
    apt-get install -y -qq "${MISSING_PKGS[@]}"
    success "Tesseract and Cyrillic language packs installed."
else
    success "Tesseract and language packs already installed."
fi

# ── Step 1: Port check ────────────────────────────────────────────────────────
step "1/8  Port availability check"

CHECK_PORTS="${V1_INSTALL_DIR}/check_ports.sh"
[[ -x "$CHECK_PORTS" ]] && bash "$CHECK_PORTS" || true

# A re-run (an update) finds the port held by this very service — that is not
# a conflict. Reuse the port the existing install is configured for.
EXISTING_INSTALL=false
EXISTING_PORT=""
if [[ -f "$SYSTEMD_UNIT" && -f "${INSTALL_DIR}/.env" ]]; then
    EXISTING_INSTALL=true
    EXISTING_PORT=$(grep "^TRANSLATE_PORT=" "${INSTALL_DIR}/.env" | cut -d= -f2- || true)
    EXISTING_PORT="${EXISTING_PORT:-$DEFAULT_PORT}"
fi

if $EXISTING_INSTALL && systemctl is-active --quiet "${SERVICE_NAME}"; then
    PORT=$EXISTING_PORT
    success "Port ${PORT} is held by the running ${SERVICE_NAME} — updating in place."
elif $EXISTING_INSTALL && ! ss -tlnp | grep -q ":${EXISTING_PORT} "; then
    PORT=$EXISTING_PORT
    success "Existing install found (service stopped) — keeping port ${PORT}."
elif ss -tlnp | grep -q ":${DEFAULT_PORT} "; then
    warn "Port ${DEFAULT_PORT} is already in use."
    read -rp "        Enter an alternative port (or press Enter to abort): " PORT
    [[ -z "$PORT" ]] && error "Aborted by user."
else
    PORT=$DEFAULT_PORT
    success "Port ${PORT} is free."
fi

# ── Step 2: Venv detection ────────────────────────────────────────────────────
step "2/8  Python venv"

SHARE_VENV=false
VENV_SOURCE=""

# Prefer v2 venv, fall back to v1
for DIR in "$V2_INSTALL_DIR" "$V1_INSTALL_DIR"; do
    if [[ -d "${DIR}/venv" ]]; then
        VENV_SOURCE="${DIR}/venv"
        break
    fi
done

if [[ -n "$VENV_SOURCE" ]]; then
    info "Found existing venv at ${VENV_SOURCE}"
    read -rp "        Share this venv? [Y/n]: " SHARE
    SHARE="${SHARE:-Y}"
    if [[ "${SHARE^^}" == "Y" ]]; then
        SHARE_VENV=true
        success "Will symlink ${VENV_SOURCE}"
    fi
fi

# ── Step 3: API keys ──────────────────────────────────────────────────────────
step "3/8  API keys"

OVHCLOUD_KEY=""
SERVICE_KEY=""
DEEPL_KEY=""

# Try to inherit from this install's own .env (a re-run), then v2, then v1
for ENV_FILE in "${INSTALL_DIR}/.env" "${V2_INSTALL_DIR}/.env" "${V1_INSTALL_DIR}/.env"; do
    if [[ -f "$ENV_FILE" ]]; then
        [[ -z "$DEEPL_KEY"    ]] && DEEPL_KEY=$(grep "^DEEPL_AUTH_KEY=" "$ENV_FILE" | cut -d= -f2- || true)
        [[ -z "$OVHCLOUD_KEY" ]] && OVHCLOUD_KEY=$(grep "^OVHCLOUD_API_KEY=" "$ENV_FILE" | cut -d= -f2- || true)
        [[ -z "$SERVICE_KEY"  ]] && SERVICE_KEY=$(grep "^TRANSLATE_API_KEY=" "$ENV_FILE" | cut -d= -f2- || true)
    fi
done

if [[ -n "$OVHCLOUD_KEY" ]]; then
    info "OVHcloud API key inherited from existing installation."
else
    read -rsp "        OVHcloud API key (input hidden): " OVHCLOUD_KEY; echo ""
    [[ -z "$OVHCLOUD_KEY" ]] && warn "No key entered — add it later to ${INSTALL_DIR}/.env"
fi

# DeepL is the primary translation engine; the OVHcloud models above still do
# OCR and QA, and take over translation whenever DeepL is unavailable.
if [[ -n "$DEEPL_KEY" ]]; then
    info "DeepL API key inherited from existing installation."
else
    read -rsp "        DeepL API key (input hidden; Enter to skip): " DEEPL_KEY; echo ""
    [[ -z "$DEEPL_KEY" ]] && warn "No DeepL key — translation will use the OVHcloud models. Add DEEPL_AUTH_KEY to ${INSTALL_DIR}/.env later."
fi

if [[ -n "$SERVICE_KEY" ]]; then
    info "Inbound API key inherited — FileMaker scripts need no header change."
else
    SERVICE_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    success "Generated new inbound API key: ${SERVICE_KEY}"
    warn "Add this key to your FileMaker X-API-Key header."
fi

# ── Step 4: Directories ───────────────────────────────────────────────────────
step "4/8  Creating directories"

mkdir -p "$INSTALL_DIR" "$LOG_DIR" "$JOB_DIR"
chown www-data:www-data "$LOG_DIR" "$JOB_DIR" 2>/dev/null || true
chmod 755 "$LOG_DIR"
chmod 700 "$JOB_DIR"   # job PDFs contain patient data — restrict to www-data
success "Directories ready. Job dir: ${JOB_DIR}"

# ── Step 5: Deploy service files ──────────────────────────────────────────────
step "5/8  Deploying service files"

cp "${SCRIPT_DIR}/service/pdf_translate_v3_service.py" "${INSTALL_DIR}/"
cp "${SCRIPT_DIR}/service/requirements.txt"            "${INSTALL_DIR}/"

# Settings in an existing .env that this installer does not manage (e.g.
# TRANSLATE_ENGINE, a bind address) must survive an update.
EXTRA_ENV=""
if [[ -f "${INSTALL_DIR}/.env" ]]; then
    EXTRA_ENV=$(grep -vE "^(OVHCLOUD_API_KEY|TRANSLATE_API_KEY|TRANSLATE_PORT|DEEPL_AUTH_KEY)=" "${INSTALL_DIR}/.env" || true)
fi

cat > "${INSTALL_DIR}/.env" <<EOF
OVHCLOUD_API_KEY=${OVHCLOUD_KEY}
TRANSLATE_API_KEY=${SERVICE_KEY}
TRANSLATE_PORT=${PORT}
DEEPL_AUTH_KEY=${DEEPL_KEY}
EOF
[[ -n "$EXTRA_ENV" ]] && printf '%s\n' "$EXTRA_ENV" >> "${INSTALL_DIR}/.env"
chmod 600 "${INSTALL_DIR}/.env"
chown -R www-data:www-data "$INSTALL_DIR"
success "Files deployed."

# ── Step 6: Venv ──────────────────────────────────────────────────────────────
step "6/8  Python virtual environment"

if [[ "$SHARE_VENV" == "true" ]]; then
    ln -sfn "$VENV_SOURCE" "${INSTALL_DIR}/venv"
    success "Symlinked ${VENV_SOURCE} → ${INSTALL_DIR}/venv"
    # A shared venv predates newer dependencies (e.g. deepl) — top it up.
    if sudo -u www-data "${INSTALL_DIR}/venv/bin/pip" install --quiet \
        -r "${INSTALL_DIR}/requirements.txt"; then
        success "Shared venv dependencies are up to date."
    else
        warn "Could not update the shared venv — run: ${INSTALL_DIR}/venv/bin/pip install -r ${INSTALL_DIR}/requirements.txt"
    fi
else
    if ! python3 -m venv --help &>/dev/null; then
        apt-get install -y python3-venv python3-pip
    fi
    [[ ! -d "${INSTALL_DIR}/venv" ]] && sudo -u www-data python3 -m venv "${INSTALL_DIR}/venv"
    sudo -u www-data "${INSTALL_DIR}/venv/bin/pip" install --quiet --upgrade pip
    sudo -u www-data "${INSTALL_DIR}/venv/bin/pip" install --quiet \
        -r "${INSTALL_DIR}/requirements.txt"
    success "Fresh venv created and dependencies installed."
fi

# Verify the OCR-preprocessing libraries actually import. Both the OpenCV
# preprocessing step and the Tesseract fallback degrade silently if their
# libraries are missing (by design — the service continues working), so a
# missing cv2 or pytesseract shows up only later as "the Ukrainian scan
# produced garbage". Catch it here instead.
if sudo -u www-data "${INSTALL_DIR}/venv/bin/python" -c \
    "import cv2, numpy, pytesseract" 2>/dev/null; then
    success "OpenCV, numpy and pytesseract import cleanly in the venv."
else
    warn "OpenCV / numpy / pytesseract did not import — OCR preprocessing"
    warn "and the Cyrillic fallback will be disabled. Service still runs."
    warn "Debug with: ${INSTALL_DIR}/venv/bin/python -c 'import cv2, numpy, pytesseract'"
fi

if sudo -u www-data "${INSTALL_DIR}/venv/bin/python" -c "import deepl" 2>/dev/null; then
    success "deepl imports cleanly in the venv."
else
    warn "deepl did not import — translation will fall back to the OVHcloud models."
fi

# ── Step 7: Firewall ──────────────────────────────────────────────────────────
step "7/8  Firewall"

if $EXISTING_INSTALL; then
    info "Existing install — firewall rules left as they are."
elif command -v ufw &>/dev/null; then
    if ufw status | grep -q "${PORT}.*${PRESIDA_IP}"; then
        info "UFW rule for port ${PORT} already exists."
    else
        ufw allow from "${PRESIDA_IP}" to any port "${PORT}" proto tcp
        success "UFW: allowed port ${PORT} from ${PRESIDA_IP} (Presida)."
    fi
else
    warn "ufw not found — add firewall rule manually:"
    warn "  ufw allow from ${PRESIDA_IP} to any port ${PORT} proto tcp"
fi

# ── Step 8: Systemd ───────────────────────────────────────────────────────────
step "8/8  Systemd service"

# Never overwrite a unit that has been edited on this box (e.g. to bind the
# WireGuard address instead of 0.0.0.0) — keep it and say so.
REPO_UNIT="${SCRIPT_DIR}/scripts/skycare-pdf-translate-v3.service"
if [[ -f "$SYSTEMD_UNIT" ]] && ! cmp -s "$REPO_UNIT" "$SYSTEMD_UNIT"; then
    warn "Installed systemd unit differs from the repo copy — keeping the installed one."
    warn "Compare with: diff ${REPO_UNIT} ${SYSTEMD_UNIT}"
else
    cp "$REPO_UNIT" "$SYSTEMD_UNIT"
fi
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"
sleep 3

# Health check
info "Running health check..."
HEALTH=$(curl -sf "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "FAILED")

if echo "$HEALTH" | grep -q "\"status\": \"ok\""; then
    success "Health check passed."
    echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
else
    warn "Health check did not return expected response."
    warn "Check: journalctl -u ${SERVICE_NAME} -n 30"
    echo "Response: ${HEALTH}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║              v3 Installation complete                   ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  Service:    ${GREEN}${SERVICE_NAME}${RESET}"
echo -e "  Port:       ${GREEN}${PORT}${RESET}"
echo -e "  Install:    ${INSTALL_DIR}"
echo -e "  Job dir:    ${JOB_DIR} (TTL: 2 hours)"
echo -e "  Logs:       ${LOG_DIR}/${SERVICE_NAME}-{access,error}.log"
echo ""
echo -e "  ${BOLD}FileMaker script pattern:${RESET}"
echo -e "    1. POST /translate/submit  → receive job_id"
echo -e "    2. Loop: wait 10s → GET /translate/status/<job_id>"
echo -e "    3. GET /translate/result/<job_id>  → store PDF"
echo ""
echo -e "  Quick async test:"
echo -e "    ${CYAN}# Submit${RESET}"
echo -e "    ${CYAN}curl -X POST http://ancilla.skycare.uk:${PORT}/translate/submit \\${RESET}"
echo -e "    ${CYAN}  -H 'X-API-Key: ${SERVICE_KEY}' \\${RESET}"
echo -e "    ${CYAN}  -F 'file=@/tmp/test.pdf'${RESET}"
echo ""
echo -e "    ${CYAN}# Poll (replace JOB_ID)${RESET}"
echo -e "    ${CYAN}curl -H 'X-API-Key: ${SERVICE_KEY}' \\${RESET}"
echo -e "    ${CYAN}  http://ancilla.skycare.uk:${PORT}/translate/status/JOB_ID${RESET}"
echo ""
echo -e "    ${CYAN}# Fetch result${RESET}"
echo -e "    ${CYAN}curl -H 'X-API-Key: ${SERVICE_KEY}' \\${RESET}"
echo -e "    ${CYAN}  http://ancilla.skycare.uk:${PORT}/translate/result/JOB_ID \\${RESET}"
echo -e "    ${CYAN}  --output /tmp/translated.pdf${RESET}"
echo ""
echo -e "  Sync test (direct, no polling):"
echo -e "    ${CYAN}curl -X POST http://ancilla.skycare.uk:${PORT}/translate \\${RESET}"
echo -e "    ${CYAN}  -H 'X-API-Key: ${SERVICE_KEY}' \\${RESET}"
echo -e "    ${CYAN}  -F 'file=@/tmp/test.pdf' --output /tmp/translated.pdf${RESET}"
echo ""
echo -e "  Previous versions remain on:"
echo -e "    v1: port 5105   v2: port 5106"
echo ""
