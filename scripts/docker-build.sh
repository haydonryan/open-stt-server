#!/usr/bin/env bash
set -euo pipefail

tag="${DOCKER_TAG:-$(date +%Y%m%d%H%M%S)-$(git rev-parse --short HEAD)}"
timeout="${DOCKER_HEALTH_TIMEOUT:-180}"
prune_age="${DOCKER_BUILDER_PRUNE_UNTIL:-168h}"

export OPEN_STT_IMAGE="open-stt-server-debian:${tag}"

echo "[docker-build] deploying tag ${tag}"
docker compose up -d --build --force-recreate --remove-orphans

mapfile -t containers < <(docker compose ps -q)
if [ "${#containers[@]}" -eq 0 ]; then
    echo "[docker-build] no compose containers found after deploy" >&2
    exit 1
fi

pending=1
current_image_ids=()
deadline=$((SECONDS + timeout))
while (( SECONDS < deadline )); do
    pending=0
    mapfile -t snapshot < <(docker inspect "${containers[@]}" | python - <<'PY2'
import json, sys
for doc in json.load(sys.stdin):
    state = doc.get('State', {})
    print('	'.join([
        doc.get('Id', ''),
        doc.get('Name', '').lstrip('/'),
        state.get('Status', ''),
        state.get('Health', {}).get('Status', 'none'),
        doc.get('Image', ''),
    ]))
PY2
)

    current_image_ids=()
    for row in "${snapshot[@]}"; do
        IFS=$'	' read -r container_id name state health image_id <<<"$row"
        current_image_ids+=("$image_id")

        if [[ "$state" == "exited" || "$state" == "dead" ]]; then
            echo "[docker-build] container failed: ${name}" >&2
            docker logs --tail 100 "$container_id" || true
            exit 1
        fi

        if [[ "$state" != "running" ]]; then
            pending=1
            continue
        fi

        if [[ "$health" != "none" && "$health" != "healthy" ]]; then
            pending=1
        fi
    done

    if [[ "$pending" -eq 0 ]]; then
        break
    fi

    sleep 2
done

if [[ "$pending" -ne 0 ]]; then
    echo "[docker-build] timed out waiting for containers to become ready" >&2
    docker compose ps
    exit 1
fi

mapfile -t current_image_ids < <(printf '%s\n' "${current_image_ids[@]}" | sort -u)

cleanup_repo() {
    local repo="$1"
    shift
    local keep_refs=("$@")
    mapfile -t rows < <(docker image ls "$repo" --no-trunc --format json | python - <<'PY2'
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    repo = obj.get('Repository')
    tag = obj.get('Tag')
    image_id = obj.get('ID')
    if repo and repo != '<none>' and tag and tag != '<none>' and image_id:
        print(f"{repo}:{tag}	{image_id}")
PY2
)

    for row in "${rows[@]}"; do
        IFS=$'	' read -r ref image_id <<<"$row"
        local keep=0

        for wanted in "${keep_refs[@]}"; do
            if [[ "$ref" == "$wanted" ]]; then
                keep=1
                break
            fi
        done

        if [[ "$keep" -eq 1 ]]; then
            continue
        fi

        for current_id in "${current_image_ids[@]}"; do
            if [[ "$image_id" == "$current_id" ]]; then
                keep=1
                break
            fi
        done

        if [[ "$keep" -eq 0 ]]; then
            docker image rm "$ref" >/dev/null 2>&1 || true
        fi
    done
}

cleanup_repo "open-stt-server-debian" "${OPEN_STT_IMAGE}"

docker image prune -f >/dev/null
docker container prune -f >/dev/null
docker builder prune -af --filter "until=${prune_age}" >/dev/null

echo "[docker-build] completed tag ${tag}"
