#!/usr/bin/env bash
set -euo pipefail

source_image="${SOURCE_IMAGE:-sparkring-glm53-official-spark:mtp-mapped-v14}"
destination="${DEST_IMAGE:?set DEST_IMAGE to an authenticated OCI registry tag}"

architecture="$(docker image inspect -f '{{.Architecture}}' "${source_image}")"
if [[ "${architecture}" != arm64 ]]; then
  echo "refusing to publish non-ARM image: ${architecture}" >&2
  exit 2
fi
if ! docker run --rm --entrypoint /bin/sh "${source_image}" -c \
  'test -r /opt/sparkring/nccl/libnccl.so.2'; then
  echo "source image lacks the patched NCCL runtime" >&2
  exit 2
fi

docker tag "${source_image}" "${destination}"
docker push "${destination}"
docker image inspect -f '{{json .RepoDigests}}' "${destination}"

