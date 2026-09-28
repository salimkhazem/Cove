#!/usr/bin/env bash
# Oracle curves for every reported (dataset, encoder, score); skips existing outputs.
# Usage: oracle_all.sh GPU [reverse]
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=$1
specs="imagenet:dinov2_b:raw imagenet:clip_b16:raw imagenet:resnet50:raw sop:dinov2_b:raw sop:clip_b16:raw sop:resnet50:raw cub:dinov2_b:raw cub:clip_b16:raw cub:resnet50:raw cars:dinov2_b:raw cars:clip_b16:raw cars:resnet50:raw inshop:dinov2_b:raw inshop:clip_b16:raw inshop:resnet50:raw gldv2mini:dinov2_b:raw gldv2mini:clip_b16:raw gldv2mini:resnet50:raw sop:dinov2_s:raw sop:dinov2_l:raw sop:clip_l14:raw sop:siglip_b16:raw sop:convnext_b:raw imagenet:dinov2_s:raw imagenet:dinov2_l:raw imagenet:clip_l14:raw imagenet:siglip_b16:raw imagenet:convnext_b:raw sop:dinov2_b:csls sop:clip_b16:csls sop:resnet50:csls sop:dinov2_b:snorm sop:clip_b16:snorm sop:resnet50:snorm imagenet:dinov2_b:csls imagenet:clip_b16:csls imagenet:resnet50:csls imagenet:dinov2_b:snorm imagenet:clip_b16:snorm imagenet:resnet50:snorm"
[ "${2:-}" = reverse ] && specs=$(echo $specs | tr ' ' '\n' | tac | tr '\n' ' ')
[ "${2:-}" = middle ] && specs=$(echo $specs | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')
for spec in $specs; do
  IFS=: read -r ds enc sc <<< "$spec"
  out=results/raw/oracle/${ds}__${enc}__${sc}.json
  [ -f "$out" ] && continue
  lock=results/raw/oracle/.${ds}__${enc}__${sc}.lock
  mkdir -p results/raw/oracle; ( set -o noclobber; echo $$ > "$lock" ) 2>/dev/null || continue
  case $ds in sop|inshop) ms="30 100 300 1000 3000 10000";; gldv2mini) ms="30 100 300 1000";; imagenet) ms="10 30 100 300";; *) ms="10 30 60";; esac
  python scripts/oracle_curve.py --dataset $ds --encoder $enc --score $sc --ms $ms
  rm -f "$lock"
done
