#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
BUILD_DIR=${BUILD_DIR:-"$PROJECT_ROOT/artifacts/java_build"}
MAIN_CLASS=${MAIN_CLASS:-NonUniformPrior}
DATA_ROOT=${DATA_ROOT:-"$PROJECT_ROOT/data"}
WORK_ROOT=${WORK_ROOT:-"$PROJECT_ROOT/artifacts/java_work"}

default_output_dir() {
  case "$1" in
    NonUniformPrior)
      printf '%s\n' "$DATA_ROOT/Krebs_Cycle_Normalised_3_TS"
      ;;
    Main)
      printf '%s\n' "$DATA_ROOT/Krebs_Cycle_3_TS"
      ;;
    GroundTruthGraph)
      printf '%s\n' "$DATA_ROOT"
      ;;
    *)
      printf '%s\n' "$DATA_ROOT/$1"
      ;;
  esac
}

OUTPUT_DIR=${OUTPUT_DIR:-"$(default_output_dir "$MAIN_CLASS")"}

default_target_file() {
  case "$1" in
    GroundTruthGraph)
      printf '%s\n' "$DATA_ROOT/krebsgenerator_groundtruth_lag_edges.txt"
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

TARGET_FILE=${TARGET_FILE:-"$(default_target_file "$MAIN_CLASS")"}

default_work_dir() {
  case "$1" in
    GroundTruthGraph)
      printf '%s\n' "$WORK_ROOT/$1"
      ;;
    *)
      printf '%s\n' "$OUTPUT_DIR"
      ;;
  esac
}

WORK_DIR=${WORK_DIR:-"$(default_work_dir "$MAIN_CLASS")"}

if [ -d "/usr/local/opt/openjdk/bin" ]; then
  export PATH="/usr/local/opt/openjdk/bin:$PATH"
fi

if ! command -v java >/dev/null 2>&1; then
  echo "java was not found on PATH." >&2
  exit 1
fi

if ! command -v javac >/dev/null 2>&1 || ! javac -version >/dev/null 2>&1; then
  echo "javac was not found on PATH. Install a JDK before running the generator." >&2
  exit 1
fi

mkdir -p "$BUILD_DIR" "$OUTPUT_DIR" "$WORK_DIR"

javac -d "$BUILD_DIR" \
  "$SCRIPT_DIR"/*.java \
  "$SCRIPT_DIR"/betterChemicalReactions/*.java

cp "$SCRIPT_DIR/challenge.txt" "$WORK_DIR/challenge.txt"

(
  cd "$WORK_DIR"
  java -cp "$BUILD_DIR" "$MAIN_CLASS"
)

if [ "$MAIN_CLASS" = "GroundTruthGraph" ]; then
  if [ -n "$TARGET_FILE" ] && [ -f "$WORK_DIR/groundtruth.txt" ]; then
    mv "$WORK_DIR/groundtruth.txt" "$TARGET_FILE"
  fi
  rm -f "$WORK_DIR/challenge.txt"
  echo "Generated file: ${TARGET_FILE:-$WORK_DIR/groundtruth.txt}"
else
  echo "Generated files in $OUTPUT_DIR"
fi
