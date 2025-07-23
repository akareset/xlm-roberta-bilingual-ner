#!/bin/bash

#SBATCH --job-name=xlm-roberta-build-and-run
#SBATCH --output=xlm_roberta_%A.out
#SBATCH --error=xlm_roberta_%A.err
#SBATCH --time=24:00:00  # Increased time to account for building + running
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antoni.mackowiak@stud-mail.uni-wuerzburg.de
#SBATCH -p h100
#SBATCH -c 16
#SBATCH --tmp=300G

# Properly pass GPU device information to the container
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

#If SLURM_TMPDIR wasn't set, fall back to the $TMPDIR that Slurm creates under /var/tmp
export SLURM_TMPDIR=${SLURM_TMPDIR:-$TMPDIR}

# Use unique Apptainer tmp and cache per job to avoid sharing
export APPTAINER_TMPDIR=${SLURM_TMPDIR}/apptainer_tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
export APPTAINER_CACHEDIR=${SLURM_TMPDIR}/apptainer_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
export BUILD_DIR=${SLURM_TMPDIR}/apptainer_build_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
export IMAGE_DIR=${SLURM_TMPDIR}/singularity_images_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}

# Create cache directory in the temporary space
export CACHE_DIR=${SLURM_TMPDIR}/model_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}

# Logs are small — send them to your large /data scratch
export LOG_DIR=/home/s478234/logs_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}

mkdir -p \
  $APPTAINER_TMPDIR \
  $APPTAINER_CACHEDIR \
  $BUILD_DIR \
  $IMAGE_DIR \
  $LOG_DIR \
  $CACHE_DIR

echo "Using fast node-local NVMe: $SLURM_TMPDIR"
echo "Writing logs to $LOG_DIR"
echo "Using cache directory: $CACHE_DIR"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_PROJECT="XLM-RoBERTa-Continual-Pretraining"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# ---------- BUILD PHASE ----------

echo "Starting container build phase..."

# Create temporary directory for building
cd "$BUILD_DIR"

# Copy necessary files
cp "${SLURM_SUBMIT_DIR}/requirements.txt" .
cp "${SLURM_SUBMIT_DIR}/main.py" .
if [ -f "${SLURM_SUBMIT_DIR}/.env" ]; then
    cp "${SLURM_SUBMIT_DIR}/.env" .
fi

# Generate unique tag for this build
export UNIQUE_TAG=$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}

# Define the experiment
EXPERIMENT_NAME="pytorch_2.7.1_cuda11.8"
PYTORCH_VERSION="2.7.1"
CUDA_VERSION="11.8"
CUDNN_VERSION="9"

echo "Building Apptainer image for $EXPERIMENT_NAME"
echo "PyTorch: $PYTORCH_VERSION, CUDA: $CUDA_VERSION, cuDNN: $CUDNN_VERSION"

# Create a definition file for this experiment
cat > "${EXPERIMENT_NAME}_${UNIQUE_TAG}.def" <<EOL
Bootstrap: docker
From: pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

%files
    requirements.txt /app/requirements.txt
    main.py /app/main.py

%post
    apt-get update && apt-get install -y \\
        git \\
        build-essential \\
        wget
    
    # Copy requirements and install them
    mkdir -p /app
    cd /app
    
    # Install requirements
    pip install -r requirements.txt
    
    # Create logs and cache directories
    mkdir -p /app/logs /app/cache /app/checkpoints

%environment
    export PYTHONUNBUFFERED=1
    export EXPERIMENT_NAME=${EXPERIMENT_NAME}
    export GIT_PYTHON_REFRESH=quiet
    export WANDB_PROJECT="XLM-RoBERTa-Continual-Pretraining"

%labels
    Author Antoni Mackowiak
    Purpose XLM-RoBERTa Bilingual MLM Training
EOL

echo "Starting building container for ${EXPERIMENT_NAME}"
# Build the Apptainer image
srun --exclusive -c ${SLURM_CPUS_PER_TASK:-1} \
  apptainer --debug build "${IMAGE_DIR}/${EXPERIMENT_NAME}_${UNIQUE_TAG}.sif" "$BUILD_DIR/${EXPERIMENT_NAME}_${UNIQUE_TAG}.def"

# Check if build was successful
if [ ! -f "${IMAGE_DIR}/${EXPERIMENT_NAME}_${UNIQUE_TAG}.sif" ]; then
  echo "Failed to build image for ${EXPERIMENT_NAME}. Exiting."
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "Container built successfully"

# ---------- EXPERIMENT PHASE ----------

echo "Starting training phase..."

CONTAINER="${IMAGE_DIR}/${EXPERIMENT_NAME}_${UNIQUE_TAG}.sif"

# Check if the container file exists
if [ ! -f "$CONTAINER" ]; then
  echo "Container $CONTAINER does not exist, exiting."
  exit 1
fi

echo "Running XLM-RoBERTa bilingual MLM training"

# Check GPU availability
nvidia-smi

srun apptainer exec --nv \
  --bind $PWD:/app/workspace \
  --bind ${LOG_DIR}:/app/logs \
  --bind ${CACHE_DIR}:/app/cache \
  --bind ${CACHE_DIR}:/app/checkpoints \
  $(if [ -f "$PWD/.env" ]; then echo "--bind $PWD/.env:/app/.env"; fi) \
  --env HUGGINGFACE_HUB_CACHE=/app/cache/huggingface \
  --env HF_HOME=/app/cache/huggingface \
  --env TRANSFORMERS_CACHE=/app/cache/huggingface/transformers \
  --env HF_DATASETS_CACHE=/app/cache/huggingface/datasets \
  --env TORCH_HOME=/app/cache/torch \
  --env WANDB_PROJECT="XLM-RoBERTa-Continual-Pretraining" \
  --env GIT_PYTHON_REFRESH=quiet \
  --env PYTHONUNBUFFERED=1 \
  "$CONTAINER" bash -c "mkdir -p /app/cache/huggingface /app/cache/torch /app/cache/tmp && cd /app && python main.py"

echo "Training completed"
echo "Job finished at: $(date)"