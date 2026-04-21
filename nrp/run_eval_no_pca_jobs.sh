#!/bin/bash
# Submit NRP eval jobs with --raw_md_axis2 (no PCA at eval time) for multiple checkpoints.
# Usage:
#   bash nrp/run_eval_no_pca_jobs.sh           # submit all jobs
#   bash nrp/run_eval_no_pca_jobs.sh --dry-run  # print generated YAMLs only

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

BASE=/axovol/contrastive_axis

# Format: "k8s-job-name|checkpoint-filename|outdir-name|wandb-run-name"
JOBS=(
  # Apr 15 runs
  "eval-nopca-supcon-ce-nopca-0415a|embedding_hlt_linformer_supcon_ce_no_pca_encoder_20260415_172538.pth|abcd_outputs_supcon_ce_no_pca_0415a_raw_md|supcon_ce_no_pca_0415a_raw_md"
  "eval-nopca-supcon-ce-nopca-bs2048-0415|embedding_hlt_linformer_supcon_ce_no_pca_bs2048_encoder_20260415_222604.pth|abcd_outputs_supcon_ce_no_pca_bs2048_0415_raw_md|supcon_ce_no_pca_bs2048_0415_raw_md"
  # Apr 17 run
  "eval-nopca-supcon-ce-nopca-0417|embedding_hlt_linformer_supcon_ce_no_pca_encoder_20260417_013044.pth|abcd_outputs_supcon_ce_no_pca_0417_raw_md|supcon_ce_no_pca_0417_raw_md"
  # Apr 20/21 runs
  "eval-nopca-supcon-ce-nopca-0420|embedding_hlt_linformer_supcon_ce_no_pca_encoder_20260420_235818.pth|abcd_outputs_supcon_ce_no_pca_0420_raw_md|supcon_ce_no_pca_0420_raw_md"
  "eval-nopca-bs4096-lowdisco-0421|embedding_hlt_linformer_supcon_ce_no_pca_bs4096_lowdisco_encoder_20260421_040204.pth|abcd_outputs_supcon_ce_no_pca_bs4096_lowdisco_0421_raw_md|supcon_ce_no_pca_bs4096_lowdisco_0421_raw_md"
  "eval-nopca-1024model-lowdisco-0421|embedding_hlt_linformer_supcon_ce_no_pca_1024model_lowdisco_encoder_20260421_053859.pth|abcd_outputs_supcon_ce_no_pca_1024model_lowdisco_0421_raw_md|supcon_ce_no_pca_1024model_lowdisco_0421_raw_md"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r JOB_NAME CKPT OUTDIR WANDB_NAME <<< "$entry"
  echo "==> $JOB_NAME  ($CKPT)"

  YAML=$(cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: axol1tl
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: eval
          image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -e
              echo "=== Job started: \$(date) ==="
              echo "Host: \$(hostname)"

              export PYTHONPATH=$BASE/pylib:\$PYTHONPATH

              cd $BASE/modularized_nurd_hlt_con_ae
              pip install --quiet --force-reinstall --target $BASE/pylib -e . 2>/dev/null || true

              python eval_abcd.py --contrast_ckpt $BASE/modularized_nurd_hlt_con_ae/checkpoints/$CKPT --contrast_test_pt $BASE/data/hlt_smcocktail_test.pt --signal_pt $BASE/data/signal_pt/hlt_signal_TpTp.pt --outdir $BASE/abcd_outputs/$OUTDIR --raw_md_axis2 --wandb_run_name $WANDB_NAME

              echo "=== Job finished: \$(date) ==="

          env:
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-secret
                  key: WANDB_API_KEY

          resources:
            requests:
              memory: "16Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
            limits:
              memory: "32Gi"
              cpu: "8"
              nvidia.com/gpu: "1"

          volumeMounts:
            - name: axovol
              mountPath: /axovol

      volumes:
        - name: axovol
          persistentVolumeClaim:
            claimName: axovol
EOF
)

  if $DRY_RUN; then
    echo "$YAML"
    echo "---"
  else
    echo "$YAML" | kubectl apply -f - -n axol1tl
  fi
done
