now=$(date +"%Y%m%d_%H%M%S")
gcloud ai-platform jobs submit training rsna_inception_$now \
    --stream-logs \
    --runtime-version 1.14 \
    --job-dir gs://gcc-models \
    --package-path trainer \
    --module-name trainer.task \
    --region us-central1 \
    --python-version 3.5 \
    --config config.yaml
