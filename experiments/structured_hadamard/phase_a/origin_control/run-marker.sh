#!/bin/bash
printf 'token=origin-control-marker-r1-6c6043979a4a4a32 slurm_job_id=%s hostname=%s\n' "$SLURM_JOB_ID" "$HOSTNAME"
exit 0
