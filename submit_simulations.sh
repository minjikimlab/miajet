#!/bin/bash


FILES=(
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1b_s-23/out_fast_bounded_jets/r2.14/test1b_s-23_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1a_s-23/out_fast_unbounded_jets/r2.14/test1a_s-23-fast_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim/5e+05_0.8_0.6_0.2_0.75_0.33_0.33_s-23/out_fast_unbounded_jets/r2.14/chromsim1-fast-unbounded_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim/5e+05_0.8_0.6_0.2_0.75_0.33_0.33_s-23/out_fast_bounded_jets/r2.14/chromsim1-fast-bounded_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1a_s-23/out_slow_unbounded_jets/r2.14/test1a_s-23-slow_hic_003.hic" # Should be much smaller window size and highest resolution possible
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1c_s-23/out_fast_bounded_jets/r2.14/test1c_s-23_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1cii_s-23/out_fast_bounded_jets/r2.14/test1cii_s-23_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1e_s-23/out_tads/r2.14/test1e_s-23_hic_003.hic"
    "/nfs/turbo/umms-minjilab/sionkim/miajet_simulations/3DPolyS-LE/chrom_sim2/test1eii_s-23/out_tads/r2.14/test1eii_s-23_hic_003.hic"
)

# CHROM_SIZES=(
#     "/nfs/turbo/umms-minjilab/sionkim/Fun_modified/sim_chrom_size_40Mb.txt"
#     "/nfs/turbo/umms-minjilab/sionkim/Fun_modified/sim_chrom_size_40Mb.txt"
#     "/nfs/turbo/umms-minjilab/sionkim/Fun_modified/sim_chrom_size_20Mb.txt"
#     "/nfs/turbo/umms-minjilab/sionkim/Fun_modified/sim_chrom_size_20Mb.txt"
# )

# Old method before runtime monitoring
# for i in "${!FILES[@]}"; do
#     sbatch  --export=HIC_FILE="${FILES[$i]}" \
#       job_simulation.sh
# done


RUN_STAMP=$(date +%Y%m%d_%H%M%S)
JOBS_TSV="submitted_simulation_jobs_${RUN_STAMP}_$$.tsv"
printf 'jobid\tlabel\thic_file\n' > "$JOBS_TSV"

for i in "${!FILES[@]}"; do
    hic="${FILES[$i]}"
    sim_dir=$(basename "$(dirname "$(dirname "$hic")")")   # e.g. out_fast_bounded_jets
    base=$(basename "$hic" .hic) # e.g. test1b_s-23_hic_003
    label="${sim_dir}/${base}"

    jobid=$(sbatch --parsable \
        --job-name="sim_${sim_dir}_${base}" \
        --export=HIC_FILE="$hic" \
        job_simulation.sh)
    jobid=${jobid%%;*} 

    if [[ -n "$jobid" ]]; then
        printf '%s\t%s\t%s\n' "$jobid" "$label" "$hic" >> "$JOBS_TSV"
        echo "Submitted $label as job $jobid"
    else
        echo "WARNING: submission failed for $hic" >&2
    fi
done

echo "Wrote job records to $JOBS_TSV"