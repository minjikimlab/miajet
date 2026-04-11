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

# for i in "${!FILES[@]}"; do
#     sbatch  --export=HIC_FILE="${FILES[$i]}",CHROM="chrS",NORM="NONE",RES="25000",EXP="hic",WIN="5000000" \
#       job_all_noproc.sh
# done

for i in "${!FILES[@]}"; do
    sbatch  --export=HIC_FILE="${FILES[$i]}" \
      job_simulation.sh
done