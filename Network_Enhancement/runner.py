import os
import subprocess
from scipy.io import savemat, loadmat

def enhance_network(A, order, num_neighbors, alpha, workdir, ne_path, matlab_module="matlab/R2024b"):
    os.makedirs(workdir, exist_ok=True)
    savemat(os.path.join(workdir, "input.mat"), {"A": A})

    cmd = f"""
    module load {matlab_module} && \
    matlab -nodisplay -nosplash -batch "try; \
      addpath(genpath('{ne_path}')); \
      S = load('input.mat'); \
      E = Network_Enhancement(S.A,{order},{num_neighbors},{alpha}); \
      save('output.mat','E'); \
    catch ME; \
      disp(getReport(ME)); \
      exit(1); \
    end;"
    """
    subprocess.run(cmd, shell=True, cwd=workdir, check=True, executable="/bin/bash")

    data = loadmat(os.path.join(workdir, "output.mat"))
    return data["E"]
