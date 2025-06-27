import os
import subprocess
import imagej
from multiprocess import Pool 
# import scyjava
import psutil
import time
import random
import sys


def get_free_display(low=1000, high=9999, max_tries=10):
    """
    Pick a random display number in [low..high], skipping any that's
    already in use under /tmp/.X11-unix. Retries up to max_tries times.

    Returns an integer display number (e.g., 1234 for ":1234").

    Raises RuntimeError if unable to find a free display after max_tries.
    """
    for i in range(max_tries):
        candidate = random.randint(low, high)
        # If the X socket file doesn't exist, we assume it's unused:
        if not os.path.exists(f"/tmp/.X11-unix/X{candidate}"):
            print(f"\tFound XVFB display resolving {i} conflicts")
            return candidate
        
    raise RuntimeError(f"No free display found in {max_tries} tries.")



# def process_sigma_headless(s, lt, ut, image_path, save_path, root, memory_alloc, macro_path, verbose):
#     s_str = str(s)
#     max_attempts = 5
#     timeout_seconds = 120
#     attempt = 0
#     current_ut, current_lt = ut, lt
#     t0 = time.time()

#     if verbose:
#         print(f"\tHeadless Fiji: max_attempts={max_attempts}, initial timeout={timeout_seconds}s")

#     while attempt < max_attempts:
#         args = "|||".join([s_str, image_path, save_path, str(current_ut), str(current_lt), root])
#         fiji_cmd = [
#             "/nfs/turbo/umms-minjilab/sionkim/finding_jets/localfiji/ImageJ-linux64",
#             "--headless",
#             "--console",
#             f"--mem={int(memory_alloc / (1024 * 1024))}M",
#             "-macro", macro_path,
#             args
#         ]

#         try:
#             if verbose:
#                 print(f"\tAttempt {attempt+1}: s={s:.2f}, ut={current_ut:.2f}, lt={current_lt:.2f}")
#             subprocess.run(fiji_cmd, timeout=timeout_seconds, check=True)
#             if verbose:
#                 print(f"\tFiji finished for s={s:.2f} in {time.time() - t0:.0f}s")
#             break

#         except subprocess.TimeoutExpired:
#             print(f"\tTimeout after {timeout_seconds}s on attempt {attempt+1} (s={s:.2f})")
#             # increase thresholds then retry
#             current_ut *= 1.5
#             current_lt *= 1.5
#             attempt += 1
#             timeout_seconds += 30
#             if verbose:
#                 print(f"\tRetrying with ut={current_ut:.2f}, lt={current_lt:.2f}, new timeout={timeout_seconds}s")

#         except subprocess.CalledProcessError as e:
#             print(f"\tFiji failed with exit code {e.returncode}")
#             sys.exit(e.returncode)

#     else:
#         print("All attempts exhausted; please increase your thresholds or timeout")
#         sys.exit(1)

from pyvirtualdisplay import Display

def process_sigma_pyvd(s, lt, ut, image_path, save_path, root, memory_alloc, macro_path, verbose):
    s_str = str(s)
    max_attempts = 5
    timeout_seconds = 120
    attempt = 0
    current_ut, current_lt = ut, lt
    t0 = time.time()

    if verbose:
        print(f"\tFiji-with-Xvfb: max_attempts={max_attempts}, initial timeout={timeout_seconds}s")

    while attempt < max_attempts:
        args = "|||".join([s_str, image_path, save_path,
                           str(current_ut), str(current_lt), root])
        
        if verbose:
            print(f"\tAttempt {attempt+1}: s={s:.2f}, ut={current_ut:.2f}, lt={current_lt:.2f}")

        # start an offscreen X server
        with Display(visible=False, backend="xvfb", size=(100, 100), color_depth=24) as disp:

            try:
                fiji_cmd = [
                    "/nfs/turbo/umms-minjilab/sionkim/finding_jets/localfiji/ImageJ-linux64",
                    "--forbid-single-instance",
                    "-Dij.no-legacy-single-instance=true",
                    f"--mem={int(memory_alloc / (1024 * 1024))}M",
                    # "-batch",
                    "-macro", macro_path,
                    # macro_path,
                    args
                ]

                subprocess.run(fiji_cmd, timeout=timeout_seconds, check=True)

                if verbose:
                    elapsed = time.time() - t0
                    print(f"\tFiji finished for s={s:.2f} in {elapsed:.0f}s")

                break

            except subprocess.TimeoutExpired:
                print(f"\tTimeout after {timeout_seconds}s on attempt {attempt+1} (s={s:.2f})")
                current_ut *= 1.5
                current_lt *= 1.5
                attempt += 1
                timeout_seconds += 30
                if verbose:
                    print(f"\tRetrying with ut={current_ut:.2f}, lt={current_lt:.2f}, new timeout={timeout_seconds}s")

            except subprocess.CalledProcessError as e:
                print(f"\tFiji failed with exit code {e.returncode}")
                sys.exit(e.returncode)

    else:
        print("All attempts exhausted; please increase your thresholds or timeout.")
        sys.exit(1)


from scyjava import jimport

def process_sigma_pyimagej(
    s, lt, ut,
    image_path, save_path, root,
    memory_alloc,  # in bytes
    macro_path,
    verbose=True
):
    """
    Fully headless via PyImageJ. Retries up to 5× on any exception,
    bumping thresholds by 1.5× each time.
    """
    # 1) Convert memory_alloc bytes → megabytes for Fiji's --mem=
    mem_mb = int(memory_alloc // (1024 * 1024))
    
    # 2) Launch Fiji headless with that heap
    ij = imagej.init('/nfs/turbo/umms-minjilab/sionkim/miajet/localfiji') 
    # ij = imagej.init('sc.fiji:fiji')
    
    # 2) Build a little IJ macro (as a string) that takes parameters
    macro = f"""
#@ Float sigma
#@ Float upper
#@ Float lower
#@ String imagePath
#@ String saveDir
#@ String root
#@output Object dummy

open(imagePath);
run("Source Steger's Algorithm",
    "detection=[White lines on dark background] "
    + "line=" + sigma
    + " maximum=2.5 split minimum=0 correct compute "
    + "maximum_0=2.5 add=[Only lines] color=Rainbow "
    + "upper=" + upper + " lower=" + lower
);
saveAs("PNG",
    saveDir + "/" + root
    + "_imagej_results_s-" + d2s(sigma,3)
    + "_lt-" + d2s(lower,3)
    + "_ut-" + d2s(upper,3)
    + ".png"
);
saveAs("Results",
    saveDir + "/" + root
    + "_imagej_results_s-" + d2s(sigma,3)
    + "_table.csv"
);
dummy = "ok";
"""

    # 3) Prepare the Python-side args dict
    args = {
        'sigma': float(s),
        'upper': float(ut),
        'lower': float(lt),
        'imagePath': image_path,
        'saveDir': save_path,
        'root': root
    }

    max_attempts = 5
    timeout = 120
    attempt = 0
    start = time.time()

    # 4) Try up to max_attempts
    while attempt < max_attempts:
        if verbose:
            print(f"[Attempt {attempt+1}] sigma={s:.2f}, upper={ut:.2f}, lower={lt:.2f}, timeout={timeout}s")

        try:
            # run_macro blocks until the macro finishes or errors
            ij.py.run_macro(macro, args)
            if verbose:
                print(f"✔ Done in {time.time() - start:.1f}s")
            return

        except Exception as e:
            attempt += 1
            ut *= 1.5
            lt *= 1.5
            timeout += 30
            if verbose:
                print(f"⚠ Run failed: {e!r}")
                print(f"↻ Retrying with upper={ut:.2f}, lower={lt:.2f}, timeout={timeout}s")

    # 5) If we get here, we ran out of retries
    raise RuntimeError(f"process_sigma failed after {max_attempts} attempts")




def process_sigma(s, lt, ut, image_path, save_path, root, memory_alloc, macro_path, verbose):
    """
    Helper function to run ImageJ in parallel for each scale.

    If the ImageJ process hangs (i.e. doesn't finish within the timeout), increase the thresholds and try again
    There is a silent bug in ImageJ that causes it to hang if the thresholds are too low (i.e. too many ridges)

    Parameters:
    s: scale (sigma) to process
    lt: lower threshold for the hysteresis thresholding scheme used by the Curve Tracing plugin
    ut: upper threshold for the hysteresis thresholding scheme used by the Curve Tracing plugin
    image_path: path to the input image to process
    save_path: path to the directory where the output images and results will be saved
    root: root name for the output files
    memory_alloc: amount of RAM allocated for ImageJ in Bytes
    macro_path: path to the ImageJ macro file
    verbose: whether to print verbose output (default: True)
    Returns:
    None, but saves the output images and (.csv) tables directly from the imageJ macro
    """
    s_str = str(s)
    max_attempts = 5 # 5 attempts to run Fiji with increasing thresholds
    timeout_seconds = 120  # wait at most 120 seconds for Fiji to complete 
    attempt = 0 # attempt counter
    current_ut = ut
    current_lt = lt
    t0 = time.time()

    print(f"\tImageJ Meta Parameters: max attempts: {max_attempts} timeout: {timeout_seconds}s")

    while attempt < max_attempts:
        # Build the argument string with the current thresholds
        args = s_str + "|||" + image_path + "|||" + save_path + "|||" + str(current_ut) + "|||" + str(current_lt) + "|||" + root

        # Check if results already exist
        # expected_image = os.path.join(save_path, f"image_s-{s_str}.png")
        # expected_csv = os.path.join(save_path, f"image_s-{s_str}_results.csv")
        # if os.path.exists(expected_csv):
        #     if verbose:
        #         print("Results already exist, skipping processing.")
        #     return  # Skip processing
        
        # if verbose:
        #     print(f"Attempt {attempt+1} with s={s:.2f}, upper={current_ut}, lower={current_lt}")

        # pick a display number for Xvfb
        server_num = get_free_display(low=1000, high=9999, max_tries=50)

        # 2) Start Xvfb in the background
        xvfb_cmd = [
            "Xvfb",
            f":{server_num}",
            "-screen", "0", "100x100x8"
        ]
        xvfb_proc = subprocess.Popen(
            xvfb_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # point display to the Xvfb server
        os.environ["DISPLAY"] = f":{server_num}"

        # wait for Xvfb to start
        time.sleep(2)

        # fiji command
        fiji_cmd = [
            "/nfs/turbo/umms-minjilab/sionkim/finding_jets/localfiji/ImageJ-linux64",
            "-Dij.no-legacy-single-instance=true",   
            f"--mem={int(memory_alloc / (1024 * 1024))}M", # convert from Bytes to MB
            "-macro",
            macro_path,
            args
        ]

        try:
            subprocess.run(fiji_cmd, timeout=timeout_seconds)
            # If Fiji finishes before the timeout, break out of the loop
            if verbose:
                print(f"Fiji finished successfully for s={s:.2f} in {time.time() - t0:.0f}s.")
            xvfb_proc.terminate()
            xvfb_proc.wait()
            break

        except subprocess.TimeoutExpired:
            # Timeout occurred
            print(f"Attempt {attempt+1} for s={s:.2f}: Fiji process timed out after {timeout_seconds} seconds "
                  f"with thresholds upper={current_ut} and lower={current_lt}.")
            xvfb_proc.terminate()
            xvfb_proc.wait()
            # Increment the thresholds to avoid ImageJ silent crash
            current_ut *= 1.5
            current_lt *= 1.5
            attempt += 1
            timeout_seconds += 30  # increase timeout for next attempt by 30s
            print(f"Retrying with new thresholds for s={s:.2f}: upper={current_ut}, lower={current_lt}")

    else:
        print("Please increase the `thresholds` parameters")
        sys.exit(1)

# import glob
# import getpass

# def process_sigma(s, lt, ut, image_path, save_path, memory_alloc, macro_path, verbose):
#     """
#     Run Fiji/ImageJ with the given sigma and threshold parameters.
#     If the Fiji process hangs (i.e. doesn't finish within the timeout),
#     increment the thresholds and try again.
#     """
#     s_str = f"{s:.3f}"  # round sigma to 3 decimal places
#     max_attempts = 5
#     timeout_seconds = 120  # wait at most 120 seconds for Fiji to complete 
#     attempt = 0
#     current_ut = ut
#     current_lt = lt
#     t0 = time.time()

#     print(f"\tImageJ Meta Parameters: max attempts: {max_attempts} timeout: {timeout_seconds}s")

#     while attempt < max_attempts:
#         # Build the argument string with the current thresholds.
#         args = s_str + "|||" + image_path + "|||" + save_path + "|||" + str(current_ut) + "|||" + str(current_lt)

#         # Check if results already exist 
#         # expected_image = os.path.join(save_path, f"image_s-{s_str}.png")
#         # expected_csv = os.path.join(save_path, f"image_s-{s_str}_results.csv")
#         # if os.path.exists(expected_csv):
#         #     if verbose:
#         #         print("Results already exist, skipping processing.")
#         #     return  # Skip processing
        
#         if verbose:
#             print(f"Attempt {attempt+1} with s={s:.2f}, upper={current_ut}, lower={current_lt}")

#         # remove any stale stubs
#         username = getpass.getuser()
#         for stub_path in glob.glob(f"/tmp/ImageJ-{username}-_*"):
#             try:
#                 os.remove(stub_path)
#                 if verbose:
#                     print(f"\tRemoved stale stub: {stub_path}")
#             except OSError:
#                 pass


#         os.environ["JAVA_TOOL_OPTIONS"] = "-Dij.no-legacy-single-instance=true"
#         fiji_cmd = [
#             "xvfb-run", "--auto-servernum",
#             "--server-args=-screen 0 100x100x8",
#             "/nfs/turbo/umms-minjilab/sionkim/finding_jets/localfiji/ImageJ-linux64",
#             f"--mem={int(memory_alloc / (1024 * 1024))}M", # convert from Bytes to MB
#             "-macro",
#             macro_path,
#             args
#         ]

#         try:
#             subprocess.run(fiji_cmd, timeout=timeout_seconds)
#             # If Fiji finishes before the timeout, break out of the loop.
#             if verbose:
#                 print(f"Fiji finished successfully for s={s:.2f} in {time.time() - t0:.0f}s.")
#             break

#         except subprocess.TimeoutExpired:
#             # Timeout occurred
#             print(f"Attempt {attempt+1} for s={s:.2f}: Fiji process timed out after {timeout_seconds} seconds "
#                   f"with thresholds upper={current_ut} and lower={current_lt}.")
#             current_ut *= 1.5 # increase the thresholds to avoid imageJ silent crash
#             current_lt *= 1.5
#             attempt += 1
#             timeout_seconds += 30 # increase timeout for next attempt by 30s
#             print(f"Retrying with new thresholds for s={s:.2f}: upper={current_ut}, lower={current_lt}")

#     else:
#         print("Please increase the `thresholds` parameters")
#         sys.exit(1)



def allocate_mem(frac, verbose):
    """
    Finds the available RAM using psutil.virtual_memory() and allocates
    a fraction of it for ImageJ. 

    Parameters:
    -------------
    frac: The fraction of available RAM to allocate for ImageJ

    Returns:
    -------------
    memory_alloc: The amount of RAM allocated for ImageJ in Bytes
    """
    memory = psutil.virtual_memory()
    memory_alloc = memory.available * frac
    if verbose:
        print(f"\tAvailable total RAM: {memory.available / (1024 ** 3) :.2f}GB")
        print(f"\tAllocating {memory_alloc / (1024 ** 3) :.2f}GB to ImageJ")
    return memory_alloc
    

def call_imagej_scale_space(scale_range, lt, ut, image_path, save_path, root, num_cores=1, verbose=True):
    """
    Runs an ImageJ macro stored in './imagej_macros/curve_tracing_batch.ijm' for each scale in scale_range
    in parallel if num_cores > 1, otherwise runs sequentially

    ImageJ is assumed to be installed in './localfiji/ImageJ-linux64' and a virtual display server (Xvfb)
    is used to run ImageJ, as opposed to headless mode which fails to run the Curve Tracing plugin

    Unfortunately, the automatic display number assignment does not work (i.e. xvfb-run --auto-servernum)
    so we manually assign a display number for each process, which is picked randomly in the range [1000..9999]
    Notably, there is a chance of conflict if multiple processes are started at the same time, but this is rare.
    
    Each Xvfb process is initialized with a memory allocation of 50% of available RAM detemined by psutil.virtual_memory()

    Parameters:
    scale_range: list of scales (sigmas) to process
    lt: lower threshold for the hysteresis thresholding scheme used by the Curve Tracing plugin
    ut: upper threshold for the hysteresis thresholding scheme used by the Curve Tracing plugin
    image_path: path to the input image to process
    save_path: path to the directory where the output images and results will be saved
    root: root name for the output files
    num_cores: number of cores to use for parallel processing (default: 1)
    verbose: whether to print verbose output (default: True)

    Returns:
    None, but saves the output images and (.csv) tables directly from the imageJ macro
    """
    memory_alloc = allocate_mem(0.5, verbose)

    # Find the ImageJ macro file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, '..'))
    macro_path = os.path.join(project_dir, 'imagej_macros', 'curve_tracing_batch.ijm')  

    if verbose: print("\tProcessing scale:", end=" ")

    if num_cores == 1: 
        # print("\tWARNING: ImageJ parallelization DISABLED")
        # no parallelization
        for s in scale_range:
            process_sigma_pyvd(s, lt, ut, image_path, save_path, root, memory_alloc, macro_path, verbose)
            # process_sigma_pyimagej(s, lt, ut, image_path, save_path, root, memory_alloc, macro_path, verbose)
            # process_sigma(s, lt, ut, image_path, save_path, root, memory_alloc, macro_path, verbose)
            # process_sigma_python(s, image_path, save_path, memory_alloc, macro_path, verbose)

    else:
        # Allocate memory for each core
        args = [(s, lt, ut, image_path, save_path, root, memory_alloc / num_cores, macro_path, verbose) for s in scale_range]
        with Pool(int(num_cores)) as pool:
            # pool.starmap(process_sigma, args)
            pool.starmap(process_sigma_pyvd, args)
            # pool.starmap(process_sigma_pyimagej, args)

    print()


# def process_sigma_python(sigma, image_path, save_path, memory_alloc, macro_path, verbose=False):
#     """
#     Initializes ImageJ (Fiji) via pyimagej, builds a macro string embedding
#     sigma, image_path, and save_path, runs the macro, then disposes ImageJ.
#     """
#     # Initialize Fiji (this downloads Fiji if needed)
#     # scyjava.config.add_option(f'-Xmx{int(memory_alloc // 1e3)}g') 
#     ij = imagej.init("/nfs/turbo/umms-minjilab/sionkim/finding_jets/localfiji/", mode="gui")

    
#     # Extract the base name from the input image path
#     basename = os.path.splitext(os.path.basename(image_path))[0]
    
#     # Build the macro string.
#     # This version inlines the parameters so that we no longer use getArgument()
#     # but directly define sigma_str, input_image, and save_path.
#     macro = f'''
#     // Set parameters for this run
#     sigma_str = "{sigma}";
#     input_image = "{image_path}";
#     save_path = "{save_path}";
    
#     sigma = parseFloat(sigma_str);
#     open(input_image);
    
#     img_title = getTitle();
#     img_title = substring(img_title, 0, lengthOf(img_title) - 4);
    
#     runCurveTracing(sigma, img_title, save_path);
    
#     function runCurveTracing(sigma, root, baseDir) {{
#         run("Source Steger's Algorithm", 
#             "detection=[White lines on dark background] " +
#             "line=" + sigma + " " +
#             "maximum=2.5 " +
#             "minimum=0 correct compute " +
#             "maximum_0=2.5 " +
#             "add=[Only lines] color=Rainbow " +
#             "upper=0.1 lower=0.05");
    
#         saveAs("PNG", baseDir + File.separator + root + "_s-" + sigma_str + ".png");
#         saveAs("Results", baseDir + File.separator + root + "_s-" + sigma_str + "_results.csv");
    
#         // Exit the macro (if needed)
#         eval("script", "System.exit(0);");
#     }}
#     '''

#     # Run the macro using pyimageJ
#     ij.py.run_macro(macro)
#     # Clean up the imagej instance
#     ij.dispose()
