/*
 * Macro to process a single image using Steger's curve tracing.
 *
 * Expected arguments (from getArgument()):
 *   A single string containing three parts separated by "|||":
 *     1. A sigma value (as a string)
 *     2. The full path to the input image
 *     3. The output (save) directory for the results.
 */

// Retrieve the argument string passed from Python (or other source).
arg = getArgument();
if (arg == "") {
    exit("No arguments provided. Expected: sigma|||input_image|||save_path");
}

// Split the argument string using the unique delimiter "|||".
tokens = split(arg, "|||");
if (tokens.length < 3) {
    exit("Insufficient arguments provided. Expected: sigma|||input_image|||save_path");
}

sigma_str   = tokens[0];
input_image = tokens[1];
save_path   = tokens[2];

sigma = parseFloat(sigma_str);

open(input_image);

img_title = getTitle();
img_title = substring(img_title, 0, lengthOf(img_title) - 4);

runCurveTracing(sigma, img_title, save_path);

function runCurveTracing(sigma, root, baseDir) {

    // run("Ridge Detection", "line_width=[] high_contrast=[] low_contrast=[] correct_position estimate_width method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=" + lowerThreshold + " upper_threshold=" + upperThreshold + " minimum_line_length=0 maximum=0");
    run("Ridge Detection", "line_width=[] high_contrast=[] low_contrast=[] correct_position estimate_width show_junction_points displayresults method_for_overlap_resolution=NONE sigma=" + sigma + " lower_threshold=0.05 upper_threshold=0.1 minimum_line_length=0 maximum=0");

    // saveAs("PNG", baseDir + File.separator + root + "_s-" + sigma + ".png");
    
    saveAs("Results", baseDir + File.separator + root + "_s-" + sigma + "_results.csv");

    // Need the following command to exit the console!
    eval("script", "System.exit(0);");
}
