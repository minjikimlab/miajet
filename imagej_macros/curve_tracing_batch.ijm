/*
 * Macro to process a single image using Steger's curve tracing.
 *
 * Expected arguments (from getArgument()):
 *   A single string containing three parts separated by "|||":
 *     1. A sigma value (as a string)
 *     2. The full path to the input image
 *     3. The output (save) directory for the results.
 *     4. The upper threhsold ut encoded as a str i.e. str(ut)
 *     4. The lower threhsold lt encoded as a str
 *     5. The root encoded as a str
 */

// Retrieve the argument string passed from Python (or other source).
arg = getArgument();
if (arg == "") {
    exit("No arguments provided");
}

// Split the argument string using the unique delimiter "|||".
tokens = split(arg, "|||");
if (tokens.length < 6) {
    exit("Insufficient arguments provided. Expected: sigma|||input_image|||save_path|||ut|||lt|||root");
}

sigma_str   = tokens[0];
input_image = tokens[1];
save_path   = tokens[2];
ut_value    = tokens[3];  // new: upper threshold value
lt_value    = tokens[4];  // new: lower threshold value
root        = tokens[5];  // new: root

sigma = parseFloat(sigma_str);
sigma_rounded = d2s(sigma, 3);

open(input_image);

// img_title = getTitle();
// img_title = substring(img_title, 0, lengthOf(img_title) - 4);

runCurveTracing(sigma, ut_value, lt_value, root, save_path);

function runCurveTracing(sigma, ut_value, lt_value, root, baseDir) {

    run("Source Steger's Algorithm", 
        "detection=[White lines on dark background] "
        + "line=" + sigma + " "
        + "maximum=2.5 "
        + "split minimum=0 correct compute "
        + "maximum_0=2.5 "
        + "add=[Only lines] color=Rainbow "
        + "upper=" + ut_value + " lower=" + lt_value
    );

    saveAs("PNG", baseDir + File.separator + root + "_imagej_results_s-" + sigma_rounded + "_lt-" + lt_value + "_ut-" + ut_value + ".png");
    
    saveAs("Results", baseDir + File.separator + root + "_imagej_results_s-" + sigma_rounded + "_table.csv");

    // Need the following command to exit the console!
    eval("script", "System.exit(0);");
}
