// Headless version of the original macro
// Usage:
// fiji --headless -macro script.ijm "/path/to/input.tif"
//
// Outputs are written to a sibling folder named after the input TIFF stem:
//   /path/to/input/input_Nucleus.tif
//   /path/to/input/input_green.tif
//   /path/to/input/input_red.tif
//   /path/to/input/input_purple.tif

setBatchMode(true);

// --------------------
// Input / output paths
// --------------------
path = getArgument();

if (path == "") {
    print("ERROR: Provide input .tif path as macro argument.");
    exit();
}

dir = File.getParent(path) + File.separator;
name = File.getName(path);
baseName = name;
if (endsWith(baseName, ".tif")) {
    baseName = substring(baseName, 0, lengthOf(baseName) - 4);
}
if (endsWith(baseName, ".tiff")) {
    baseName = substring(baseName, 0, lengthOf(baseName) - 5);
}
outputDir = dir + baseName + File.separator;
File.makeDirectory(outputDir);

// --------------------
// Open input
// --------------------
open(path);

// Do not rely on selectWindow(name) in headless mode.
// Just assume the opened image is active.

// --------------------
// Convert plain stack to hyperstack if needed
// --------------------
if (!Stack.isHyperstack) {
    nTotal = nSlices;
    nChannels = 4;
    nZSlices = 1;

    info = getImageInfo();
    idx = indexOf(info, "slices: ");
    if (idx >= 0) {
        idx = idx + lengthOf("slices: ");
        endIdx = indexOf(info, "\n", idx);
        if (endIdx < 0) endIdx = lengthOf(info);
        parsed = parseInt(substring(info, idx, endIdx));
        if (parsed > 0) nZSlices = parsed;
    }

    nFrames = nTotal / (nChannels * nZSlices);

    run("Stack to Hyperstack...", 
        "order=xyczt(default) channels=" + nChannels +
        " slices=" + nZSlices +
        " frames=" + nFrames +
        " display=Composite");
}

// --------------------
// Enhance contrast on all 4 channels
// --------------------
for (c = 1; c <= 4; c++) {
    Stack.setChannel(c);
    run("Enhance Contrast", "saturated=0.35");
}

// --------------------
// Drift correction
// --------------------
run("Correct 3D drift", "channel=1 correct multi_time_scale sub_pixel edge_enhance only=0 lowest=1 highest=4 max_shift_x=10.000000000 max_shift_y=10.000000000 max_shift_z=10.000");

// In headless mode, don't assume the exact title string.
// The newly created output should now be the active image.
registeredTitle = getTitle();
saveAs("Tiff", outputDir + baseName + "_correct.tif");

// --------------------
// Composite + contrast
// --------------------
run("Make Composite");
for (c = 1; c <= 4; c++) {
    Stack.setChannel(c);
    run("Enhance Contrast", "saturated=0.15");
}
saveAs("Tiff", outputDir + baseName + "_composite.tif");

// --------------------
// Max projection
// --------------------
run("Z Project...", "projection=[Max Intensity] all");
max_name = getTitle();
saveAs("Tiff", outputDir + max_name);

run("Split Channels");

// Debug: print actual split window titles
titles = getList("image.titles");
print("Open image titles after Split Channels:");
for (i = 0; i < titles.length; i++) {
    print("  " + titles[i]);
}

// Find the actual split channel windows by prefix only
c1Title = "";
c2Title = "";
c3Title = "";
c4Title = "";

for (i = 0; i < titles.length; i++) {
    t = titles[i];
    if (startsWith(t, "C1-")) c1Title = t;
    if (startsWith(t, "C2-")) c2Title = t;
    if (startsWith(t, "C3-")) c3Title = t;
    if (startsWith(t, "C4-")) c4Title = t;
}

if (c1Title == "" || c2Title == "" || c3Title == "" || c4Title == "") {
    print("ERROR: Could not find all split channel windows.");
    exit();
}

// C1 = Nucleus (blue)
selectWindow(c1Title);
run("Blue");
saveAs("Tiff", outputDir + baseName + "_Nucleus.tif");
close();

// C2 = red
selectWindow(c2Title);
run("Red");
saveAs("Tiff", outputDir + baseName + "_red.tif");
close();

// C3 = green
selectWindow(c3Title);
run("Green");
saveAs("Tiff", outputDir + baseName + "_green.tif");
close();

// C4 = purple/magenta
selectWindow(c4Title);
run("Magenta");
saveAs("Tiff", outputDir + baseName + "_purple.tif");
close();

setBatchMode(false);
