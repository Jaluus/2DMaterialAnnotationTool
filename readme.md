# A Fast Flake Annotator Tool

## Workflow

1. Generate a Dataset with the Microscope
2. Create the Folder structure
3. Find a few flakes by hand or use parameters from previous detection runs to approximate the Flake Contrast.
4. Use the found contrasts to run a Scan with Approximate Contrasts.
   1. You will find a lot of False Positives but will most likely get a lost of possible flakes.
5. Now move all the images you found into the previously generated Set Folder.
6. Run the Annotator to reannotate the Images by hand.
   1. The tool will greatly help you with this by running the Watershed algorithm to draw accurate boundries.
7. Now plot the annotated flakes and fit a GMM to the thicknesses.
   1. It is advisable to use a one more component than is actually there to also fit the noise in the background.
8. Reannotate the flakes with the GMM to get good masks and approximate the contrast to a better degree.
