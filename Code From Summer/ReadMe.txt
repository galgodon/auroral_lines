Listed in order of creation.

Check_galaxy.py ----- Given catalogue of eLIER galaxies. This script runs through each galaxy in that catalogue to insure that Graymalkin has data for each galaxy. If there is a galaxy in the catalogue that Graymalkin does not have, it will print the galaxy name.

PA_eLIER_galaxies_test.py ----- Test code, ignore.

PA_eLIER_galaxies.py ----- Finds position angles of eLIER galaxies from the catalogue. Outputs the position angle of each galaxy.

Specific_galaxy_maps.py ----- Requires input of plate and ifu for a specific galaxy. Produces the velocity map of that galaxy from the Ha velocity and from the stellar velocity. The code overlays the calculated position angle with errors.

Stacking.py ----- Creates four stacks. Stack1: Low mass aligned galaxies. Stack2: Low mass misaligned galaxies. Stack3: High mass aligned galaxies. Stack4: High mass misaligned galaxies. Outputs the four stacked spectra.

Stacking_Francesco.py ----- Replaces Stacking.py. Francesco’s code replaced the interp1d so we could account for masked pixels. 

Stacking_sres.py ----- Same as Stacking.py but for the spectral resolution rather than the spectra. This was used in order to use the continuum fitting code.

Check_galaxy_2.py ----- same as Check_galaxy.py except for a catalogue of lineless galaxies rather than the eLIER ones.




