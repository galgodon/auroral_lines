# auroral_lines
Project to find auroral lines in red galaxies. Started as a summer research project in 2018 at UCSC with Francesco Belfiore and myself. It was picked up again in the Winter 2019 quarter with the addition of Renbin Yan and Kyle Westfall. The project uses MaNGA data.

Explanation of files:
0. table.py  -  Pulls data we need from graymalkin into a data file. This code must run in graymalkin
1. galaxy_selection.py  -  Code used to find red quiescent galaxies in MaNGA. This code will also create plots to show how the galaxy selction is made
2. table_analysis.py  -  Uses the table made in table.py and runs analysis on that data. At the end there will be 3 metallicity bins and 3 velocity offset sub-bins as well as a matching control sample.
3. get_flux.py  -  This code must also run from within graymalkin. This code will take the flux data for all of the spaxels included in the bins/subbins/control.
4. stacking.py  -  This code stacks the spectra (stacks the flux files pulled from get_flux.py). This stacking includes masks. Note that this only stacks within each velocity offset sub-bin
5. analyzing_stack.py  -  This is more of a test code that looks at the output of stacking.py and makes sure everything looks fine
6. combining_subbins.py  -  This will stack all of the subbins leaving only 3 plots, one for each metallicity bin. This is where the final plots I made are.
