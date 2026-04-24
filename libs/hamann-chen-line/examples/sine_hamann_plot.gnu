# Hamann–Chen sine demo: 2×2 grid (full curve + three vertex budgets).
#
# From this crate root (libs/hamann-chen-line), after:
#   cargo run --example sine_csv
# run:
#   gnuplot examples/sine_hamann_plot.gnu
#
# Writes examples/sine_plot_out/sine_hamann_grid.png (needs pngcairo).

set terminal pngcairo size 900, 900 enhanced font "Helvetica,11"
set output "sine_hamann_grid.png"
set datafile separator comma

data = "sine_plot_out"

set xrange [0:2*pi + 0.01]
set yrange [-1.2:1.2]
set samples 300
ref(x) = sin(x)

set multiplot layout 2, 2 title "sin(x) on [0, {/Symbol p}]: Hamann–Chen curvature-based subsampling" font ",13"

set title "Reference: 100 samples (dense polyline)" font ",11"
plot ref(x) with lines lc rgb "#888888" dt 2 title "sin(x)", \
     data."/sine_full.csv" using 1:2 with linespoints lw 2 pt 7 ps 0.6 lc rgb "#0066cc" title "subsample"

set title "Hamann–Chen: m = 49 vertices"
plot ref(x) with lines lc rgb "#888888" dt 2 title "sin(x)", \
     data."/sine_m49.csv" using 1:2 with linespoints lw 2 pt 7 ps 0.6 lc rgb "#cc4400" title "subsample"

set title "Hamann–Chen: m = 24 vertices"
plot ref(x) with lines lc rgb "#888888" dt 2 title "sin(x)", \
     data."/sine_m24.csv" using 1:2 with linespoints lw 2 pt 7 ps 0.7 lc rgb "#228822" title "subsample"

set title "Hamann–Chen: m = 9 vertices"
plot ref(x) with lines lc rgb "#888888" dt 2 title "sin(x)", \
     data."/sine_m9.csv" using 1:2 with linespoints lw 2 pt 7 ps 0.9 lc rgb "#880088" title "subsample"

unset multiplot
unset output
