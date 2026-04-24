all: sine_hamann_grid.png

.PHONY: sine_hamann_grid.png

sine_hamann_grid.png:
	cargo run --example sine_csv
	gnuplot sine_hamann_plot.gnu

clean:
	$(RM) sine_hamann_grid.png
	$(RM) -r sine_plot_out
