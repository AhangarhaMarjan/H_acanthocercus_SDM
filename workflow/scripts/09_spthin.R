# -----------------------------------------------------------------------------
# Step 09 (R part) - spatial thinning with spThin (Aiello-Lammens et al. 2015)
# Called by scripts/09_occurrences_thinning.py; not meant to be run by hand.
# Usage: Rscript 09_spthin.R <input_csv> <output_dir> <seed> <reps> <dist_km> [<dist_km> ...]
#   input_csv: columns id, lon, lat, species
#   writes thin_<dist>km.csv (lon, lat of the largest retained set) and
#          thin_<dist>km_sizes.csv (retained-set size of every repetition)
# -----------------------------------------------------------------------------
suppressPackageStartupMessages(library(spThin))
args <- commandArgs(trailingOnly = TRUE)
in_csv <- args[1]; out_dir <- args[2]; seed <- as.integer(args[3]); reps <- as.integer(args[4])
dists <- as.numeric(args[5:length(args)])
d <- read.csv(in_csv)
cat("spThin", as.character(packageVersion("spThin")), "| records", nrow(d), "\n")
for (km in dists) {
  set.seed(seed)
  res <- thin(loc.data = d, lat.col = "lat", long.col = "lon", spec.col = "species",
              thin.par = km, reps = reps, locs.thinned.list.return = TRUE,
              write.files = FALSE, write.log.file = FALSE, verbose = FALSE)
  n <- sapply(res, nrow)
  best <- res[[which.max(n)]]
  pick <- function(df, exact, pattern) {
    if (exact %in% names(df)) df[[exact]] else df[[grep(pattern, names(df), ignore.case = TRUE)[1]]]
  }
  tag <- formatC(km, width = 2, flag = "0")
  write.csv(data.frame(lon = pick(best, "Longitude", "^lon"), lat = pick(best, "Latitude", "^lat")),
            file.path(out_dir, paste0("thin_", tag, "km.csv")), row.names = FALSE)
  write.csv(data.frame(rep = seq_along(n), retained = n),
            file.path(out_dir, paste0("thin_", tag, "km_sizes.csv")), row.names = FALSE)
  cat("  ", km, "km: max retained", max(n), "of", nrow(d), "\n")
}
