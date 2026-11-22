# -----------------------------------------------------------------------------
# Step 04 (R part) - bioclimatic variables with dismo::biovars
#
# Called by scripts/04_bioclim.py; not meant to be run by hand.
# Usage: Rscript 04_bioclim_biovars.R <input_dir> <output_dir>
#   input_dir  contains, for every set listed in sets.txt:
#              <set>_prec.tif, <set>_tmin.tif, <set>_tmax.tif (12 bands each)
#   output_dir receives <set>_bio.tif (19 bands, bio1..bio19, float32, NA = -9999)
# dismo::biovars is the function used in the original manuscript (decision D05).
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(raster)
  library(dismo)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript 04_bioclim_biovars.R <input_dir> <output_dir>")
in_dir <- args[1]
out_dir <- args[2]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

sets <- readLines(file.path(in_dir, "sets.txt"))
cat("R", R.version$major, ".", R.version$minor,
    "| dismo", as.character(packageVersion("dismo")),
    "| raster", as.character(packageVersion("raster")), "\n")

for (s in sets) {
  rd <- function(v) {
    r <- raster::stack(file.path(in_dir, paste0(s, "_", v, ".tif")))
    raster::NAvalue(r) <- -9999
    r
  }
  prec <- rd("prec"); tmin <- rd("tmin"); tmax <- rd("tmax")
  stopifnot(raster::nlayers(prec) == 12, raster::nlayers(tmin) == 12, raster::nlayers(tmax) == 12)
  bio <- dismo::biovars(prec, tmin, tmax)
  names(bio) <- paste0("bio", 1:19)
  raster::writeRaster(bio, file.path(out_dir, paste0(s, "_bio.tif")), format = "GTiff",
                      datatype = "FLT4S", NAflag = -9999, overwrite = TRUE,
                      options = c("COMPRESS=LZW"))
  cat("  done:", s, "\n")
}
cat("biovars finished for", length(sets), "sets\n")
