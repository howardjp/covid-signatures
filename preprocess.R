MAXTIMEINICU <- 72

library("data.table")
library("FSelectorRcpp")
library("glue")
library("here")
library("logger")
library("readr")
library("zoo")

i_am("preprocess.R")

base_base_loc <- here("data")
base_loc <- here(base_base_loc, "sepsistest")
preprocessed_loc <- here(base_loc, "preprocessed")

all_data_df <- data.frame()

psv_glob <- sort(Sys.glob(paste0(base_loc, "/*.psv")))

for(psv_file in psv_glob) {
  log_info(glue("Loading file {psv_file}"))
  tmp_df <- read_delim(psv_file, delim = "|", show_col_types = FALSE)
  tmp_df$source <- tools::file_path_sans_ext(basename(psv_file))
  all_data_df <- rbind(all_data_df, tmp_df)
}

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))

all_data_df[is.nan(all_data_df)] <- NA
all_data_dt <- data.table(all_data_df)
all_data_dt[, SepsisLabel := max(SepsisLabel), by = source]
all_data_dt[, SepsisLabel := as.factor(SepsisLabel)]

meta_data_dt <- all_data_dt[, c("source", "ICULOS")]
core_data_dt <- all_data_dt[, -c("source", "ICULOS", "SepsisLabel", "HospAdmTime")]

NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))
core_data_dt <- replace(core_data_dt, TRUE, lapply(core_data_dt, NA2mean))

core_data_dt[, SepsisLabel := all_data_dt[, SepsisLabel]]
weights <- data.table(information_gain(SepsisLabel ~ ., core_data_dt))
columns_to_keep <- weights[importance > 0.01, attributes]

prepped_data_dt <- cbind(meta_data_dt, core_data_dt[, columns_to_keep, with = FALSE])

logger <- logger()

make_json <- function(data.dt, patient_list, filename) {
    covid.data <- list()
    covid.data$time_index <- sort(unique(data.dt[, TINDEX]))
    covid.data$info <- list()
    covid.data$data <- list()
    covid.data$outcome <- list()

    for(i in patient_list) {
        mimic3.dt.i <- data.dt[PATIENT == i]
        setorderv(mimic3.dt.i, "TINDEX")
        outcome.i <- mimic3.dt.i[1, OUTCOME]
        covid.data$info[[i]] <- list()
        for(j in mimic3.dt.i[, TINDEX]) {
            id <- uuid::UUIDgenerate()
            mimic3.dt.i.j <- mimic3.dt.i[TINDEX <= j]
            dindex.i.j <- mimic3.dt.i[, max(TINDEX)] - mimic3.dt.i.j[, max(TINDEX)]
            debug(logger, paste("Patient:", i, "Time:", j, "Outcome:", outcome.i, "Time Remaining:", dindex.i.j))
            mimic3.dt.i.j <- mimic3.dt.i.j[, -c("PATIENT", "DINDEX", "OUTCOME")]
            covid.data$info[[i]][[id]] <- list(time = j)
            covid.data$outcome[[id]] <- list(outcome = outcome.i, time = dindex.i.j)
            covid.data$data[[id]] <- as.matrix(mimic3.dt.i.j)
        }
    }

    info(logger, "Converting to JSON")
    covid.data.json <- toJSON(covid.data, digits = NA, pretty = TRUE, na = "null", auto_unbox = TRUE)

    info(logger, paste("Writing JSON output to", filename))
    write(covid.data.json, filename)

    info(logger, "Compressing JSON output file")
    system(paste("bzip2 -9", filename))
}