

#################
### FUNCTIONS ###
#################
separate_days <- function(precip_all_grids, day){
  locs = seq(1, length(precip_all_grids), 4868) + day - 1
  grid = precip_all_grids[c(locs)]
  return(grid)
}
precip_manip <- function(annual_precip){
  annual_precip2 = lapply(annual_precip,`+`,1)
  return(annual_precip2)
}

path = "/Users/noellepatterson/apps/Other/Climate_change_research /data_inputs/detrended-one-8th-rdata/"
setwd(path)
filenames = list.files(pattern="*.rds")
files = lapply(filenames, readRDS)
# File organization: 64 files, each for a year of data from 1950-2013. Leap years included, so every forth file is larger.
# Within a file, included is every day of data for each grid in the state. Grids are rep'd as points,
# 4868 grids/points total. Loc, date, precip, and temp (as min and max) are included in each row. 
# I remove precipe column for processing. Data is organized by listing the first day of year's data
# for every grid, then moving to the next date and the next. So to pull out a timeseries for a single 
# grid, need to pull out every 4868th data point...

# Perform process for all 64 years in huge db
for(file_num in seq(1, length(files))){ 
  # proof of concept for first year data (out of 64 total years)
  precip_all_grids = files[[file_num]][,6] # precipe is 6th col of df
  grid_ls = seq(1, 4868, 1) 
  # am somehow losing leap years in this step, then can't reinsert data into orig df
  all_grids = lapply(grid_ls, separate_days, precip_all_grids=precip_all_grids)
  # next: apply an intensity alteration to each time series
  new_grids = lapply(all_grids, precip_manip)
  new_grids = lapply(new_grids, unlist)
  # then: re-insert to db for that year
  precip_all_grids_updated = vector(mode="double", length=length(precip_all_grids))
  
  for(i in seq(1, 4868, 1)){
    locs = seq(1, length(precip_all_grids), 4868) + i - 1
    precip_all_grids_updated[c(locs)] <- new_grids[[i]]
  }
  files[[file_num]][,6] <- precip_all_grids_updated
  # saveRDS(precip_all_grids_updated, "updated_precip.rds")
}

