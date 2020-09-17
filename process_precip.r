#################
### LIBRARIES ###
#################


#################
### FUNCTIONS ###
#################
separate_days <- function(precip_all_grids, grid_num){
  locs = seq(1, length(precip_all_grids), 4868) + grid_num - 1
  grid = precip_all_grids[c(locs)]
  return(grid)
}

list_grids <- function(files){
  # function takes all data and reformats into 4868 lists (one for each grid), each 
  # containing 64 lists (one for each year), each containing a 365/366 day precip trace 
  all_grids <- vector(mode = "list", length = 4868)
  # loop through each grid 
  for(grid_count in 1:4868){
    # populate each grid with its 64-yr timeseries
    # grid <- vector(mode = "list", length = 64)
    # for(year_count in 1:64){
    grid <- vector(mode = "list", length = 6) # for test runs
    for(year_count in 1:6){ # for test runs
      year <- separate_days(files[[year_count]][,6], grid_count)
      # insert first year as first entry in grid
      grid[[year_count]] <- year
      # repeat for all years (put process into for loop or apply func)
    }
    all_grids[[grid_count]] <- grid
  }
  return(all_grids)
}

convert_grids_to_files <- function(files, grid_list){
  # go through each file to update column 6
  for(file_num in seq(1:length(files))){
    # in each file, go through each grid
    for(grid_num in seq(1:length(grid_list))){
      # for first file, put each grid's first list values into 4868th file vals
      locs = seq(1, length(files[[file_num]][,6]), 4868) + grid_num - 1
      files[[file_num]][,6][locs] <- grid_list[[grid_num]][[file_num]]
    }
  } 
  return(files)
}

interannual_precip_manip <- function(files){
  # intensity parameters
  dry_year_threshold = .2 # val 0-1, based on Persad e.a. 2020
  wet_year_threshold = .8 # val 0-1, based on Persad e.a. 2020
  freq_20th_perc = .33 # val 0-1, percentage of years under 20th perc in output
  # To achieve metric in Persad paper, increase occurrence of years in 20th/80th percentage
  # of precip to get an increased frequency of years in these extreme bins. Assumption for
  # calc that shift set for the dry season will reflect similarly in the wet years. 
  # Separate each grid into a 64-yr timeseries (list of 64 lists)
  grid_list <- list_grids(files)
  
  # within 64-yr timeseries for each grid: 
  for(grid_num in seq(1:length(grid_list))){
    grid <- grid_list[[grid_num]]
    # assign all years their percentile based on cumsum of precip
    cumsums <- unlist(lapply(grid, sum))
    # percentiles <- rank(cumsums)/length(cumsums)
    # e.g., 20th percentile value before modification
    dry_threshold <- quantile(cumsums, dry_year_threshold)
    # a higher percentile to lower down to (e.g.) 20th percentile values
    lowering_threshold <- quantile(cumsums, freq_20th_perc)
    # total precip volume to remove from all years in lower-down percentile
    reduction_volume <- lowering_threshold-dry_threshold
    # target certain number of dry years to remove precip from
    dry_years_locs <- which(cumsums < lowering_threshold)
    # harvest precip proportionally off all flow days in dry years
    for(dry_year in dry_years_locs){
      reduction_perc <- reduction_volume/sum(grid[[dry_year]])
      grid[[dry_year]] <- grid[[dry_year]]*(1-reduction_perc)
    }
    # tally total amt of dry year precip harvest
    precip_harvest <- reduction_volume*length(dry_years_locs)
    # target wet years to add precip to, split precip harvest among these years
    wet_threshold <- quantile(cumsums, 1-freq_20th_perc)
    wet_year_locs <- which(cumsums > wet_threshold)
    # add harvested recip to each flow day equally
    each_year_addition <- precip_harvest/length(wet_year_locs)
    for(wet_year in wet_year_locs) {
      perc_increase <- (sum(grid[[wet_year]]) + each_year_addition)/sum(grid[[wet_year]])
      grid[[wet_year]] <- grid[[wet_year]]*perc_increase
    }
    # calc final distribution of years (years in extremes, 20/80 bins?)
    updated_cumsums <- unlist(lapply(grid, sum))
    lower_bin_freq <- length(which(updated_cumsums < dry_threshold))/length(grid)
    upper_bin_freq <- length(which(updated_cumsums > wet_threshold))/length(grid)
    grid_list[[grid_num]] <- grid
  }
  # convert updated grids back into original (updated) file format
  updated_files <- convert_grids_to_files(files, grid_list)
  return(updated_files)
}

intra_precip_manip <- function(annual_precip){
  # Function for modifying precip intensity, input is a one-year precip timeseries
  
  # set intensity parameters
  dry_percent = .25 # val 0-1, percent annual precip in dry months at end
  wet_percent = 1 - dry_percent
  highflow_perc = .15 # val 0-1, percent annual flow in five highest days at end
  
  # sum up precip from April - Oct (dry season)
  wet = c(1:91, 306:length(annual_precip)) # Nov-March
  dry = c(92:305) # April - Oct
  
  # create backup of original annual precip
  orig_annual_precip <- annual_precip
  annual_cum <- sum(annual_precip)
  # calculate volume of precip in dry and wet months, end goal
  final_dry_vol <- annual_cum*dry_percent
  final_wet_vol <- annual_cum*wet_percent
  # calculate current vol in dry months, and difference to goal
  current_dry_vol <- sum(annual_precip[dry])
  dry_diff <- current_dry_vol - final_dry_vol
  perc_dry_reduce <- final_dry_vol/current_dry_vol
  # Apply reduction percentage across all dry season, save harvest amt for later
  annual_precip[dry] <- annual_precip[dry] * perc_dry_reduce
  dry_harvest <- sum(orig_annual_precip[dry] - annual_precip[dry])
  
  # Onto extreme wet days
  # Calc current percent rainfall in 5 highest wet days
  current_high_vol <- sum(rev(sort(annual_precip[wet]))[1:5])
  high_vol_cutoff <- rev(sort(annual_precip[wet]))[5]
  final_high_vol <- highflow_perc * annual_cum
  high_diff <- final_high_vol - current_high_vol
  # If there is enough water from dry harvest for high day goal, add amt needed and 
  # distribute rest across wet season precip days
  if(high_diff <= dry_harvest){
    high_days_locs <- which(annual_precip[wet] >= high_vol_cutoff)
    annual_precip[wet][high_days_locs] <- annual_precip[wet][high_days_locs] + high_diff/length(high_days_locs)
    remaining_harvest <- dry_harvest - high_diff
    wet_days_locs <- which(annual_precip[wet]>0 & annual_precip[wet]<high_vol_cutoff)
    annual_precip[wet][wet_days_locs] <- annual_precip[wet][wet_days_locs] + remaining_harvest/length(wet_days_locs)
  } else{
    # Otherwise if dry harvest is not enough to fill high days deficit, shave some additional precip off wet season days
    remaining_deficit <- high_diff - dry_harvest
    # Shave off precip from wet season days to make up high day deficit
    annual_precip[wet][wet_days_locs] <- annual_precip[wet][wet_days_locs] - remaining_deficit/length(wet_days_locs)
    # Add deficit to high precip days
    annual_precip[wet][high_days_locs] <- annual_precip[wet][high_days_locs] + (dry_harvest+remaining_deficit)/length(high_days_locs)
  }
}

#################
##### MAIN ######
#################

path = "/Users/noellepatterson/apps/Other/Climate_change_research/data_inputs/detrended-one-8th-rdata/"
path = "/Users/noellepatterson/apps/Other/Climate_change_research/data_inputs/test_data/"
setwd(path)
filenames = list.files(pattern="*.rds")
files = lapply(filenames, readRDS)
# File organization: 64 files, each for a year of data from 1950-2013. Leap years included, so every forth file is larger.
# Within a file, included is every day of data for each grid in the state. Grids are rep'd as points,
# 4868 grids/points total. Loc, date, precip, and temp (as min and max) are included in each row. 
# I remove precip column for processing. Data is organized by listing the first day of year's data
# for every grid, then moving to the next date and the next. So to pull out a timeseries for a single 
# grid, need to pull out every 4868th data point...

# apply interannual changes before entering into loop
updated_files <- interannual_precip_manip(files)
# to test:
all.equal(files[[1]], updated_files[[1]])

# Perform intraannual changes for all 64 years on updated files
for(file_num in seq(1, length(updated_files))){ 
  # proof of concept for first year data (out of 64 total years)
  precip_all_grids = updated_files[[file_num]][,6] # precipe is 6th col of df
  grid_ls = seq(1, 4868, 1) 
  all_grids = lapply(grid_ls, separate_days, precip_all_grids=precip_all_grids)
  # next: apply an intensity alteration to each time series
  new_grids = lapply(all_grids, intra_precip_manip)
  new_grids = lapply(new_grids, unlist)
  # then: re-insert to db for that year
  precip_all_grids_updated = vector(mode="double", length=length(precip_all_grids))
  
  for(i in seq(1, 4868, 1)){
    locs = seq(1, length(precip_all_grids), 4868) + i - 1
    precip_all_grids_updated[c(locs)] <- new_grids[[i]]
  }
  # rebuild file with the updated precip
  updated_file = files[[file_num]]
  updated_file[,6] <- precip_all_grids_updated
  print(paste(file_num," done"))
  new_path = "/Users/noellepatterson/apps/Other/Climate_change_research/data_outputs/detrended-one-8th-rdata/"
  setwd(new_path)
  saveRDS(updated_file, paste("updated_precip",filenames[file_num], sep="_"))
}
path = "/Users/noellepatterson/apps/Other/Climate_change_research/data_outputs/detrended-one-8th-rdata/"
test_file <- readRDS(paste(path, "updated_precip_year1file.rds", sep=""))
