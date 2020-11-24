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
    grid <- vector(mode = "list", length = 64)
    for(year_count in 1:64){
      year <- separate_days(files[[year_count]][,6], grid_count)
      # insert first year as first entry in grid
      grid[[year_count]] <- year
      # repeat for all years (put process into for loop or apply func)
    }
    all_grids[[grid_count]] <- grid
  }
  return(all_grids)
}

list_grids_merced <- function(files){
  # function takes all data and reformats into 30 lists (one for each grid), each 
  # containing 64 lists (one for each year), each containing a 365/366 day precip trace 
  merced <- read.delim(
    "/Users/noellepatterson/apps/Other/Climate_change_research/data_inputs/Merced_grids.txt", header=FALSE, sep="")
  m_lat <- merced[,1]
  m_lon <- merced[,2]
  test_year <- files[[1]]
  test_lat <- test_year[,2]
  test_year$latlon <- with(test_year, paste0(lat, lon))
  unique_grids <- test_year[1:4868,]
  # identify merced grids out of the total 4868
  which(unique_grids$lon == m_lon & unique_grids$lat == m_lat)
  for(index in seq(length(merced))){
    unique_grids[unique_grids$lon == m_lon[index] & unique_grids$lat == m_lat[index]]
    unique_grids[unique_grids$lon == m_lon[index]]
  }
  unique_grids[unique_grids$lon == m_lon & unique_grids$lat == m_lat]
  
  all_grids <- vector(mode = "list", length = 4868)
  # loop through each grid 
  for(grid_count in 1:4868){
    # populate each grid with its 64-yr timeseries
    grid <- vector(mode = "list", length = 64)
    for(year_count in 1:64){
      year <- separate_days(files[[year_count]][,6], grid_count)
      # insert first year as first entry in grid
      grid[[year_count]] <- year
      # repeat for all years (put process into for loop or apply func)
    }
    all_grids[[grid_count]] <- grid
  }
  return(all_grids)
}

list_grids_test <- function(files){
  # function takes test data and reformats into 4868 lists (one for each grid), each 
  # containing 6 lists (one for each year), each containing a 365/366 day precip trace 
  all_grids <- vector(mode = "list", length = 4868)
  # loop through each grid 
  for(grid_count in 1:4868){
    # populate each grid with its 6-yr timeseries
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

perc_wet_days_calc <- function(annual_data){
  # calc the percent of precip in the year's five highest days
  high_days <- rev(sort(annual_data))[1:5]
  high_days_perc <- sum(high_days)/sum(annual_data)
  return(high_days_perc)
}

perc_wet_months_calc <- function(annual_data){
  # calc the percentage of precip occuring in Nov-March months
  wet_months_loc <- c(1:91, 306:length(annual_data)) # Nov-March
  wet_month_perc <- sum(annual_data[wet_months_loc])/sum(annual_data)
  return(wet_month_perc)
}

calc_sd_annual <- function(all_data){
  year_cums <- unlist(lapply(all_data, sum))
  sd_annual <- sd(year_cums)
  return(sd_annual)
}

get_perc_20th80th <- function(all_data){
  # calculate 20th and 80th percentiles across entire POR
  perc_20th <- rep(NA, 4868)
  perc_80th <- rep(NA, 4868)
  for(grid in seq(1:length(perc_20th))){
    year_sums <- unlist(lapply(all_data[[grid]], sum))
    perc_20th[grid] <- quantile(year_sums, probs=c(0.2))
    perc_80th[grid] <- quantile(year_sums, probs=c(0.8))
  }
  perc_20th80th <- list(perc_20th, perc_80th)
  return(perc_20th80th)
}

calc_metric_20th80th <- function(all_data, perc_20th, perc_80th){
  # calc number of years below 20th percentile and above 80th percentile (of orig precip)
  # perc20th and perc80th specific to this grid are input as single values
  year_sums <- unlist(lapply(all_data, sum))
  count <- 0
  for(year in year_sums){
    if(year < perc_20th){
      count <- count + 1
    } else if(year > perc_80th){
      count <- count + 1
    }
  }
  return(count/length(year_sums))
}
  
create_summary <- function(files, orig_files){
  # Metric defs: cumulative = total precip in entire POR. 
  # perc_wet_months: percentage of precip in Nov-Mar
  # perc_wet_days: percentate of precip in five highest days of the year
  # 20th80th_perc: number of years above 80th perc and below 20th perc (percentiles based on original data)
  summary_df <- data.frame("Metric" = c("mean", "median", "cumulative", "sd_daily", "sd_annual", 
                                        "perc_wet_months", "perc_wet_days", "20th80th_perc"))
  summary_df$original <- NA
  grid_list <- list_grids_test(files)
  grid_list_orig <- list_grids_test(orig_files)
  # calculate 20th and 80th annual percentiles of original data, to use in 20th/80th calc
  perc_20th_80th <- get_perc_20th80th(grid_list_orig)
  perc_20th <- perc_20th_80th[1]
  perc_80th <- perc_20th_80th[2]
  
  # create variables for metrics needing daily data. One element in list for each year. 
  mean_stat <- rep(NA, 4868)
  median_stat <- rep(NA, 4868)
  cumul <- rep(NA, 4868)
  sd_daily <- rep(NA, 4868)
  perc_wet_days <- rep(NA, 4868)
  perc_wet_months <- rep(NA, 4868)
  sd_annual <- rep(NA, 4868)
  perc_20th80th <- rep(NA, 4868)

  # populate summary data grids with x annual vals in list, to average later
  for(count in seq(1:length(mean_stat))){
    mean_stat[count] <- list(rep(NA, 6))
    median_stat[count] <- list(rep(NA, 6))
    cumul[count] <- list(rep(NA, 6))
    sd_daily[count] <- list(rep(NA, 6))
    perc_wet_days[count] <- list(rep(NA, 6))
    perc_wet_months[count] <- list(rep(NA, 6))
  }
  # populate summary data grids: one value for each year, for each grid
  for(grid in seq(1:length(mean_stat))){
    mean_stat[[grid]] <- unlist(lapply(grid_list[[grid]], mean))
    median_stat[[grid]] <- unlist(lapply(grid_list[[grid]], median))
    cumul[[grid]] <- unlist(lapply(grid_list[[grid]], sum))
    sd_daily[[grid]] <- unlist(lapply(grid_list[[grid]], sd))
    perc_wet_days[[grid]] <- unlist(lapply(grid_list[[grid]], perc_wet_days_calc))
    perc_wet_months[[grid]] <- unlist(lapply(grid_list[[grid]], perc_wet_months_calc))
    sd_annual[[grid]] <- calc_sd_annual(grid_list[[grid]])
    perc_20th80th[[grid]] <- calc_metric_20th80th(grid_list[[grid]], perc_20th[[1]][[grid]], perc_80th[[1]][[grid]])
  }
  # average annual stats, where necessary
  for(grid in seq(1:length(mean_stat))){
    mean_stat[[grid]] <- mean(mean_stat[[grid]])
    median_stat[[grid]] <- mean(median_stat[[grid]])
    cumul[[grid]] <- mean(cumul[[grid]])
    sd_daily[[grid]] <- mean(sd_daily[[grid]])
    perc_wet_days[[grid]] <- mean(perc_wet_days[[grid]])
    perc_wet_months[[grid]] <- mean(perc_wet_months[[grid]])
  }
  # create por metrics
  cb <- cbind(mean_stat, median_stat, cumul, sd_daily, perc_wet_days, perc_wet_months, sd_annual, perc_20th80th)
  summary_df <- data.frame(cb)
  return(summary_df)
}

intra_precip_manip <- function(annual_precip){
  # Function for modifying precip intensity within a year, input is a one-year precip timeseries
  
  # set intensity parameters
  wet_percent = .0556 # val 0-1, percent increase in wet season precip
  dry_percent = 1 - wet_percent
  highflow_perc = .1435 # val 0-1, percent increase in flow of three highest days
  
  # sum up precip from April - Oct (dry season)
  wet = c(1:91, 306:length(annual_precip)) # Nov-March
  dry = c(92:305) # April - Oct
  
  # create backup of original annual precip
  orig_annual_precip <- annual_precip
  orig_wet_precip <- sum(orig_annual_precip[wet])
  annual_cum <- sum(annual_precip)
  # calculate volume of precip in dry and wet months, end goal
  final_wet_vol <- orig_wet_precip + (orig_wet_precip * wet_percent)
  final_dry_vol <- annual_cum - final_wet_vol
  
  # calculate current vol in dry months, and difference to goal
  current_dry_vol <- sum(annual_precip[dry])
  dry_diff <- current_dry_vol - final_dry_vol
  perc_dry_reduce <- final_dry_vol/current_dry_vol
  # Apply reduction percentage across all dry season, save harvest amt for later
  annual_precip[dry] <- annual_precip[dry] * perc_dry_reduce
  dry_harvest <- sum(orig_annual_precip[dry] - annual_precip[dry])
  
  # Onto extreme wet days
  # Calc current percent rainfall in 3 highest wet days
  current_high_vol <- sum(rev(sort(annual_precip[wet]))[1:3])
  high_vol_cutoff <- rev(sort(annual_precip[wet]))[3]
  # Set highflow increase to be a percentage increase from original value. 
  final_high_vol <- current_high_vol + (highflow_perc * current_high_vol) 
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

interannual_precip_manip <- function(files){
  # Function for modifying precip intensity across all years, input is entire collection of timeseries
  # intensity parameters
  orig_perc_low = .4 # val 0-1, avg annual precip value low before shifting precip across years 
  orig_perc_high = .6 # val 0-1, avg annual precip value high before shifting precip across years 
  final_perc_low = .2 # val 0-1, avg annual precip value low before shifting precip across years. Must be lower than orig. 
  final_perc_high = .8 # val 0-1, avg annual precip value high before shifting precip across years. Must be higher than orig.
  extreme_shift_low = .1 # val -(0-1), reduce driest years this far below current low
  extreme_shift_high = .1 # val 0-1, raise highest years this far above current high
  extreme_shift_percent = 0.05 # val 0-1, number of years corresponding to this percentage will be
    # shifted out to new extremes on min and max
  # To achieve metric in Persad paper, increase occurrence of years in 20th/80th percentage
  # of precip to get an increased frequency of years in these extreme bins. Assumption for
  # calc that shift set for the dry season will reflect similarly in the wet years. 
  # Separate each grid into a 64-yr timeseries (list of 64 lists)
  # grid_list <- list_grids_test(files)
  grid_list <- list_grids(files)
  
  # within 64-yr timeseries for each grid: 
  for(grid_num in seq(1:length(grid_list))){
    grid <- grid_list[[grid_num]]
    # assign all years their percentile based on cumsum of precip
    cumsums <- unlist(lapply(grid, sum))
    # e.g., 20th percentile value before modification
    orig_dry_threshold <- quantile(cumsums, orig_perc_low)
    # a higher percentile to lower down to (e.g.) 20th percentile values
    new_dry_threshold <- quantile(cumsums, final_perc_low)
    # total precip volume to remove from all years in lower-down percentile
    reduction_volume <- orig_dry_threshold-new_dry_threshold
    # target certain number of dry years to remove precip from
    dry_years_locs <- which(cumsums <= orig_dry_threshold & cumsums >= new_dry_threshold)
    # harvest precip proportionally off all flow days in dry years
    for(dry_year in dry_years_locs){
      reduction_perc <- reduction_volume/sum(grid[[dry_year]])
      grid[[dry_year]] <- grid[[dry_year]]*(1-reduction_perc)
    }
    # Tally up extreme dry years 
    low_days_count <- round(length(grid)*extreme_shift_percent)
    # find low diff, value to subract off lowest current days to bring it x% lower than historic
    low_diff <- min(cumsums)*extreme_shift_low
    low_thresh <- sort(cumsums)[low_days_count]
    extreme_low_locs <- which(cumsums <= low_thresh)
    # subtract the low diff from x lowest years in record so they all move toward new dry extreme
    for(year in extreme_low_locs){
      reduction_perc <- low_diff/sum(grid[[year]])
      grid[[year]] <- grid[[year]]*(1-reduction_perc)
    }

    extreme_dry_harvest <- low_diff * low_days_count
    
    # tally total amt of dry year precip harvest
    precip_harvest <- reduction_volume*length(dry_years_locs) + extreme_dry_harvest
    # Calc amt of precip needed to fulfill changes, add water from dry harvest first, and 
    # if more needed add from middle-ground years
    
    # target wet years to add precip to, split precip harvest among these years
    orig_wet_threshold <- quantile(cumsums, orig_perc_high)
    final_wet_threshold <- quantile(cumsums, final_perc_high)
    addition_volume <- final_wet_threshold - orig_wet_threshold
    wet_year_locs <- which(cumsums >= orig_wet_threshold & cumsums <= final_wet_threshold)
    total_wet_addition <- length(wet_year_locs) * addition_volume
    
    # Add to each year in percentile category based on total percent increase
    for(wet_year in wet_year_locs) {
      perc_increase <- (sum(grid[[wet_year]]) + addition_volume)/sum(grid[[wet_year]])
      grid[[wet_year]] <- grid[[wet_year]]*perc_increase
    }
    
    # Extreme wet years
    high_days_count <- round(length(grid)*extreme_shift_percent)
    # find high diff, value to add onto highest current days to bring it x% above historic
    high_diff <- max(cumsums)*extreme_shift_high
    high_thresh <- tail(sort(cumsums), high_days_count)
    # Add the high diff to the x highest years in record so they all move toward new wet extreme
    extreme_high_locs <- which(cumsums >= high_thresh)
    for(year in extreme_high_locs){
      perc_increase <- (sum(grid[[year]]) + high_diff)/sum(grid[[year]])
      grid[[year]] <- grid[[year]]*(perc_increase)
    }

    extreme_wet_addition <- high_diff * high_days_count
    
    # Figure out how much the wet year additions exceed dry harvest
    total_wet_take <- total_wet_addition + extreme_wet_addition
    remaining_take <- total_wet_take - precip_harvest
    if(remaining_take[[1]] <= 0){
      print("no wet surplus!")
    } else {
      # For remaining surplus of wet addition, get there from taking flow off middle years
      take_each_year <- remaining_take/10 # pick a number?
      print(paste("take this amt off the middle years:", take_each_year))
      # define middle years, by rank order of 10 middle years
      middle_years <- seq(28, 37, by=1)
      middle_years_low <- sort(cumsums)[min(middle_years)]
      middle_years_high <- sort(cumsums)[max(middle_years)]
      middle_years_loc <- which(cumsums >= middle_years_low & cumsums <= middle_years_high)
      # remove take each year from each using multiply by percent method
      for(year in middle_years_loc){
        reduction_perc <- take_each_year/sum(grid[[year]])
        grid[[year]] <- grid[[year]]*(1-reduction_perc)
      }
    }
    
    # calc final distribution of years (years in extremes, 20/80 bins)
    # make sure you have precip val corresponding to orig 20th and 80th percentiles
    # first get number of years blw 20 and abv 80, orig
    # then get number of years blw 20 and abv 80, final
    # report them out somehow. These vals only calculated once per dataset. Just print for now?
    updated_cumsums <- unlist(lapply(grid, sum))
    lower_bin_freq <- length(which(updated_cumsums < orig_dry_threshold))/length(grid)
    upper_bin_freq <- length(which(updated_cumsums > orig_wet_threshold))/length(grid)
    grid_list[[grid_num]] <- grid
  }
  # convert updated grids back into original (updated) file format
  updated_files <- convert_grids_to_files(files, grid_list)
  return(updated_files)
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
# grid, pull out every 4868th data point

# apply interannual (across years) changes before entering into loop
updated_files <- interannual_precip_manip(files)
# to test:
all.equal(files[[1]], updated_files[[1]])

# Perform intraannual changes for all 64 years on updated files (within-year)
for(file_num in seq(1, length(updated_files))){ 
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

# summary stats requires two inputs: 1) files to calculate metrics on, and 2) original files to calculate original
# 20th/80th percentiles of years on. Need this whether or not the first file is the original or not. 
summary_stats_orig <- create_summary(files, files)
summary_stats_updated <- create_summary(updated_files, files)

