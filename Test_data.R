# make a dummy test script to run through precip manip
# turn timeseries into 4868 identical grids so it runs in same way as dwr data
# 6 year timeseries with various levels of total precip.  
# w/in a year, year is a single block of flow with 2 high precip days on Jan 1/2. 
# Save as separate rds file for each year. Each should have 4868X365 vals=1776820 (ignore leap years)

year1 <- rep.int(1, 365)
year1[1:2] <- 10
year1file <- rep.int(0,1776820)
# Set base as all years are day 1,2=10 and day 3-365=1
for(day in seq(1:length(year1))){
  vals <- rep(year1[day], 4868)
  day1 <- 1+4868*(day-1)
  dayend <- 4868 + 4868*(day-1)
  year1file[day1:dayend] <- vals
}
# add some more, wetter years 
year2file <- year1file*2
year3file <- year1file*3
year4file <- year1file*4
year5file <- year1file*5
year6file <- year1file*6
all_files <- list(year1file, year2file, year3file, year4file, year5file, year6file)
filenames <- list("year1file.rds", "year2file.rds", "year3file.rds", "year4file.rds", "year5file.rds", "year6file.rds")
# convert 6 test years into RDS format
path = "/Users/noellepatterson/apps/Other/Climate_change_research/data_inputs/test_data/"
for(f in seq(1:length(all_files))){
  dummycol <- rep(0, length(year1file))
  all_files[[f]] <- as.data.frame((cbind(dummycol, dummycol, dummycol, dummycol, dummycol, all_files[[f]]))) 
  saveRDS(all_files[[f]], file = paste(path, filenames[[f]], sep=""))
}



  