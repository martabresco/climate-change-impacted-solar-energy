#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=ACCESS-CM2
#SBATCH --nodes=1
#SBATCH --mail-user=s233224@dtu.dk
#SBATCH --mail-type=END
#SBATCH --exclusive=user
#SBATCH --partition=rome,workq
set echo
#https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/CSIRO-ARCCSS/
# ACCESS-CM2/historical/r1i1p1f1/3hr/rsds/gn/v20210325/rsds_3hr_ACCESS-CM2_historical_r1i1p1f1_gn_195001010130-195912312230.nc

# link for tas: 
#https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/CSIRO-ARCCSS/
#ACCESS-CM2/historical/r1i1p1f1/3hr/tas/gn/v20210325/tas_3hr_ACCESS-CM2_historical_r1i1p1f1_gn_195001010300-196001010000.nc

#https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/CSIRO-ARCCSS/
#ACCESS-CM2/historical/r1i1p1f1/3hr/tas/gn/v20210325/tas_3hr_ACCESS-CM2_historical_r1i1p1f1_gn_198001010130-19901010000.nc

root=https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/CSIRO-ARCCSS
MODEL=ACCESS-CM2
EXPR=historical
VAR=r1i1p1f1
VDATE=v20210325
TABLE=3hr
GRID=gn



mkdir -p /work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation//$MODEL/$EXPR/$VAR/http_files
cd /work/users/s233224/Climate-Change-Impacted-Solar-Energy-Generation/$MODEL/$EXPR/$VAR/http_files


#10 years at a time for rsds and rsdsdiff
year=1980
last_year=2014
vars=("rsds rsdsdiff")
((yearP10=$year+9))
while [ $year -le $last_year ]
do
   echo $year $yearP10
     for var in $vars
     do
         DATES=${year}01010130-${yearP10}12312230
         URL=${root}/${MODEL}/${EXPR}/${VAR}/${TABLE}/${var}/$GRID/$VDATE
         echo $URL
         echo $DATES

         wget -nc -nd -t 3 $URL/${var}_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc
     done

 ((year+=10))
 ((yearP10+=10)) 
 done

 
 #10 years at a time for tas
yearTAS=1980
last_yearTAS=2015
 tas=("tas")
((yearP10TAS=$yearTAS+10))
while [ $yearTAS -le $last_yearTAS ]
do
   echo $yearTAS $yearP10TAS
     for var in $tas
     do
         DATES=${yearTAS}01010300-${yearP10TAS}01010000
         URL=${root}/${MODEL}/${EXPR}/${VAR}/${TABLE}/${var}/$GRID/$VDATE
         echo $URL
         echo $DATES

         wget -nc -nd -t 3 $URL/${var}_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc
     done
echo "Before Increment: yearTAS=$yearTAS, yearP10TAS=$yearP10TAS"
 ((yearTAS+=10)) 
 ((yearP10TAS+=10)) 
echo "After Increment: yearTAS=$yearTAS, yearP10TAS=$yearP10TAS"
done 

# FIVE YEARS at a time for the last file of rsds rsdsdiff
year=2010
last_year=2014
vars=("rsds rsdsdiff")
((yearP5=$year+4))
while [ $year -le $last_year ]
do
    echo $year $yearP5

    for var in $vars
    do
        DATES=${year}01010130-${yearP5}12312230
        URL=${root}/${MODEL}/${EXPR}/${VAR}/${TABLE}/${var}/$GRID/$VDATE
        echo $URL
        echo $DATES
        
        wget -nc -nd -t 3 $URL/${var}_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc
    done

((year+=5))
((yearP5+=5))
done
#
## FIVE YEARS at a time for the last file of tas
yearTAS=2010
last_yearTAS=2015
tas=("tas")
((yearP5TAS=$yearTAS+5))
while [ $yearTAS -le $last_yearTAS ]
do
   echo $yearTAS $yearP5TAS
     for var in $tas
     do
         DATES=${yearTAS}01010300-${yearP5TAS}01010000
         URL=${root}/${MODEL}/${EXPR}/${VAR}/${TABLE}/${var}/$GRID/$VDATE
         echo $URL
         echo $DATES

         wget -nc -nd -t 3 $URL/${var}_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc
     done

 ((yearTAS+=5)) 
 ((yearP5TAS+=5)) 
 done 







# ONE YEAR at a time
# vars=("rsds tsa")
# while [ $year -le $last_year ]

# FIVE YEARS at a time
#vars=("rsds rsdsdiff tas")
#((yearP5=$year+4))
#while [ $year -le $last_year ]
#do
#    echo $year $yearP5
#
#    for var in $vars
#    do
#        DATES=${year}01010130-${yearP5}12312230
#        URL=${root}/${MODEL}/${EXPR}/${VAR}/${TABLE}/${var}/$GRID/$VDATE/$DATE
#        echo $URL
#        echo $DATES
#        
#        wget -nc -nd -t 3 $URL/${var}_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc
#    done
#
#((year+=5))
#((yearP5+=5))
#done

# do
#     ((yearP20=$year+20))
#     # ((yearP10=$year+10))
#     # ((yearP5=$year+5))
#     # ((yearP1=$year+1))

#     DATES=${year}01010300-${yearP20}01010000
    
#     for var in $vars
#     do
#         filename=${var}_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc
#         wget -nc -nd -np -t 5 $root/$EXPR/$VAR/$TABLE/$var/$GRID/$VDATE/$filename

#     done
#     ((year+=20))
#     # ((year+=10))
#     # ((year+=5))
#     # ((year+=1))
# done
# 10 YEARS at a time
# ((yearP10=$year+10))
# vars=("ua va ta hus ps")

# while [ $year -le $last_year ]
# do
#     echo $year $year10

#     TABLE="6hrLev"
#     DATES=${year}01010600-${yearP10}01010000
#     for var in $vars
#     do
#         wget -nc -nd -np -t 5 ${root}/${EXPR}/${VAR}/${TABLE}/${var}/${GRID}/${DATE}/${var}_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc
#     done

    # TABLE="3hr"
    # DATES=${year}01010300-${yearP10}01010000
    # wget -nc -nd -np -t 5 ${root}/${EXPR}/${VAR}/${TABLE}/huss/gn/${DATE}/huss_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc
    # wget -nc -nd -np -t 5 ${root}/${EXPR}/${VAR}/${TABLE}/tas/gn/${DATE}/tas_${TABLE}_${MODEL}_${EXPR}_${VAR}_${GRID}_${DATES}.nc

#     ((year+=10))
#     ((yearP10=$year+10))
# done


# ONE YEAR at a time, but 4 files per year, 3 months each

# ((yearP1=$year+1))
# while [ $year -le $last_year ]
#     do
#     DATES=${year}01010600-${year}04010000
#     wget -c -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/ua/gn/${DATE}/ua_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${DATES}.nc
#     wget -c -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/va/gn/${DATE}/va_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${DATES}.nc

#     DATES=${year}04010600-${year}07010000
#     wget -c -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/ua/gn/${DATE}/ua_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${DATES}.nc
#     wget -c -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/va/gn/${DATE}/va_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${DATES}.nc

#     DATES=${year}07010600-${year}10010000
#     wget -c -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/ua/gn/${DATE}/ua_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${DATES}.nc
#     wget -c -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/va/gn/${DATE}/va_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${DATES}.nc

#     DATES=${year}10010600-${yearP1}01010000
#     wget -c -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/ua/gn/${DATE}/ua_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${DATES}.nc
#     wget -c -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/va/gn/${DATE}/va_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${DATES}.nc
    
#     ((year+=1))
#     ((yearP1+=1))
# done


# FIVE YEARS at a time
# year=2050
# ((yearP5=$year+5))
# while [ $year -le 2050 ]
# do
# echo $year $yearP5

# wget -nc -nd -t 3 ${root}/${EXPR}/${DATE}/6hrLev/ps/gn/v20200622/ps_6hrLev_CMCC-CM2-SR5_ssp585_r1i1p1f1_gn_${year}01010600-${yearP5}01010000.nc

# ((year+=5))
# ((yearP5+=5))
# done


# TWENTY YEARS at a time
# ((yearP20=$year+20))
# while [ $year -le $last_year ]
# do
#     echo $year $year20

#     # wget -nc -nd -np -t 1 ${root}uas/gn/${DATE}/uas_3hr_${MODEL}_${EXPR}_${VAR}_gn_${year}01010300-${yearP20}01010000.nc
#     # wget -nc -nd -np -t 1 ${root}vas/gn/${DATE}/vas_3hr_${MODEL}_${EXPR}_${VAR}_gn_${year}01010300-${yearP20}01010000.nc
#     # wget -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/ua/gn/${DATE}/ua_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${year}01010600-${yearP20}01010000.nc
#     # wget -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/va/gn/${DATE}/va_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${year}01010600-${yearP20}01010000.nc
#     # wget -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/ta/gn/${DATE}/ta_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${year}01010600-${yearP20}01010000.nc
#     # wget -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/hus/gn/${DATE}/hus_6hrLev_${MODEL}_${EXPR}_${VAR}_gn_${year}01010600-${yearP20}01010000.nc
#     wget -nc -nd -np -t 3 ${root}/${EXPR}/${VAR}/${TABLE}/hus/gn/${DATE}/hus_${TABLE}_MPI-ESM1-2-LR_ssp585_r1i1p1f1_gn_${year}0101-${yearP20}1231.nc
#     ((year+=20))
#     ((yearP20=$year+20))
# done


# FIVE YEARS at a time
# year=2015
# yearP5=2020
# while [ $year -le 2050 ]
# do
# echo $year $yearP5

# wget ${root}uas/gn/v20201113/uas_3hr_HadGEM3-GC31-MM_ssp585_r1i1p1f3_gn_${year}01010300-${yearP5}01010000.nc
# wget ${root}vas/gn/v20201113/vas_3hr_HadGEM3-GC31-MM_ssp585_r1i1p1f3_gn_${year}01010300-${yearP5}01010000.nc

# ((year+=5))
# ((yearP5+=5))
# done



