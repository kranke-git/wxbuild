# kranke - June 2026
# Script to upload locations to svante server using scp
# Bash

# Input begins
datadir='./epwdata'
#locations=( 'Manchester_ENG' 'Worcester_MA' 'Pittsfield_MA' 'Singapore__Singapore' )
locations=('Seattle_WA_USA')
svantedir='pgiani@svante6.mit.edu:/home/pgiani/public_html/wxbuild_data'
# Input ends
for location in "${locations[@]}"; do
    scp -r "$datadir/$location" "$svantedir/"
done
