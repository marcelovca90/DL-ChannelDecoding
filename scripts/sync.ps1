Start-Sleep -Seconds 600;
while ($true) {
    Get-Date;
    docker cp -a da14d40102b2:/DL-ChannelDecoding/experiments/scenario-9-sbrt-timed /Cysneiros/DL-ChannelDecoding/container/;
    Start-Sleep -Seconds 900;
}
