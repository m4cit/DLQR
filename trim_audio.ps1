$correction_data = (Import-Csv ".\data_and_models\data\correction.csv")

if (-not (Test-Path ".\data_and_models\data\train\audio\trimmed")) {
    mkdir ".\data_and_models\data\train\audio\trimmed"
}
if (-not (Test-Path ".\data_and_models\data\test\audio\trimmed")) {
    mkdir ".\data_and_models\data\test\audio\trimmed"
}

# creating an array with links used
$set_dirs = foreach($line in $correction_data) {
    @(".\data_and_models\data\$($line.type)\audio\trimmed\$($line.folder)")
}

# removing duplicates
$set_dirs = $set_dirs | Select-Object -Unique

# creating directories (if they don't exist)
foreach($_dir in $set_dirs) {
    if (-not (Test-Path $_dir)) {
        mkdir $_dir
    }  
}

foreach($sample in $correction_data) {
    $input_dir =  ".\data_and_models\data\$($sample.type)\audio\$($sample.folder)\$($sample.chapter).mp3"
    $output_dir = ".\data_and_models\data\$($sample.type)\audio\trimmed\$($sample.folder)\$($sample.chapter).mp3"
    
    if (-not (Test-Path $output_dir)) {
        ffmpeg -y -ss $sample.starting_time -i $input_dir -map_metadata -1 -map 0:a -c:a copy $output_dir
    }
}

Write-Host "`r`nFinished trimming audio files!`r`n" -Foregroundcolor "DarkGreen"
