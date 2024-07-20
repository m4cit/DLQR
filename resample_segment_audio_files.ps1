$train_metadata = (Import-Csv ".\data_and_models\data\train\metadata\metadata.csv")
$segments_path = ".\data_and_models\data\train\audio\transformed\segments\"
$transformed_path = $segments_path.Replace("segments\", "")
$new_metadata_path = ".\data_and_models\data\train\metadata\segments_metadata.csv"

# create dir for segmented audio files
if (-not (Test-Path $segments_path)) {
    if (Test-Path $new_metadata_path) {
        Remove-Item $new_metadata_path -Force
    }
    mkdir $segments_path
}

# dictionary for chapters
$dict_chapters = @{}
foreach ($line in $train_metadata) {
    if (-not ($dict_chapters.Contains($line.'chapter num'))) {
        $dict_chapters.Add($line.'chapter num', $line.'chapter name')
    }
}

# dictionary for reciters (reciter name being the key this time)
$dict_reciters = @{}
foreach ($line in $train_metadata) {
    if (-not ($dict_reciters.Contains($line.'folder'))) {
        $dict_reciters.Add($line.'folder', $line.'reciter id')
    }
}

# create new metadata for segmented files
if (-not (Test-Path $new_metadata_path -PathType leaf)) {
    $segments_metadata = New-Item -ItemType File $new_metadata_path
    "folder,chapter num,chapter name,reciter id,file path" | Out-File $segments_metadata -Append
}
else {
    $segments_metadata = $new_metadata_path
}

$reciters = foreach($line in $train_metadata) {
    @($line.folder)
}

# removing duplicates / making a set
$reciters = $reciters | Select-Object -Unique

# create dir for resampled / transformed audio files
if (-not (Test-Path $transformed_path)) {
    mkdir $transformed_path
}

# create dirs for every reciter
foreach ($reciter in $reciters) {
    if (-not (Test-Path "$($transformed_path)$($reciter)")) {
        mkdir "$($transformed_path)$($reciter)"
    }
    if (-not (Test-Path "$($segments_path)$($reciter)")) {
        mkdir "$($segments_path)$($reciter)"
    }
}

Write-Host "`r`nResampling and transforming files..." -ForegroundColor "Blue"
Start-Sleep -Seconds 4

# resampling and transforming audio files (stereo to mono)
foreach ($sample in $train_metadata) {
    $input_dir =  ".\data_and_models\data\train\audio\$($sample.folder)\$('{0:d3}' -f [int]$sample.'chapter num').mp3"
    $output_dir = ".\data_and_models\data\train\audio\transformed\$($sample.folder)\$('{0:d3}' -f [int]$sample.'chapter num')"

    if (-not (Test-Path "$($output_dir).wav" -PathType leaf)) {
        $num_channels = ffprobe -i $input_dir -select_streams a:0 -show_entries format=nb_streams
        $sample_rate = ffprobe -i $input_dir -select_streams a:0 -show_entries stream=sample_rate
        
        # check whether the file is mono or stereo (-ac option), or whether the sample rate is 44100Hz
        # wav format to avoid further downgrading of audio after splitting / segmentation
        if ($num_channels -ne "nb_streams=1" -Or $sample_rate -ne "sample_rate=44100") {
            ffmpeg -i $input_dir -map_metadata -1 -map 0:a -ac 1 -ar 44100 "$($output_dir).wav"
        }
        else {
            Copy-Item $input_dir "$($output_dir).wav"
        }
    }
}

Write-Host "`r`nSplitting audio files into 15 second chunks..." -ForegroundColor "Blue"
Start-Sleep -Seconds 4

# splitting audio files into 15 second chunks
foreach ($sample in $train_metadata) {
    $input_dir =  ".\data_and_models\data\train\audio\transformed\$($sample.folder)\$('{0:d3}' -f [int]$sample.'chapter num').wav"
    $output_dir = ".\data_and_models\data\train\audio\transformed\segments\$($sample.folder)\$('{0:d3}' -f [int]$sample.'chapter num')"
    
    if (-not (Test-Path "$($output_dir)_000.wav" -PathType leaf)) {
        ffmpeg -i $input_dir -map_metadata -1 -f segment -segment_time 15 "$($output_dir)_%03d.wav"
    }
}

Write-Host "`r`nDeleting unusable and excessive files..." -ForegroundColor "Blue"

foreach ($reciter in $reciters) {
    $all_files_del = Get-ChildItem ".\data_and_models\data\train\audio\transformed\segments\$($reciter)"
    for ($i=1; $i -le $all_files_del.Count) {
        if ("$($all_files_del[$i])".EndsWith("_000.wav") -And -not("$($all_files_del[$i-1])".EndsWith("_000.wav"))) {
            Remove-Item ".\data_and_models\data\train\audio\transformed\segments\$($reciter)\$($all_files_del[$i-1])" -Force
        }
        $i++
    }
    Remove-Item ".\data_and_models\data\train\audio\transformed\segments\$($reciter)\$($all_files_del[-1])" -Force
}

Write-Host "`r`nWriting to segmented_metadata.csv..." -ForegroundColor "Blue"

# write to metadata
foreach ($reciter in $reciters) {
    $ch = 1..114 # chapters 1 to 114
    $line_to_add = @()
    $all_files = Get-ChildItem ".\data_and_models\data\train\audio\transformed\segments\$($reciter)"
    foreach ($file in $all_files) {
        foreach ($i in $ch) {
            if ("$($file)".StartsWith("$('{0:d3}' -f $i)_")) {
                $chapter_num = $i
                $chapter_name = $dict_chapters."$($i)"
                $reciter_id = $dict_reciters.$reciter
                $line_to_add += "$($reciter),$($chapter_num),$($chapter_name),$($reciter_id),$($file)"
            }
        }
    }
    # check if line already exists before writing it to the metadata file
    if (-not (Select-String $segments_metadata -Pattern $line_to_add)) {
        $line_to_add | Out-File $segments_metadata -Append
    }
}

Write-Host "`r`nFinished transforming and splitting audio files!`r`n" -Foregroundcolor "DarkGreen"
(New-Object System.Media.SoundPlayer $(Get-ChildItem -Path "$env:windir\Media\tada.wav")).Play()
