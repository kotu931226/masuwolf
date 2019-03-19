$INPUT_DATA_NAME = "src_wolf_320_pad.csv"
$LABEL_DATA_NAME = "tgt_wolf.csv"
$EPOCHS = 25
# $INPUT_DATA_NAME = $null
# $LABEL_DATA_NAME = $null
# $EPOCHS = 200

# set BUCKET_NAME
if ($global:BUCKET_NAME -eq $null){ $global:BUCKET_NAME = Read-Host("input BUCKET_NAME") }
else {
    $DIR = Read-Host("BUCKET_NAME is " + $global:BUCKET_NAME + "? [Y/n]")
    if ($DIR -eq "n"){ $global:BUCKET_NAME = Read-Host("input BUCKET_NAME") }
}
# set REGION
if ($global:REGION -eq $null){ $global:REGION = Read-Host("input REGION") }
else {
    $DIR = Read-Host("REGION is " + $global:REGION + "? [Y/n]")
    if ($DIR -eq "n"){ $global:REGION = Read-Host("input REGION") }
}
# set INPUT_DATA_NAME
if (($global:INPUT_DATA_NAME -eq $null) -And ($INPUT_DATA_NAME -eq $null)){
    $global:INPUT_DATA_NAME = Read-Host("input INPUT_DATA_NAME")
}
else {
    if ($INPUT_DATA_NAME -ne $null){
        $global:INPUT_DATA_NAME = $INPUT_DATA_NAME
    }
    $DIR = Read-Host("INPUT_DATA_NAME is " + $global:INPUT_DATA_NAME + "? [Y/n]")
    if ($DIR -eq "n") { $global:INPUT_DATA_NAME = Read-Host("input INPUT_DATA_NAME") }
}
# set LABEL_DATA_NAME
if (($global:LABEL_DATA_NAME -eq $null) -And ($LABEL_DATA_NAME -eq $null)){
    $global:LABEL_DATA_NAME = Read-Host("input LABEL_DATA_NAME")
}
else {
    if ($LABEL_DATA_NAME -ne $null){
        $global:LABEL_DATA_NAME = $LABEL_DATA_NAME
    }
    $DIR = Read-Host("LABEL_DATA_NAME is " + $LABEL_DATA_NAME + "? [Y/n]")
    if ($DIR -eq "n") { $global:LABEL_DATA_NAME = Read-Host("input LABEL_DATA_NAME") }
}

$BUCKET_PATH = "gs://" + $global:BUCKET_NAME
$INPUT_DATA = $BUCKET_PATH + "/data/" + $global:INPUT_DATA_NAME
$LABEL_DATA = $BUCKET_PATH + "/data/" + $global:LABEL_DATA_NAME
$NOW_TIME = Get-Date -UFormat "%Y_%m_%d_%H_%M"
$JOB_NAME = "training_" + $NOW_TIME
$OUTPUT_PATH = $BUCKET_PATH + "/" + $JOB_NAME
$ARCHIVE_PATH = $BUCKET_PATH + "/training_2000_00_00_00_00/classify.pt"

gcloud ml-engine jobs submit training $JOB_NAME `
--runtime-version 1.10 --python-version 3.5 `
--module-name trainer.task --package-path trainer/ `
--region $REGION --job-dir $ARCHIVE_PATH --config config.yaml `
-- `
--input-data $INPUT_DATA --label-data $LABEL_DATA `
--archive-path $ARCHIVE_PATH --epochs $EPOCHS `
# --input-model $INPUT_MODEL_PATH
