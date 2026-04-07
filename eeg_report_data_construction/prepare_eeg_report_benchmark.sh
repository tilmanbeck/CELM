# ============================================================
# Configuration — set all paths here once
# ============================================================
export LD_LIBRARY_PATH="/home/tbeck/miniconda3/envs/CELM/lib:$LD_LIBRARY_PATH"

# Source data (from BDSP, read-only)
data_path='/home/tbeck/data/heedb/neurology_reports'       # contains I0001_Neurology_Reports CSV + year folders with .txt
notes_path='/home/tbeck/data/heedb/neurology_reports'       # where unzipped year/<report>.txt files live
HEEDB_metadata_path='/home/tbeck/data/heedb/HEEDB_Metadata' # contains HEEDB_patients.csv, S0001_EEG__reports_findings.csv
recordings_s3_path='/home/tbeck/data/heedb/EEG'             # S3 (or local) base path for EEG recordings

# Generated data (outputs from this pipeline)
save_path='/home/tbeck/repos/CELM/dataset/processed_reports' # step 1 output: LLM-extracted reports
output_path='/home/tbeck/repos/CELM/dataset/matched_reports' # steps 2-4 output: matched EEG-report pairs

# Pipeline parameters
site='S0001'
model_name='meta-llama/Meta-Llama-3-8B-Instruct'
device=0
num_repetitions=5
overwrite_existing_reports=False
start_index=0
# end_index=2000

# Short model name (used for folder naming by step 1)
if [[ "$model_name" == */* ]]; then
    model_short_name="${model_name##*/}"
else
    model_short_name="$model_name"
fi

# ============================================================
# Step 1: Extract and format neurology reports using LLM
# ============================================================
python extract_and_format_neurology_reports_1.py \
    --data_path "$data_path" \
    --save_path "$save_path" \
    --notes_path "$notes_path" \
    --site_id_filter "$site" \
    --HEEDB_patients_path "$HEEDB_metadata_path" \
    --model_name "$model_name" \
    --device $device \
    --num_repetitions $num_repetitions \
    --overwrite_existing_reports $overwrite_existing_reports \
    --start_index $start_index \
    --load_in_4bit
    # --end_index $end_index

# ============================================================
# Step 2: Match reports with recordings, download from AWS S3
# ============================================================
python match_reports_with_recordings_2.py \
    --site "$site" \
    --heedb_metadata_path "$HEEDB_metadata_path" \
    --recordings_data_path "$recordings_s3_path" \
    --save_path "$save_path" \
    --output_path "$output_path" \
    --model_name "$model_short_name"

# ============================================================
# Step 3: Preprocess the EEG recordings
# ============================================================
python preprocess_eeg_3.py \
    --site "$site" \
    --output_path "$output_path"

# ============================================================
# Step 4: Create description dataframe
# ============================================================
python create_description_df_4_eff.py \
    --site "$site" \
    --num_workers 8 \
    --heedb_metadata_path "$HEEDB_metadata_path" \
    --save_path "$save_path" \
    --output_path "$output_path" \
    --model_name "$model_short_name"
