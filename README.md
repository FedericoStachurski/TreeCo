***TreeCo***

TreeCo is a pipeline for estimating tree size classes, carbon stock, and carbon sequestration from citizen-science observations collected through CommuniMap.

The pipeline works with geolocated tree submissions containing images, descriptions, optional species information, and optional tree measurements such as height and circumference.

**Pipeline**

CommuniMap tree submission  
→ image cleaning and tree extraction  
→ RGB, segmentation, and depth processing  
→ height and DBH prediction  
→ carbon stock estimation  
→ annual sequestration estimation  
→ mapping  

**Status**

Early development.

---

**Setup and Data Preparation**

**Download Required Models**

`treeco-download-models`

See available options with:

`treeco-download-models --help`

**Build the Image Dataset**

Processes CommuniMap images using GroundingDINO, SAM, and Depth Anything.

`treeco-build-dataset \
  --raw_data "$RAW_DATA" \
  --out_root /home/fss6k/TreeCo/data/treeco_datasets`

---

**Height Models**

**Image-Based Height Model**

Example using ResNet18:

`treeco-train-height \
  --dataset_path "$DATASET" \
  --out_dir "$MODELS" \
  --run_name height_resnet18_rgb_sam3 \
  --backbone resnet18 \
  --input_mode rgb_sam3 \
  --image_source full \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --dropout_rate 0.3 \
  --val_size 0.2 \
  --criterion cross_entropy \
  --scheduler cosine`

**Height with Images, Species, and DBH**

Combines image information with species and DBH.

`treeco-train-height-with-species-dbh \
  --dataset_path "$DATASET" \
  --out_dir "$MODELS" \
  --run_name height_resnet18_rgb_sam3_species_dbh \
  --backbone resnet18 \
  --input_mode rgb_sam3 \
  --image_source full \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --dropout_rate 0.3 \
  --val_size 0.2`

**Tabular Height Model with DBH and Species**

Predicts tree height class using DBH and species information.

`treeco-train-height-with-DBHspecies-tab \
  --dataset_path "$DATASET" \
  --out_dir "$MODELS" \
  --run_name height_tabular_dbh_species \
  --val_size 0.2`

This approach is useful when the full tree is not visible in the image.

---

**Diameter / DBH Models**

**DBH Regression**

`treeco-train-width \
  --dataset_path "$DATASET" \
  --out_dir "$MODELS" \
  --run_name dbh_resnet18_rgb_sam3 \
  --backbone resnet18 \
  --input_mode rgb_sam3 \
  --image_source full \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --dropout_rate 0.3 \
  --val_size 0.2 \
  --scheduler cosine`

**DBH Regression with Species**

`treeco-train-width-species \
  --dataset_path "$DATASET" \
  --out_dir "$MODELS" \
  --run_name dbh_resnet18_rgb_sam3_species \
  --backbone resnet18 \
  --input_mode rgb_sam3 \
  --image_source full \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-3 \
  --dropout_rate 0.3 \
  --val_size 0.2`

**DBH Classification**

Example using four DBH classes:

- 0–30 cm
- 30–60 cm
- 60–100 cm
- 100+ cm

`treeco-train-width-classification \
  --dataset_path "$DATASET" \
  --out_dir "$MODELS" \
  --run_name DBHclass4_resnet18_gray_sam3_overlay_tapeOnly \
  --backbone resnet18 \
  --input_mode gray_sam3_overlay \
  --image_source full \
  --batch_size 16 \
  --epochs 20 \
  --lr 3e-5 \
  --weight_decay 1e-3 \
  --dropout_rate 0.3 \
  --val_size 0.2 \
  --criterion cross_entropy \
  --scheduler cosine \
  --no_estimates \
  --no_species \
  --dbh_class_scheme coarse`

Remove `--no_species` to include species.

Remove `--no_estimates` to include estimated DBH observations.

---

**Training Diagnostics**

**Plot a Training Run**

`treeco-plot-training \
  --run_path /home/fss6k/TreeCo/models/YOUR_RUN_DIRECTORY`

**Plot Tree-Level Aggregated Results**

`treeco-plot-training-aggregate \
  --run_path /home/fss6k/TreeCo/models/YOUR_RUN_DIRECTORY`

---

**Inference**

**Infer Tree Height**

`treeco-infer-heights \
  --raw_data "$RAW_DATA" \
  --dataset_dir "$DATASET" \
  --run_path /home/fss6k/TreeCo/models/YOUR_HEIGHT_MODEL`

**Infer Tree DBH**

`treeco-infer-widths \
  --raw_data "$RAW_DATA" \
  --dataset_dir "$DATASET" \
  --run_path /home/fss6k/TreeCo/models/YOUR_DBH_MODEL`

**Infer DBH with Species**

`treeco-infer-widths-species \
  --raw_data "$RAW_DATA" \
  --dataset_dir "$DATASET" \
  --run_path /home/fss6k/TreeCo/models/tree_dbh_models/YOUR_DBH_MODEL`

---

**Combine Predictions**

Combine height and DBH outputs:

`treeco-combine-height-width \
  --height_file PATH_TO_HEIGHT_RESULTS \
  --width_file PATH_TO_DBH_RESULTS \
  --out_file treeco_combined_predictions.csv`

See all options with:

`treeco-combine-height-width --help`

---

**Carbon Estimation**

`treeco-estimate-carbon \
  --input treeco_combined_predictions.csv \
  --output treeco_carbon_estimates.csv`

See available options with:

`treeco-estimate-carbon --help`

---

**Mapping**

**Create a TreeCo Map**

`treeco-make-map \
  --input treeco_combined_predictions.csv \
  --output treeco_map.html`

**Create a Map from Raw CommuniMap Data**

`treeco-rawdata-make-map \
  --raw_data "$RAW_DATA" \
  --output communimap_trees.html`

---

**General Prediction**

`treeco-predict \
  --input PATH_TO_INPUT_DATA \
  --model PATH_TO_MODEL \
  --output predictions.csv`

For any command, use `--help` to see the available options.

For any TreeCo command, use:

treeco-COMMAND --help

to view the available arguments.
