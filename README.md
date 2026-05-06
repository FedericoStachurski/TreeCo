# TreeCo

TreeCo is a pipeline for estimating tree size classes, carbon stock, and carbon sequestration from citizen-science observations collected through CommuniMap.

The pipeline is designed to work with geolocated tree submissions containing images, short textual descriptions, optional species information, and optional measured tree attributes such as circumference or diameter.

## Pipeline

CommuniMap tree submission  
→ image cleaning and tree crop extraction  
→ RGB + depth tree size estimation  
→ height / trunk / canopy proxy prediction  
→ above-ground biomass estimation  
→ carbon stock and annual sequestration estimation  

## Main modules

- `treeco.data`: CommuniMap loading, cleaning, and manifest creation
- `treeco.models`: image/depth models for tree size prediction
- `treeco.training`: training and evaluation scripts
- `treeco.inference`: batch prediction pipelines
- `treeco.carbon`: allometry and sequestration calculations

## Status

Early development.
