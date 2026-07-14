pearmut bake-existing \
    humeval/campaigns/main_enja_cESA_phase1.json \
    --progress humeval/progress/main_enja_mock.json \
    --annotations humeval/annotations/main_enja.json \
    --output humeval/static/ \
;

# test run
cd humeval/static/
python3 -m http.server 8000

# add this to the url
# ?baked&campaign_id=main_cESA_phase1_v2_enja

# make assets hosted locally
for file in humeval/static/api/main_cESA_phase1_v2_enja/*.json; do
    sed -i 's|https://vilda.net/t/wmt25/en/||g' "$file"
done