FROM quay.io/qiime2/amplicon:2024.10

RUN pip install --no-cache-dir pandas numpy

WORKDIR /pipeline
COPY 01_create_manifest.py .
COPY 02_pipeline.sh        .
COPY 03_export_biomidas.py .
COPY config.sh             .
COPY metadata_example.csv      .
COPY metadata_example_wang.csv .
COPY srr_to_sample_map.csv .

RUN chmod +x 02_pipeline.sh

ENTRYPOINT ["bash", "02_pipeline.sh"]
