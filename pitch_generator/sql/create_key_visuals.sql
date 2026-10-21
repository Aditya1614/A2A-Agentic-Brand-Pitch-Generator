CREATE OR REPLACE TABLE bwg.key_visuals AS
SELECT
  campaign,
  concept,
  OBJ.FETCH_METADATA(
    OBJ.MAKE_REF(uri, 'asia-southeast2.pitch-connection')
  ) AS key_visual
FROM UNNEST([
  STRUCT('cats' AS campaign, 'Flying skateboards for cats' AS concept,
         'gs://qwiklabs-gcp-03-481bc86ad8f6-bwg/key-visuals/cats.png' AS uri),
  STRUCT('bike', 'A commuter bike built for rainy cities',
         'gs://qwiklabs-gcp-03-481bc86ad8f6-bwg/key-visuals/bike.png')
]);
