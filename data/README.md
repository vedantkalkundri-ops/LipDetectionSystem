# Dataset layout

Use this structure for reproducible training:

- `data/raw/{speaker_id}/{phrase_id}/*.mp4`
- `data/metadata.csv`
- `data/processed/{train|val|test}/*.npz`

## Bootstrap target

For initial CPU training:

- 50 phrases
- 30 clips per phrase
- 2 speakers minimum

This gives a usable baseline. Expand to 80-120 clips per phrase to improve stability.
