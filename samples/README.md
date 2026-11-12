# Local samples

`samples/test-sample.mkv` is an optional copyright-free clip for a manual analysis check. The video is gitignored and is not part of CI. When the file is in this directory, run:

```bash
pytest -m sample
```

The default `pytest` run does not read it. Synthetic clips generated during the test suite are created in a temporary directory and are not stored here.
