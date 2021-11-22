# JOSS publication

After installing Docker, you can compile the paper like so ([docs](https://joss.readthedocs.io/en/latest/submitting.html#docker)):
```bash
# from repository root
docker run --rm --volume $PWD/joss_paper:/data --user $(id -u):$(id -g) \
--env JOURNAL=joss openjournals/paperdraft
```

You may have some warnings, but probably fine if the PDF generates.
```bash
...
Output written on /tmp/tex2pdf.-00cc036c02be466c/input.pdf (7 pages).
...
```

A PDF called `paper.pdf` should appear in this directory.