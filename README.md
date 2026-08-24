# modBBarolo

## BBarolo dependency (this branch)

This branch (`feat/iseed-sampling`) adds seed-marginalization support
(`Sampler(..., sample_iseed=True)`), which requires the `Galfit_getModel_seeded`
binding added in [lucadimascolo/Bbarolo](https://github.com/lucadimascolo/Bbarolo)
(a fork of [editeodoro/Bbarolo](https://github.com/editeodoro/Bbarolo)). Build/install
`pyBBarolo` from that fork rather than upstream before using this branch.

`main` has no such dependency and works with stock/upstream BBarolo.