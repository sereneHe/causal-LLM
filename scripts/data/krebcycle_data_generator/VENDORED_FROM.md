Vendored sources in this directory come from:

- `krebsgenerator`: https://github.com/petrrysavy/krebsgenerator
  - copied top-level generator Java files from `src/`
- `Chemistry-Engine`: https://github.com/petrrysavy/Chemistry-Engine
  - copied `project/*.java` into `betterChemicalReactions/`
  - copied `commands/challenge.txt`
  - copied `commands/*.txt` into `commands/`

The generator Java files depend on the `betterChemicalReactions` package and on
`challenge.txt` being present in the runtime working directory.
