# AI-assisted development disclosure

Dosling scaler was developed with extensive assistance from **OpenAI ChatGPT/Codex**, under the direction of **Peter Balogh**. Assistance included algorithm proposals, source implementation and refactoring, debugging, construction of tests and measurement tools, interpretation of local experiments and writing documentation.

This is an AI-assisted project, not a claim of independently human-authored or independently audited implementation. Peter supplied goals, feedback and publication decisions. Human feedback, numeric scores and visual judgments are different evidence types; automated metrics are not presented as user preferences. The earlier proposed 100-patch blind study was not completed and is not claimed as completed here.

The shipped scaler is ordinary native code with compact learned/authored lookup tables and analytic geometry rules. It does not call ChatGPT, OpenAI or any other remote model at runtime. Python and offline fitting tools are development tools, not required to scale frames. No API credentials are included or required.

Private local development used game-image comparisons and feedback. The game screenshots, game payloads, private comments and conversation transcripts are not part of this repository. The complete historical dataset and experiment sequence are not published, so this release does not claim reproducibility of the original training process. The supplied model can be inspected, edited losslessly and rendered using the published source.

AI involvement does not change the provenance or LGPL obligations of HQ3x-derived code. Upstream credit, notices and modification history remain explicit. Test results describe the checked configurations only; they do not establish general superiority, a security audit or complete App Store compliance.
