# TODO list

Something I wish I can done in future...

## 26/08-09 

### Model Side

#### Train CubeSprite V4 and explore the new rules for balance competition

[ ] Stage 1: using the new training stack to scale up the CubeSprite V3 architecture series models
[ ] Stage 2: experient different model architectures to develop 3 different sizes of CubeSprite V4 models: 
    Flash: for instant response on CPU (within 3s at 512 sims)
    Balanced: for normal usage on CPU (within 10s at 512 sims)
    Pro: for high performance, recommend run on GPU (no responsibility on CPU) *(development may be limited by the GPU resource and cannot be done for now)*
[ ] Stage 3: build up **Multi-rules model** to support different rules for the competition, and explore the new rules for balance competition

[ ] Extra: build a capabale model(s) for the **Tutorial Mode** 

### APP side (maybe v0.2.0)

[x] Replay: update to V2 edition (Windows v0.2.0-alpha.1)
[x] Multi-rules: add the rules choices into app (Windows v0.2.0-alpha.1)
[ ] AI intelligence: default intelligence selection; more detailed AI settings
[ ] Opening Library: using the CubeSprite V4 models to build up the opening library to speed up the AI response in the opening stage and provide an insight in tutorial mode
[ ] Tutorial Mode (may reconstruct the replay mode)

[ ] Hint: Reconstruct! provide top-k advice in real time with much more mcts sims
[x] Game board: can access the detailed rules and AI settings during the game (Windows v0.2.0-alpha.1)

#### Windows v0.2.0-alpha.1 scope

- [x] V4 Flash (Preview1) registry entry and default model; preserve 256 simulations / temperature 0.4 defaults.
- [x] Five Stage 3 rules, model compatibility routing, forbidden-move UI, and automatic forced passes.
- [x] Seven named intelligence placeholders: 16–1024 simulations / temperature 0.5; Advance opens detailed AI settings.
- [x] Replay v2 save/import/playback/continuation; training sample interoperability.
- [x] In-game Instructions and AI Settings return to the existing game; reject incompatible models during a live game.
- [x] Remove visible app versions; update Instructions without Quick start.
- [ ] Final effort names and future detailed intelligence design (follow-up scope).
