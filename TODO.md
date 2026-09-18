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

[ ] Replay: update to V2 edition
[ ] Multi-rules: add the rules choices into app
[ ] AI intelligence: default intelligence selection; more detailed AI settings
[ ] Opening Library: using the CubeSprite V4 models to build up the opening library to speed up the AI response in the opening stage and provide an insight in tutorial mode
[ ] Tutorial Mode (may reconstruct the replay mode)

[ ] Hint: Reconstruct! provide top-k advice in real time with much more mcts sims
[ ] Game board: can access the detailed rules and AI settings during the game  