# Actor/Pico build inputs

`pico_binding/` contains the exact XRoboToolkit binding sources from the original
deployment snapshot, including its custom `get_body_pose_snapshot` implementation,
the snapshot parser, SDK declarations, nlohmann headers, and original binding
license. Its source README is retained for provenance; use this repository's
`scripts/build_pico_arm64.sh` for the current paths and build command.

`double-conversion/` contains the source/header files needed to rebuild the
private Pico service dependency, together with its original license. Test data,
old Git state and build outputs from the upstream snapshot are excluded.

The Pico service, its SDK shared library and Qt/plugin dependencies are delivered
separately as a hash-pinned runtime archive under `deploy/actor_s100/runtime/`.
They are copied from the supplied ARM64 deployment snapshot. This archive is not
a source build of the vendor service. Saved device/deployment rows, saved user
settings, demo executables and recording data are excluded. A fresh empty
factory database retains only the vendor schema. Its `setting.ini`
is a fresh loopback/local-service template without saved device identities or
credentials. Other source licenses do not replace vendor binary terms.

The prebuilt Python 3.10 ARM64 binding is also delivered, so a normal deployment
does not require a compiler. `bootstrap_actor_s100.py --build-pico` rebuilds the
binding and double-conversion on S100 from these tracked sources; it requires
`g++` and the matching Python development headers on the target OS.

`motion_stream_bridge/` contains the unchanged Vita02A GMR solver, geometry
helpers, coordinate transform, JSON tuning and mesh-free robot XML from commit
`763f5607c3f0accb61a415cfd676af54b72e1232`. `SOURCE.json` records every original
file hash. No Aorta transport, action requester or bridge service is included.
GMR's upstream MIT license is retained as `GMR-LICENSE`; original file comments
retain the bridge modifications' provenance.
