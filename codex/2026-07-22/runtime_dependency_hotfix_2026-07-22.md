# Release 4 runtime dependency hotfix

Base commit: `b118dc486a36ebbd75246c9b7d405b61d9b1624f`

The accepted Release 4 environment gate did not import the pinned author
runtime augmentation modules. Bounded smoke runs exposed missing `nlpaug` in
Patch-Mix, PAFA, SG-SCL, and MVST. ADD-RSC first exposed missing `cv2`; direct
author-loader validation then exposed `cmapy` and the same `nlpaug` import.

The hotfix pins `nlpaug==1.1.11` and its resolved dependency set
(`gdown==6.1.0`, `beautifulsoup4==4.15.0`, `soupsieve==2.9.1`) in all five
Linux environments. ADD-RSC also receives the OpenCV/cmapy combination already
validated by Release 4 for MVST: `opencv-python-headless==4.11.0.86`,
`opencv-python==4.11.0.86`, and `cmapy==0.6.6`.

The first post-hotfix Patch-Mix smoke then reached its author data utility,
which directly imports both `cv2` and `cmapy`. Patch-Mix therefore receives the
same already validated exact OpenCV/cmapy set. PAFA and SG-SCL do not import
these modules and remain unchanged.

Primary candidate artifact hashes:

- `nlpaug-1.1.11-py3-none-any.whl`: `01d3befce09e46cb7d990839e0b7dd80ba3e991485f772e678d329ffeb97fd80`
- `opencv-python-headless-4.11.0.86`: `0e0a27c19dd1f40ddff94976cfe43066fbbe9dfbb2ec1907d66c19caef42a57b`
- `opencv-python-4.11.0.86`: `6b02611523803495003bd87362db3e1d2a0454a6a63025dc6658a9830570aa0d`
- `cmapy-0.6.6.tar.gz`: `cb52a6b3057c49a146fb0964b8302f2fb7d61dfe6ae6de1a98b636aace805255`

Server evidence is under
`result/runtime_hotfix_20260722_141512/`. The CUDA 11.8 scratch environment
passed imports for all four affected author augmentation modules. The CUDA 12.1
scratch environment passed the ADD-RSC author loader import. Both passed strict
`pip check`, CUDA metadata, NVIDIA L40 availability, and a finite CUDA kernel.

This change does not alter model code, training code, data, split,
preprocessing, loss, hyperparameters, checkpoint identity, metrics, or
selection policy. Existing Release 4 environments are incrementally repaired
only for bounded server validation and are explicitly receipted as such.

Verified incremental-repair receipt hashes:

- Patch-Mix: `eba799fbe8b66ec72ce0b1cddadb63be7cb29bf131972e9901bb05e748b800f8`
- PAFA: `feaab94f8b0e764912f4a233b497975cfe75c2aeed2f02358326ac9303097258`
- SG-SCL: `5f8c738f8e6cf5206972a7120e685fe58b9888ca5b8aa25ada2fed49fc44e1f8`
- MVST: `12467a02d92d9216473aabfe29f708d3739000a4f511b41501735f55ae7e9dd3`
- ADD-RSC: `107cd5ba5ec53c0de67e1661ac875e493cb0129588a4db97d1b3229ac89e4015`

Secondary Patch-Mix OpenCV/cmapy repair receipt:
`47f1d2b14335ee33e23eb95625e9adfc907b986e44b9b837180be82fe28f9999`.
