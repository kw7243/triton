# Absorbable-symmetry v2 E0 baseline overlay

This directory preserves the experiment files executed from the captain-owned retained QuaRot source. Apply the contents of `e0/` at the `e0/` path of that repository; the files are not part of Triton's runtime.

The verified retained source was clean at commit `4bc8433528cec54e6b0c32ce94bd755f80303e33`, tree `c6892dfc0662f8f74952b51566427d92fc9afe8e`. The immutable execution stage has the same identity. The local object store no longer contains the older recorded commit `93e5c01e5bf6bc9985a7774361258c7aaa42e9c4`, so this overlay was reconstructed byte-for-byte from the retained source instead of importing unverifiable history.

Durable records establish that the missing commit added `--no-rc` to the existing micromamba create and install operations. Preserved remote history carries that repair at `079d8737fe3e55abd0c9bcc0131b33dd4d7eb116`; the executed source later isolated all preflight state at `4bc8433528cec54e6b0c32ce94bd755f80303e33`. The denied earlier transport attempt copied no bytes and changed no remote Git state.

The terminal r3 record is `artifacts/e0-preflight-r3-20260914T182817Z-c0865a-terminal.json`, SHA-256 `1622c1c9af7e823044fc55c1ca62ea9a4a92b3ea346fb2aec8aab8ea765c8649`.

The retained record-only update is commit `24be8d71b2733908723262ba244c2c33ed2a06d5`, tree `ac504f5708c918e1c0b42e47182dc6123e54cd6d`, on `fm/absorbable-symmetry-v2-e0-preflight-r3`. The experiment source and immutable stage remain the earlier `4bc8433528cec54e6b0c32ce94bd755f80303e33` identity above.
