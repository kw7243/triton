# Absorbable-symmetry v2 E0 baseline overlay

This directory preserves the experiment files executed from the captain-owned retained QuaRot source. Apply the contents of `e0/` at the `e0/` path of that repository; the files are not part of Triton's runtime.

The verified retained source was clean at commit `4bc8433528cec54e6b0c32ce94bd755f80303e33`, tree `c6892dfc0662f8f74952b51566427d92fc9afe8e`. The immutable execution stage has the same identity. The local object store no longer contains the older recorded commit `93e5c01e5bf6bc9985a7774361258c7aaa42e9c4`, so this overlay was reconstructed byte-for-byte from the retained source instead of importing unverifiable history.

Durable records establish that the missing commit added `--no-rc` to the existing micromamba create and install operations. Preserved remote history carries that repair at `079d8737fe3e55abd0c9bcc0131b33dd4d7eb116`; the executed source later isolated all preflight state at `4bc8433528cec54e6b0c32ce94bd755f80303e33`. The denied earlier transport attempt copied no bytes and changed no remote Git state.

The terminal r3 record is `artifacts/e0-preflight-r3-20260914T182817Z-c0865a-terminal.json`, SHA-256 `1622c1c9af7e823044fc55c1ca62ea9a4a92b3ea346fb2aec8aab8ea765c8649`.

The retained record-only update is commit `24be8d71b2733908723262ba244c2c33ed2a06d5`, tree `ac504f5708c918e1c0b42e47182dc6123e54cd6d`, on `fm/absorbable-symmetry-v2-e0-preflight-r3`. The experiment source and immutable stage remain the earlier `4bc8433528cec54e6b0c32ce94bd755f80303e33` identity above.

The captain-authorized corrected attempt reused that stage and changed only the submission working directory. Its terminal record is `artifacts/e0-preflight-r3-corrected-20260916T192229Z-fb5cd9-terminal.json`, SHA-256 `2c7a6d508c33986c3a049bc68ea23b0e7f9b37b7b2c397abbd34e1b988b47da6`.

The corresponding retained record-only update is commit `a86aef458fba1ad816d90d28c280cdf447d4f738`, tree `f316c7032b3ede18a44a36b2f77647bb7740244b`, on the same task branch. It does not change the experiment source or frozen R/C contract.

The later registry-recovery lane preserved both failures and tested two unscheduled scratch proofs. The create proof failed after appending its scratch prefix to the AFS registry; it was retained without rollback. The install-into-an-initialized-prefix proof passed with the registry unchanged. Its aggregate record is `artifacts/e0-mamba-registry-recovery-20260916T201036Z-f20ee9.json`, SHA-256 `661df5b135940ee077d3948fe638e2e9457e18a85c76cfdc3a65e4a8fd63c731`.

The single authorized fresh CPU job `1978687` used the preserved stage as `--chdir` but failed before environment creation because the allocation could not read the AFS registry guard. Its terminal record is `artifacts/e0-preflight-r3-registry-20260916T201810Z-dc3fe6-terminal.json`, SHA-256 `947d6a05d22584b90453e5ccf4845e9082ff49983b10742394956c70d3c171c2`. No GPU job or measurement followed.

The retained source-and-record update is commit `e9742e3e026e764dc8b6345e8d15d7137ae7d496`, tree `64c2ca84858fa02a2a16d6ed7670f2116b12ab24`, on the same task branch. It was not pushed or merged.
