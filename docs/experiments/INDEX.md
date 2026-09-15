# Experiment doc pages

One row per major experiment. Every published page exists as a file in this
repository first; the local file is the source of truth and the artifact URL
points at the same content. To update a page, edit the local file, rebuild, and
republish to the same URL.

| Date | Experiment | Local file | Artifact URL |
|------|------------|------------|--------------|
| 2026-09-15 | Approach and closure admissibility for three RH56 grasp strategies and GraspGen-X (support-plane geometry measured, no fitted parameter; paper Fig. 5 grid; README lists the draft's number errata) | [paper/9-8_wxie/20260908_strategy_reachability/20260908-strategy_reachability.html](../../paper/9-8_wxie/20260908_strategy_reachability/20260908-strategy_reachability.html) | https://claude.ai/code/artifact/a2424428-921c-496b-8e47-3a660cfa051c |
| 2026-09-13 | Specification-driven QP closure, without analytical tip targets (iteration budget, tolerance-matched rate, wrist roughness, videos) | [paper/9-8_wxie/20260908_qp_antipodal_closure/20260908-qp_antipodal_closure.html](../../paper/9-8_wxie/20260908_qp_antipodal_closure/20260908-qp_antipodal_closure.html) | https://claude.ai/code/artifact/cff299c5-64c1-41f9-b979-8e07687ae84b |

## Conventions

- Date-prefix the page filename (`20260908-<slug>.html`), not just the folder,
  so pages sort chronologically in a flat listing.
- Keep analysis code, results and figures beside the page in the same
  timestamped folder; point at repository modules rather than copying them.
- Regenerate a page from its results rather than editing numbers by hand.
