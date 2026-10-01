# Route A epoch-2 recovery anchor

- Source: `runs/nomad_pg_ppo_v2_stage1_new/best_sr.pth`
- Frozen copy: `anchors/route_a_epoch002/policy_state.pth`
- SHA-256: `8da0450a0fb0a0f48a5882e1b8be5f773d75d1cc9b9c8a8baf963a58c57679e5`
- Selection rule: lexicographic best `(validation normal SR, SPL)`
- Validation result: SR `0.38`, SPL `0.21493723513859142`
- Selected epoch: `2`

This full `Policy.state_dict()` is the only allowed initialization and
anti-forgetting reference for the conservative Route-A continuation.  The
epoch-6 `latest_train.pth` and epoch-7 `post_update.pth` are intentionally not
continuation inputs.
